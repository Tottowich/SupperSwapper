import os
import sys
sys.path.insert(0, './RealESRGAN/')
import warnings
warnings.filterwarnings("ignore")
import cv2
import time
import torch
import fractions
import numpy as np
from PIL import Image
from typing import Union, Tuple
import torch.nn.functional as F
torch.nn.Module.dump_patches = True
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions,CoverOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import SoftErosion, encode_segmentation_rgb, postprocess
from util.reverse2original import reverse2wholeimage
import face_recognition
from deepface import DeepFace
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import matplotlib.pyplot as plt
import warnings
torch.nn.Module.dump_patches = True
from inpaint import Swap
from FacialFeatureDetection.model import ResNetAttributes 
from FacialFeatureDetection.dataset import translate,unnormalize,transform_image
from basicsr.archs.rrdbnet_arch import RRDBNet
import warnings
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import json
from gfpgan import GFPGANer
COVERS = {"Devin": [325,150,450,320], "MJ": [306,0,550,415],"DualWNBA": [420,220,710,600]} # x0,y0,x1,y1


@torch.no_grad()
class SuperSwapper:
    """
    A class which handles the swapping and super enhancing of facial images.

    """
    def __init__(self,
                    options: Union[TestOptions,CoverOptions],
                    #source:Union[str,np.ndarray,torch.Tensor],
                    #target:Union[str,dict,np.ndarray,torch.Tensor],
                    swap_model_path:str,
                    up_model_path:str,
                    gfpgan_model_path:str,
                    attr_model_path:str,
                    swap_model_name:str='people',
                    det_model_path:str='./insightface_func/models/antelope',
                    det_model_name:str='antelope',
                    det_crop_size:int=640,
                    swap_crop_size:int=224,
                    det_thresh:float=0.0,
                    attr_version:str="resnet34",
                    outscale:float=3.5,
                    up_tile:int=0,
                    up_tile_pad:int=10,
                    up_pre_pad:int=0,
                    up_fp32:bool=False,
                    half:bool = False,
                    gpu_id:int=0,
                    enhance_face:bool=True,
                    enhance_background:bool=False,
                    config_path:str = "data/",
                    match_index:int=None,
                ) -> None:
        self.device=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.opt = options
        #self.find_best_match, targets = self.processSourceAndTarget(source,target)
        self.swap_model_path = swap_model_path
        self.up_model_path = up_model_path
        self.gfpgan_model_path = gfpgan_model_path
        self.attr_model_path = attr_model_path
        self.swap_model_name = swap_model_name
        self.det_model_path = det_model_path
        self.det_model_name = det_model_name
        self.det_crop_size = det_crop_size
        self.swap_crop_size = swap_crop_size
        self.det_thresh = det_thresh
        self.attr_version = attr_version
        self.outscale = outscale
        self.up_tile = up_tile
        self.up_tile_pad = up_tile_pad
        self.up_pre_pad = up_pre_pad
        self.up_fp32 = up_fp32
        self.half = half
        self.gpu_id = gpu_id
        self.enhance_face = enhance_face
        self.enhance_background = enhance_background
        self.config_path = config_path
        self.best_match_index = match_index
        self.DF = DeepFace
        self.tf = transforms.Compose([
                    #transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        self.face_detect_crop = self.initializeFaceDetector() # Initialize the detector which detects and aligns the faces.
        targets = self.loadTargets(self.config_path) # Dictionary of targets
        self.find_best_match = True # Whether to find the best match or not
        self.target_crops,self.target_cut_outs,self.full_covers,self.locations,self.is_embeddings,self.embeddings = self.dictToArrays(targets) # Convert the provided backgrounds to tensors.s
        #self.target_crops,self.target_cut_outs,self.full_covers,self.locations,self.targets_encoding = self._dictToTensor(targets) # Convert the provided backgrounds to tensors.s 
        self.swapper = Swap(opt=options,
                            model_path=swap_model_path,
                            model_name=swap_model_name,
                            device=self.device) # Initialize the swapper.
        self.face_enhancer, self.cover_enhancer = self.initializeSrModel() # Initialize the super-resolution model.
        self.attr_model = ResNetAttributes(version=attr_version,
                                            pretrained=True,
                                            model_path=self.attr_model_path,
                                            device=self.device).eval().to("cpu") # Initialize the attribute detection model. This is used to match the input image to the best target image.

        
    @torch.no_grad()
    def swapAndEnhance(self,source:str) -> Tuple[np.ndarray,np.ndarray]:
        """
        Swaps the source image with the best match and super-enhances it.
        """
        target_crops = self.target_crops
        full_backgrounds = self.full_covers
        target_cut_outs = self.target_cut_outs
        source_crop, _ = self.face_detect_crop.get(readImg(source), self.swap_crop_size)
        source_crop = source_crop[0]
        
        #print(f"source crop size: {source_crop.shape}")
        if self.find_best_match:
            target_crop = self.findTarget(source_crop,target_crops,target_cut_outs) # 
            location = self.locations[self.best_match_index]
            target_cut_out = target_cut_outs[self.best_match_index]
        else:
            raise NotImplementedError("Not implemented yet")
        # x0,y0,x1,y1 = COVERS["Devin"]
        # target_crop = getCutOut(x0,y0,x1,y1,target_img)
        if self.opt.visualize_crop:
            Image.fromarray(source_crop).show(title="Source crop")
            Image.fromarray(target_cut_out).show(title="Target cut out")
           
        final_image_crop,_ = self.swapper.swapImages(self.tf(torch.Tensor(source_crop.transpose((2,0,1))[None,...]).contiguous()/255), np.array(target_cut_out))
        if self.opt.visualize_crop:
            Image.fromarray(final_image_crop).show(title="Merged Faces, no enhancement")
        background = full_backgrounds[self.best_match_index]
        merged_img = inpaintCords(background,final_image_crop,*tuple(location))
        if not self.enhance_face:
            if self.opt.visualize_crop or self.opt.show_final:
                Image.fromarray(merged_img).show(title="Merged Faces, with enhancement and correct background")
            return merged_img,target_cut_out
        self.attr_model.to("cpu")
        self.swapper.model.to("cpu")
        _, _, output = self.face_enhancer.enhance(cv2.cvtColor(merged_img,cv2.COLOR_RGB2BGR), has_aligned=False, only_center_face=False, paste_back=True)
        output_rgb = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
        if self.opt.visualize_crop or self.opt.show_final:
            Image.fromarray(source_crop).show(title="Merged Faces, with enhancement and correct background")
            Image.fromarray(output_rgb).show(title="Merged Faces, with enhancement and correct background")
            #Image.fromarray(increaseContrast(output_rgb,factor=.8)).show(title="Merged Faces, with enhancement and correct background")

        return output_rgb,target_cut_out,merged_img

    def loadTargets(self,config_path:str,config_file_name:str="cover_config.json")->dict:
        """
        Loades the target dictionary from the config file.
        """
        with open(os.path.join(config_path,config_file_name)) as f:
            targets = json.load(f)
        return targets
        
        


    def findTarget(self,
                    source_crop:np.ndarray,
                    target_crops:np.ndarray,
                    target_cut_outs:list)->np.ndarray:
        # print(f"Types: {type(source_crop)}, {type(target_crops)}, {type(target_cut_outs)}")
        # print(f"Shapes: {source_crop.shape}, {target_crops.shape}, {target_cut_outs.shape}")
        """
        Finds the target image that is most similar to the source image in order to generate the best swapping results.
        """
        assert self.find_best_match, "Find_best_match must be True to call findTarget"
        
        if not self.is_embeddings:
            # Stack attr_input and targets for computational efficiency.
            # plt.imshow(target_crops[0]/255)
            # plt.show()
            # print(f"type: {type(target_cut_outs[0])}")
            # print(f"shape: {target_cut_outs[0].shape}")
            # start = time.time()
            # cv2.imshow("Source",source_crop*255)
            # cv2.waitKey(0)
            # embeddings = np.array([self.DF.represent(cv2.cvtColor(cut_out,cv2.COLOR_RGB2BGR),enforce_detection=False) for cut_out in target_cut_outs])
            # end = time.time()
            # print(f"Time to compute embeddings: {end-start:4e}")
            # print(f"Embeddings shape: {embeddings.shape}")
            input_target_stack = torch.cat((self.tf(torch.Tensor(source_crop)[None].permute(0,3,1,2)),self.tf(torch.Tensor(target_crops).permute(0,3,1,2))),dim=0).to(self.device)
            # Find best match:
            self.feature_embeddings = self.attr_model(input_target_stack)
            input_embedding = self.feature_embeddings[0,:]
            target_embedding = self.feature_embeddings[1:,:]
            # If the wanted background match is not provided, select the one which is closest to the input image in the output space, R40.
            self.best_match_index = torch.argmin(torch.sum((input_embedding - target_embedding)**2,dim=1)) if self.best_match_index is None else self.best_match_index
            print(f"Input features: {translate(input_embedding,threshold=0.5)}")
            print(f"Target features: {translate(target_embedding[self.best_match_index],threshold=0.5)}")
            target_img = target_cut_outs[self.best_match_index]
            print(f"Best match index: {self.best_match_index}")

        else:
            # Find best match:
            # input_target_stack = torch.cat((self.tf(torch.Tensor(source_crop)[None].permute(0,3,1,2)),self.tf(torch.Tensor(target_crops).permute(0,3,1,2))),dim=0).to(self.device)
            # self.feature_embeddings = self.attr_model(input_target_stack)
            cv2.imshow("Source",source_crop)
            cv2.waitKey(0)
            try:
                input_embedding = self.DF.represent(cv2.cvtColor(source_crop,cv2.COLOR_RGB2BGR))
            except:
                raise Exception("Source image poor quality")
            target_embedding = self.embeddings
            # If the wanted background match is not provided, select the one which is closest to the input image in the output space, R40.
            self.best_match_index = torch.argmin(torch.sum((target_embedding - input_embedding)**2,dim=1)) if self.best_match_index is None else self.best_match_index
            print(f"Input features: {translate(input_embedding,threshold=0.5)}")
            print(f"Target features: {translate(target_embedding[self.best_match_index],threshold=0.5)}")
            target_img = target_cut_outs[self.best_match_index]
            print(f"Best match index: {self.best_match_index}")
            # input_attr = self.attr_model(self.attr_input)
            # input_attr = input_attr.view(input_attr.size(0), -1)
            # self.target_index = torch.argmin(torch.norm(input_attr - targets, p=2, dim=1))
        return target_img
    def dictToArrays(self,target_dict:dict)->Tuple[np.ndarray,np.ndarray,list,list,bool]:
        """
        Returns a tensor of the target images and a boolean indicating if the target images are embeddings or not
        These target images are cut out of the various covers.
        Also returns:
            list of the locations of the faces in the background image,
            list of the full background images,
            list of the cut out images,
            bool indicating whether the target images are embeddings or not. 
        """
        is_embeddings = True
        covers = target_dict["covers"] # List of cover dict containing "name", "path" to image and "location" of the face in the image. Might contain "emb" for face embeddings.
                                        # "location" is a list of [x0,y0,x1,y1]
        targets = []
        full_covers = []
        target_cut_outs = []
        embeddings = []
        locations = []
        for cover in covers:
            image_path = os.path.join(self.config_path,cover["path"])
            face_loc = cover["location"]
            locations.append(face_loc)
            full_cover = readImg(image_path)
            full_covers.append(full_cover)
            face_cut = getCutOut(face_loc[0],face_loc[1],face_loc[2],face_loc[3],full_cover)
            target_cut_outs.append(face_cut)
            embeddings.append(self.DF.represent(cv2.cvtColor(face_cut,cv2.COLOR_RGB2BGR),enforce_detection=False))
            face_cut,_ = self.face_detect_crop.get(face_cut, self.swap_crop_size) # Crop and align face to crop_size
            targets.append(face_cut[0])
            # print("\n",analysis)
            # plt.imshow(target_cut_outs[-1])
            # plt.show()

        
        return np.array(targets), target_cut_outs,full_covers,locations,is_embeddings,np.array(embeddings) # Return tensor of shape (n_targets,crop_size,crop_size,3) and boolean indicating if the target images are embeddings or not
    def _dictToTensor(self,target_dict:dict)->Tuple[torch.Tensor,list,list,list,bool]:
        """
        Returns a tensor of the target images and a boolean indicating if the target images are embeddings or not
        These target images are cut out of the various covers.
        Also returns:
            list of the locations of the faces in the background image,
            list of the full background images,
            list of the cut out images,
            bool indicating whether the target images are embeddings or not. 
        """
        is_embeddings = False
        covers = target_dict["covers"] # List of cover dict containing "name", "path" to image and "location" of the face in the image. Might contain "emb" for face embeddings.
                                        # "location" is a list of [x0,y0,x1,y1]
        targets = []
        full_covers = []
        target_cut_outs = []
        locations = []
        face_encodings = []
        for cover in covers:
            image_path = os.path.join(self.config_path,cover["path"])
            
            face_loc = cover["location"]
            locations.append(face_loc)
            full_cover = readImg(image_path)
            full_covers.append(full_cover)
            face_cut = getCutOut(face_loc[0],face_loc[1],face_loc[2],face_loc[3],full_cover).astype(np.uint8)
            target_cut_outs.append(face_cut)
            # print(f"Mug shot path: {mug_shot_path}")
            if self.opt.mug_shot:
                mug_shot_path = os.path.join(self.config_path,cover["pathMugShot"])
                mug_shot_img = readImg(mug_shot_path).astype(np.uint8)
                mug_shot_location = cover["mugShotLocation"]
                mug_shot_cut = getCutOut(mug_shot_location[0],mug_shot_location[1],mug_shot_location[2],mug_shot_location[3],mug_shot_img)
                plt.imshow(mug_shot_cut)
                plt.show()
                encodings = face_recognition.face_encodings(mug_shot_img)
            else:
                encodings = face_recognition.face_encodings(face_cut)
            
            face_cut,_ = self.face_detect_crop.get(face_cut, self.swap_crop_size) # Crop and align face to crop_size
            encodings = encodings[0]

            face_encodings.append(encodings)
            print(f"face encoding shape: {encodings.shape}, face_cut shape: {face_cut[0].shape}")
            targets.append(face_cut[0])
        return torch.Tensor(np.array(targets)), target_cut_outs,full_covers,locations,face_encodings
    def _findTarget(self,
                    source_crop:np.ndarray,
                    target_crops:np.ndarray,
                    target_cut_outs:np.ndarray)->np.ndarray:
        """
        Finds the target image that is most similar to the source image in order to generate the best swapping results.
        """
        assert self.find_best_match, "Find_best_match must be True to call findTarget"
        
        input_face_encoding = face_recognition.face_encodings(source_crop.astype(np.uint8))[0]
        face_similarity = face_recognition.face_distance(self.targets_encoding,input_face_encoding)
        self.best_match_index = np.argmin(face_similarity)
        # input_embedding = self.feature_embeddings[0,:]
        # target_embedding = self.feature_embeddings[1:,:]
        # If the wanted background match is not provided, select the one which is closest to the input image in the output space, R40.
        # self.best_match_index = torch.argmin(torch.sum((input_embedding - target_embedding)**2,dim=1)) if self.best_match_index is None else self.best_match_index
        #print(f"Input features: {translate(input_embedding)}")
        #print(f"Target features: {translate(target_embedding[self.best_match_index])}")
        target_img = target_cut_outs[self.best_match_index]
        
        return target_img

    def initializeFaceDetector(self)->Face_detect_crop:
        face_dec = Face_detect_crop(name=self.det_model_name, root=self.det_model_path) # Face detection and cropping model.
        if self.opt.crop_size == 512: # Model specifications
            opt.which_epoch = 550000
            opt.name = '512'
            mode = 'ffhq'
        else:
            mode = 'None'
        face_dec.prepare(ctx_id= 0, det_thresh=self.det_thresh, det_size=(self.det_crop_size,self.det_crop_size),mode=mode)
        return face_dec                          
    def initializeSrModel(self,)->Tuple[GFPGANer,RealESRGANer]:
        """
        Initializes the SR models.
        These models are used to generate the final image.
        They are applied to the merged image with the correct background.
        """
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=self.up_model_path,
            model=model,
            tile=self.up_tile,
            tile_pad=self.up_tile_pad,
            pre_pad=self.up_pre_pad,
            half=not self.up_fp32,
            gpu_id=self.gpu_id)
        face_enhancer = GFPGANer(
            model_path='RealESRGAN/models/GFPGANv1.3.pth',#'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=self.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
        return face_enhancer, upsampler













"""
Helper functions:
"""
def xyxy2yxyx(locations:np.ndarray)->np.ndarray:
    """
    Converts yxyx locations to xyxy locations.
    """
    return locations[::-1]

def TRBL2TLBR(TRBL:np.ndarray)->np.ndarray:
    """
    Converts top right and bottom left coordinates to top left and bottom right coordinates.
    Should be in format xyxy.
    Symmetric, can be used as TLBR2TRBL.
    """
    return np.array([TRBL[2],TRBL[1],TRBL[0],TRBL[3]])


def increaseContrast(img:np.ndarray,factor:float=2.0)->np.ndarray:
    """
    Increases the contrast of the image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    result = np.hstack((img, enhanced_img))
    return result
def inpaintCords(img:Union[torch.Tensor,np.ndarray],inpaint:Union[torch.Tensor,np.ndarray],x0:int,y0:int,x1:int,y1:int)->Union[torch.Tensor,np.ndarray]:
    img[y0:y1,x0:x1,:] = inpaint
    return img
def resizeImage(img:Union[torch.Tensor,np.ndarray],output_size:tuple)->Union[torch.Tensor,np.ndarray]:
    img = cv2.resize(img,output_size)
    return img
def getCutOut(x0:int,y0:int,x1:int,y1:int,img:Union[torch.Tensor,np.ndarray])->Union[torch.Tensor,np.ndarray]:
    """
    Must have x0 < x1, y0 < y1 and HWC fomation.
    """
    return img[y0:y1,x0:x1,:]
def readImg(path:str)->np.ndarray: # RGB images
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    return img
def getHeightWidth(img:Union[torch.Tensor,np.ndarray])->tuple:
    return img.shape[0],img.shape[1]
def saveImg(path:str,img:Union[torch.Tensor,np.ndarray])->None:
    """
    Image must be in BGR format.
    """
    cv2.imwrite(path,img)

def save_cut_out(path,img,x0,y0,x1,y1):
    img[y0:y1,x0:x1,:] = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    saveImg(path,img)
if __name__=="__main__":
    opt = CoverOptions().parse()
    start = time.monotonic()
    swapper = SuperSwapper(options=opt,
                    #source=opt.source,
                    #target=opt.target,
                    swap_model_path=opt.swap_model_path,
                    up_model_path=opt.up_model_path,
                    gfpgan_model_path=opt.gfpgan_model_path,
                    attr_model_path=opt.attr_model_path,
                    swap_model_name=opt.swap_model_name,
                    det_model_path=opt.det_model_path,
                    det_model_name=opt.det_model_name,
                    det_crop_size=opt.det_crop_size,
                    swap_crop_size=opt.swap_crop_size,
                    det_thresh=opt.det_thresh,
                    attr_version=opt.attr_version,
                    outscale=opt.outscale,
                    up_tile=opt.up_tile,
                    up_tile_pad=opt.up_tile_pad,
                    up_pre_pad=opt.up_pre_pad,
                    up_fp32=opt.up_fp32,
                    half=opt.half,
                    gpu_id=opt.gpu_id,
                    enhance_face=opt.enhance_face,
                    enhance_background=opt.enhance_background,
                    match_index=opt.match_index)
    end = time.monotonic()
    print(f"Time taken to initialize: {(end-start):4e} seconds")
    start = time.monotonic()
    
    final_crop, target_img,non_enhanced = swapper.swapAndEnhance(opt.source)
    
    end = time.monotonic()
    print(f"Time taken to swap and enhance: {end-start:.2f}s")
    if opt.save_final:
        path = f"super_res/swapped_{opt.source.split('/')[-1].split('.')[0]}.jpg"
        path_rusty = f"results/swapped_{opt.source.split('/')[-1].split('.')[0]}.jpg"
        print(f"Saving final image to {path}")
        cv2.imwrite(path,cv2.cvtColor(final_crop,cv2.COLOR_RGB2BGR))
        cv2.imwrite(path_rusty,cv2.cvtColor(non_enhanced,cv2.COLOR_RGB2BGR))
# class CoverOptions(BaseOptions):
#     def initialize(self):
#         BaseOptions.initialize(self)
#         self.parser.add_argument('--source', type=str, default='', help='source image path')
#         self.parser.add_argument('--target', type=str, default='', help='target image path')
#         self.parser.add_argument('--swap_model_path', type=str, default='', help='path to the swap model')
#         self.parser.add_argument('--up_model_path', type=str, default='', help='path to the up model')
#         self.parser.add_argument('--gfpgan_model_path', type=str, default='', help='path to the gfpgan model')
#         self.parser.add_argument('--attr_model_path', type=str, default='', help='path to the attribute model')
#         self.parser.add_argument('--swap_model_name', type=str, default='people', help='name of the swap model')
#         self.parser.add_argument('--det_model_path', type=str, default='', help='path to the detection model')
#         self.parser.add_argument('--det_model_name', type=str, default='antelope', help='name of the detection model')
#         self.parser.add_argument('--det_crop_size', type=int, default=640, help='size of the detection crop')
#         self.parser.add_argument('--det_thresh', type=float, default=0.0, help='threshold for detection')
#         self.parser.add_argument('--attr_version', type=str, default='resnet34', help='version of the attribute model')
#         self.parser.add_argument('--outscale', type=float, default=3.5, help='scale of the output')
#         self.parser.add_argument('--up_tile', type=int, default=1, help='tile the input image')
#         self.parser.add_argument('--up_tile_pad', type=int, default=10, help='pad the input image')
#         self.parser.add_argument('--up_pre_pad', type=int, default=0, help='pad the input image')
#         self.parser.add_argument('--up_fp_32', type=bool, default=False, help='use 32 bit floating point')
#         self.parser.add_argument('--half', type=bool, default=False, help='use half precision')
#         self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
#         self.parser.add_argument('--enhance_face', type=bool, default=True, help='enhance face')
#         self.parser.add_argument('--enhance_background', type=bool, default=False, help='enhance background')
#         self.isTrain = False
#         self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        
#         self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
#         self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
#         self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
#         self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
#         self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
#         self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
#         self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
#         self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
#         self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
#         self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
#         self.parser.add_argument("--Arc_path", type=str, default='models/BEST_checkpoint.tar', help="run ONNX model via TRT")
#         self.parser.add_argument("--attr_model_path", type=str, default='FacialFeatureDetection/models/resnet_attributes.pth', help="Path to Attribute detection model")
        
#         self.parser.add_argument("--cover", type=str, default='./crop_224/gdg.jpg', help="Game Cover image path")
#         self.parser.add_argument("--selfie", type=str, default='./crop_224/zrf.jpg', help="Selfie image path")
#         self.parser.add_argument("--pic_specific_path", type=str, default='./crop_224/zrf.jpg', help="The specific person to be swapped")
#         self.parser.add_argument("--multisepcific_dir", type=str, default='./demo_file/multispecific', help="Dir for multi specific")
#         self.parser.add_argument("--video_path", type=str, default='./demo_file/multi_people_1080p.mp4', help="path for the video to swap")
#         self.parser.add_argument("--temp_path", type=str, default='./temp_results', help="path to save temporarily images")
#         self.parser.add_argument("--output_path", type=str, default='./output/', help="results path")
#         self.parser.add_argument('--id_thres', type=float, default=0.03, help='how many test images to run')
#         self.parser.add_argument('--no_simswaplogo', action='store_true', help='Remove the watermark')
#         self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
#         self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')