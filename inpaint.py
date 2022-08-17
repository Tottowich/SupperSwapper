import os
import sys
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
torch.nn.Module.dump_patches = True
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions,CoverOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import SoftErosion, encode_segmentation_rgb, postprocess
from util.reverse2original import reverse2wholeimage
#from deepface import DeepFace
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import matplotlib.pyplot as plt
import warnings
from FacialFeatureDetection.model import ResNetAttributes 
from FacialFeatureDetection.dataset import translate,unnormalize,transform_image
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
THRESH = 0.0
COVERS = {"Devin": [325,150,450,320], "MJ": [306,0,550,415],"DualWNBA": [420,220,710,600]} # x0,y0,x1,y1

class Swap:
    def __init__(self,
                opt,
                model_path: str,
                model_name: str,
                # attr_model_path: str,
                crop_size:int=224,
                attr_version:str="resnet34",
                attribute_threshold:float=0.5,
                device:str="cuda:0",
                ):
        self.opt = opt
        # self.source = source # NumPy array
        # self.target = target # NumPy array
        #self.source = self._totensor(self.source)
        #self.target = self._totensor(self.target)
        self.crop_size = crop_size
        # self.attribute_threshold = attribute_threshold
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.model_attr = ResNetAttributes(opt.attr_model_path,pretrained=True,model_path=opt.attr_model_path,version=attr_version)
        # self.model_attr.eval()
        if self.crop_size == 512:
            self.opt.which_epoch = 550000
            self.opt.name = '512'
            self.mode = 'ffhq'
        else:
            self.mode = 'None'
        # self.transformer_Arcface = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        self.logoclass = watermark_image('./simswaplogo/simswaplogo.png')
        self.model = create_model(self.opt)
        self.model.eval()
        self.spNorm =SpecificNorm()
        self.app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.app.prepare(ctx_id= 0, det_thresh=THRESH, det_size=(640,640),mode=self.mode)

    def _totensor(self,array):
        tensor = torch.from_numpy(array)
        img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)
    @torch.no_grad()
    def swapImages(self,source,target):
        # plt.imshow(self.source)
        # plt.show()
        #img_a_align_crop, _ = self.app.get(source,self.crop_size)

        
        #img_a_align_crop_pil = Image.fromarray(img_a_align_crop[0])
        # img_a_align_crop_pil.resize(size=(512,512)).show(title="Face cropped input image.")
        # plt.imshow(img_a_align_crop_pil)
        # plt.show()
        img_a = source.to(self.device)
        img_id = img_a
        
        #img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(self.device)
        # attr_out = self.model_attr(img_id)
        # attr_out = attr_out.squeeze(0).cpu().numpy()
        # print(f"Predicted Attributes: {translate(attr_out,self.attribute_threshold)}")
        # # plt.imshow(unnormalize(img_a.cpu()))
        # # plt.show()
        # self.model_attr.to("cpu")
        # self.model.to("cpu")
       
        # attr = DeepFace.analyze(opt.selfie,enforce_detection=False,prog_bar=False)
        # text_promt =f"\'{attr['dominant_race']} {attr['gender']} Basketball player {attr['dominant_emotion']} face\'"
        # print(f"Input text promt: {text_promt}")
        # self.model.to("cuda:0")
        # predicted = attr_out
        # predicted[attr_out<0.5] = 0
        # predicted[attr_out>=0.5] = 1
        #self.model.to(self.device)

        # convert numpy to tensor
        # img_id = img_id.to(self.device)

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = self.model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)


        ############## Forward Pass ######################

        # pic_b = self.opt.cover
        # img_b_whole = cv2.imread(pic_b)
        
        img_b_whole = cv2.cvtColor(target,cv2.COLOR_RGB2BGR)
        img_b_align_crop_list, b_mat_list = self.app.get(img_b_whole,self.crop_size)
        
        # detect_results = None
        swap_result_list = []

        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = self._totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].to(self.device)
            swap_result = self.model(None, b_align_crop_tenor, latend_id, None, True)[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        if self.opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None

        final_image, save_path = self._reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, self.crop_size, img_b_whole, self.logoclass, \
            os.path.join(self.opt.output_path, 'result_whole_swapsingle.jpg'), self.opt.no_simswaplogo,pasring_model=net,use_mask=self.opt.use_mask, norm = self.spNorm)
        final_image = final_image[...,[2,1,0]] ## BGR to RGB
        print(' ')

        print('************ Swapped! ************')
        return final_image, save_path












    def _reverse2wholeimage(self,b_align_crop_tenor_list,swaped_imgs, mats, crop_size, oriimg, logoclass, save_path = '', \
                    no_simswaplogo = False,pasring_model =None,norm = None, use_mask = False):

        target_image_list = []
        img_mask_list = []
        if use_mask:
            smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
        else:
            pass

        # print(len(swaped_imgs))
        # print(mats)
        # print(len(b_align_crop_tenor_list))
        for swaped_img, mat ,source_img in zip(swaped_imgs, mats,b_align_crop_tenor_list):
            swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
            img_white = np.full((crop_size,crop_size), 255, dtype=float)

            # inverse the Affine transformation matrix
            mat_rev = np.zeros([2,3])
            div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
            mat_rev[0][0] = mat[1][1]/div1
            mat_rev[0][1] = -mat[0][1]/div1
            mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
            div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
            mat_rev[1][0] = mat[1][0]/div2
            mat_rev[1][1] = -mat[0][0]/div2
            mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

            orisize = (oriimg.shape[1], oriimg.shape[0])
            if use_mask:
                source_img_norm = norm(source_img)
                source_img_512  = F.interpolate(source_img_norm,size=(512,512))
                out = pasring_model(source_img_512)[0]
                parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
                vis_parsing_anno = parsing.copy().astype(np.uint8)
                tgt_mask = encode_segmentation_rgb(vis_parsing_anno)
                if tgt_mask.sum() >= 5000:
                    # face_mask_tensor = tgt_mask[...,0] + tgt_mask[...,1]
                    target_mask = cv2.resize(tgt_mask, (crop_size,  crop_size))
                    # print(source_img)
                    target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask,smooth_mask)
                    

                    target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
                    # target_image_parsing = cv2.warpAffine(swaped_img, mat_rev, orisize)
                else:
                    target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
            else:
                target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
            # source_image   = cv2.warpAffine(source_img, mat_rev, orisize)

            img_white = cv2.warpAffine(img_white, mat_rev, orisize)


            img_white[img_white>20] =255

            img_mask = img_white

            # if use_mask:
            #     kernel = np.ones((40,40),np.uint8)
            #     img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            # else:
            kernel = np.ones((40,40),np.uint8)
            img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            kernel_size = (20, 20)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

            # kernel = np.ones((10,10),np.uint8)
            # img_mask = cv2.erode(img_mask,kernel,iterations = 1)



            img_mask /= 255

            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

            # pasing mask

            # target_image_parsing = postprocess(target_image, source_image, tgt_mask)

            if use_mask:
                target_image = np.array(target_image, dtype=np.float64) * 255
            else:
                target_image = np.array(target_image, dtype=np.float64)[..., ::-1] * 255


            img_mask_list.append(img_mask)
            target_image_list.append(target_image)
            

        # target_image /= 255
        # target_image = 0
        img = np.array(oriimg, dtype=np.float64)
        for img_mask, target_image in zip(img_mask_list, target_image_list):
            img = img_mask * target_image + (1-img_mask) * img
            
        final_img = img.astype(np.uint8)
        if not no_simswaplogo:
            final_img = logoclass.apply_frames(final_img)
        #cv2.imwrite(save_path, final_img)
        return final_img, save_path
def inpaint(img,mask,inpaint):
    img[mask==0] = inpaint
    return img
def inpaint_cords(img,inpaint,x0,x1,y0,y1):
    img[y0:y1,x0:x1,:] = inpaint
    return img
def resize_image(inpaint,size_of_inpaint):
    inpaint = cv2.resize(inpaint,size_of_inpaint)
    return inpaint
def get_cut_out(x0,y0,x1,y1,img):
    return img[y0:y1,x0:x1,:]
def read_img(path):
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    return img
def get_height_width(img):
    return img.shape[0],img.shape[1]
def save_img(path,img):
    cv2.imwrite(path,img)
def save_cut_out(path,img,x0,y0,x1,y1):
    img[y0:y1,x0:x1,:] = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    save_img(path,img)
if __name__=="__main__":
    opt = CoverOptions().parse()
    cover = read_img(opt.cover)
    inpaint_img = read_img(opt.selfie)
    og_height,og_width = get_height_width(cover)
    print("og_height:",og_height,"og_width:",og_width)
    if opt.cover.endswith("over.jpg") or opt.cover.endswith("beard.jpg"):
        # x0 = 325 # Single Cover
        # y0 = 150 # Single Cover
        # x1 = 450 # Single Cover
        # y1 = 320 # Single Cover
        x0,y0,x1,y1 = COVERS["Devin"]
    elif opt.cover.endswith("Dual.jpg"):
        # x0 = 420 # Double Cover
        # y0 = 220 # Double Cover
        # x1 = 710 # Double Cover
        # y1 = 600 # Double Cover
        x0,y0,x1,y1 = COVERS["DualWNBA"]
    elif opt.cover.endswith("ichaela.jpg"):
        # x0 = 306 # Michaela
        # y0 = 0 # Michaela
        # x1 = 550 # Michaela
        # y1 = 415 # Michaela
        x0,y0,x1,y1 = COVERS["Michael"]
    cut_out = get_cut_out(x0,y0,x1,y1,cover)
    height_cut, width_cut = get_height_width(cut_out)
    print("height_cut:",height_cut,"width_cut:",width_cut)
    print(f"Shape of cut_out: {cut_out.shape}")
    # plt.imshow(cut_out)
    # plt.show()
    swapper = Swap(opt,inpaint_img,cut_out,attr_version="resnet34")
    finalImage, save_path = swapper.swapImages()
    plt.imshow(finalImage)
    plt.show()
    print(f"Shape of finalImage: {finalImage.shape}")
    result = inpaint_cords(cover,finalImage,x0,x1,y0,y1)
    plt.imshow(result)
    plt.show()
    
    cv2.imwrite(f"images/result_{opt.selfie.split('/')[-1].split('.')[0]}.jpg",cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
    #x0,y0,x1,y1 = int(635/og_width*cover_image_size),int(265/og_height*cover_image_size),int(900/og_height*cover_image_size),int(555/og_height*cover_image_size)