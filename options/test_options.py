'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-23 17:08:08
Description: 
'''
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--Arc_path", type=str, default='models/BEST_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--pic_a_path", type=str, default='./crop_224/gdg.jpg', help="Person who provides identity information")
        self.parser.add_argument("--pic_b_path", type=str, default='./crop_224/zrf.jpg', help="Person who provides information other than their identity")
        self.parser.add_argument("--pic_specific_path", type=str, default='./crop_224/zrf.jpg', help="The specific person to be swapped")
        self.parser.add_argument("--multisepcific_dir", type=str, default='./demo_file/multispecific', help="Dir for multi specific")
        self.parser.add_argument("--video_path", type=str, default='./demo_file/multi_people_1080p.mp4', help="path for the video to swap")
        self.parser.add_argument("--temp_path", type=str, default='./temp_results', help="path to save temporarily images")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="results path")
        self.parser.add_argument('--id_thres', type=float, default=0.03, help='how many test images to run')
        self.parser.add_argument('--no_simswaplogo', action='store_true', help='Remove the watermark')
        self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
        self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')
        
        self.isTrain = False
class CoverOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--source', type=str, default='', help='source image path')
        self.parser.add_argument('--target', type=str, default='', help='target image path')
        self.parser.add_argument('--config_path', type=str, default='data/', help='path to the swap model')
        self.parser.add_argument('--swap_model_path', type=str, default='', help='path to the swap model')
        self.parser.add_argument('--up_model_path', type=str, default='', help='path to the up model')
        self.parser.add_argument('--gfpgan_model_path', type=str, default='', help='path to the gfpgan model')
        self.parser.add_argument('--attr_model_path', type=str, default='', help='path to the attribute model')
        self.parser.add_argument('--swap_model_name', type=str, default='people', help='name of the swap model')
        self.parser.add_argument('--det_model_path', type=str, default='./insightface_func/models', help='path to the detection model')
        self.parser.add_argument('--det_model_name', type=str, default='antelope', help='name of the detection model')
        self.parser.add_argument('--det_crop_size', type=int, default=640, help='size of the detection crop')
        self.parser.add_argument('--swap_crop_size', type=int, default=224, help='size of the detection crop')
        self.parser.add_argument('--det_thresh', type=float, default=0.0, help='threshold for detection')
        self.parser.add_argument('--attr_version', type=str, default='resnet34', help='version of the attribute model')
        self.parser.add_argument('--outscale', type=float, default=3.5, help='scale of the output')
        self.parser.add_argument('--up_tile', type=int, default=0, help='tile the input image')
        self.parser.add_argument('--up_tile_pad', type=int, default=10, help='pad the input image')
        self.parser.add_argument('--up_pre_pad', type=int, default=0, help='pad the input image')
        self.parser.add_argument('--up_fp32', action='store_true', help='use 32 bit floating point')
        self.parser.add_argument('--half', type=bool, default=False, help='use half precision')
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
        self.parser.add_argument('--enhance_face', action='store_true', help='Use GFPGAN to enhance face')
        self.parser.add_argument('--enhance_background', type=bool, default=False, help='enhance background')
        self.parser.add_argument('--mug_shot', type=bool, default=True, help='Use mug shot to get features of the covers')
        self.parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
        
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--Arc_path", type=str, default='models/BEST_checkpoint.tar', help="run ONNX model via TRT")
        
        self.parser.add_argument("--cover", type=str, default='./crop_224/gdg.jpg', help="Game Cover image path")
        self.parser.add_argument("--selfie", type=str, default='./crop_224/zrf.jpg', help="Selfie image path")
        self.parser.add_argument("--pic_specific_path", type=str, default='./crop_224/zrf.jpg', help="The specific person to be swapped")
        self.parser.add_argument("--multisepcific_dir", type=str, default='./demo_file/multispecific', help="Dir for multi specific")
        self.parser.add_argument("--video_path", type=str, default='./demo_file/multi_people_1080p.mp4', help="path for the video to swap")
        self.parser.add_argument("--temp_path", type=str, default='./temp_results', help="path to save temporarily images")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="results path")
        self.parser.add_argument('--id_thres', type=float, default=0.03, help='how many test images to run')
        self.parser.add_argument('--no_simswaplogo', action='store_true', help='Remove the watermark')
        self.parser.add_argument('--save_final', action='store_true', help='Save Final Cropped image')
        self.parser.add_argument('--show_final', action='store_true', help='Show Final image')

        self.parser.add_argument('--visualize_crop', action='store_true', help='Visualize the cropped images')
        self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
        self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')
        self.parser.add_argument('--match_index', type=int, default=None, help='If specified, match the specific index')
        self.isTrain = False