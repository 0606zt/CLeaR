from vision_encoder.vanilla_clip import VanillaCLIP
from vision_encoder.robust_clip import RobustCLIP
from vision_encoder.dino import DINO
from vision_encoder.csd_clip import CSD_CLIP
from vision_encoder.inception import Inception
from vision_encoder.vgg import VGG
from image_generator.sdxl import SDXL
from util.loss_func import cal_cos_sim_loss, cal_gram_loss, cal_kid_loss

model_classes = {
    'VanillaCLIP': VanillaCLIP,
    'RobustCLIP': RobustCLIP,
    'CSD_CLIP': CSD_CLIP,
    'DINO': DINO,
    'Inception': Inception,
    'VGG': VGG,
    'SDXL': SDXL
}

model_loss_funcs = {
    'cos_sim': cal_cos_sim_loss,
    'gram': cal_gram_loss,
    'kid': cal_kid_loss
}
