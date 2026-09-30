import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import vgg19
from typing import Union
from util.model_util import cal_gram_matrix
from util.data_util import create_transform


class VGG(nn.Module):
    def __init__(
            self,
            model,
            style_weight: float,
            slice1: nn.Sequential,
            slice2: nn.Sequential,
            slice3: nn.Sequential,
            slice4: nn.Sequential,
            prep_info: dict
    ):
        super().__init__()
        self.style_weight = style_weight
        self.model = model
        self.slice1 = slice1
        self.slice2 = slice2
        self.slice3 = slice3
        self.slice4 = slice4

        self.prep_info = prep_info
        self.prep_val = create_transform(is_train=False, **self.prep_info)
        self.prep_train = create_transform(is_train=True, **self.prep_info)

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4

    @classmethod
    def load_model(cls, device, dtype=torch.float32, is_train=False, **kwargs):
        prep_info = kwargs.get('preprocess', None)
        style_weight = kwargs.get('style_weight', 250.0)
        model = vgg19(pretrained=True).features.to(device=device, dtype=dtype)
        if not is_train:
            model.eval()

        slice1 = nn.Sequential()
        slice2 = nn.Sequential()
        slice3 = nn.Sequential()
        slice4 = nn.Sequential()
        for x in range(4):
            slice1.add_module(str(x), model[x])
        for x in range(4, 9):
            slice2.add_module(str(x), model[x])
        for x in range(9, 18):
            slice3.add_module(str(x), model[x])
        for x in range(18, 27):
            slice4.add_module(str(x), model[x])

        vgg = cls(
            model=model,
            style_weight=style_weight,
            slice1=slice1,
            slice2=slice2,
            slice3=slice3,
            slice4=slice4,
            prep_info=prep_info
        )
        return vgg

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_image_embeds(self, image: Union[Image.Image, torch.Tensor], is_train=False):
        preprocess = self.prep_train if is_train else self.prep_val
        image_inputs = preprocess(image).to(device=self.device, dtype=self.dtype)
        if image_inputs.dim() == 3:
            image_inputs = image_inputs.unsqueeze(0)
        layer_embeds = self.forward(image_inputs)
        grams = [cal_gram_matrix(embeds) for embeds in layer_embeds]
        return grams
