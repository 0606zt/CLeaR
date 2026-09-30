import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import inception_v3
from typing import Union
from util.data_util import create_transform


class Inception(nn.Module):
    def __init__(self, model, prep_info: dict):
        super().__init__()
        self.model = model
        self.prep_info = prep_info
        self.prep_val = create_transform(is_train=False, **self.prep_info)
        self.prep_train = create_transform(is_train=True, **self.prep_info)

    @classmethod
    def load_model(cls, device, dtype=torch.float32, is_train=False, **kwargs):
        prep_info = kwargs.get('preprocess', None)
        model = inception_v3(pretrained=True, transform_input=False).to(device=device, dtype=dtype)
        if not is_train:
            model.eval()
        inception = cls(model=model, prep_info=prep_info)
        return inception

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
        image_embeds = self.model(image_inputs)
        return image_embeds


