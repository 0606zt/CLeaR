import torch
import torch.nn as nn
from PIL import Image
import open_clip
from typing import Union
from util.data_util import create_transform


class RobustCLIP(nn.Module):
    def __init__(self, model, prep_info: dict):
        super().__init__()
        self.model = model
        self.prep_info = prep_info
        self.prep_val = create_transform(is_train=False, **self.prep_info)
        self.prep_train = create_transform(is_train=True, **self.prep_info)

    @classmethod
    def load_model(cls, device, dtype=torch.float32, is_train=False, **kwargs):
        model_dir = kwargs.get('model_dir', 'hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K')
        prep_info = kwargs.get('preprocess', None)
        model, _, _ = open_clip.create_model_and_transforms(model_dir, device=device)
        model = model.to(dtype=dtype)
        if not is_train:
            model.eval()
        model = cls(model=model, prep_info=prep_info)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def get_text_embeds(self, text: Union[str, list[str]]):
        text_inputs = open_clip.tokenize(text).to(self.device)
        text_embeds = self.model.encode_text(text_inputs)
        return text_embeds

    def get_image_embeds(self, image: Union[Image.Image, torch.Tensor], is_train=False):
        preprocess = self.prep_train if is_train else self.prep_val
        image_inputs = preprocess(image).to(device=self.device, dtype=self.dtype)
        if image_inputs.dim() == 3:
            image_inputs = image_inputs.unsqueeze(0)
        image_embeds = self.model.encode_image(image_inputs)
        return image_embeds

