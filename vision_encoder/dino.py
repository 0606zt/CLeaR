import torch
import torch.nn as nn
from PIL import Image
from typing import Union
from util.data_util import create_transform


class DINO(nn.Module):
    def __init__(self, model, prep_info: dict):
        super().__init__()
        self.model = model
        self.prep_info = prep_info
        self.prep_val = create_transform(is_train=False, **self.prep_info)
        self.prep_train = create_transform(is_train=True, **self.prep_info)

    @classmethod
    def load_model(cls, device, dtype=torch.float32, is_train=False, **kwargs):
        model_name = kwargs.get('model_name', 'dinov3_vitl16')
        model_dir = kwargs.get('model_dir', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        repo_dir_full = kwargs.get('repo_dir', 'github:facebookresearch/dinov3')
        repo_source, repo_dir = repo_dir_full.split(':', 1)
        prep_info = kwargs.get('preprocess', None)
        model = torch.hub.load(
            repo_or_dir=repo_dir,
            model=model_name,
            source=repo_source,
            weights=model_dir
        ).to(device=device, dtype=dtype)
        if not is_train:
            model.eval()
        dino = cls(model=model, prep_info=prep_info)
        return dino

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

