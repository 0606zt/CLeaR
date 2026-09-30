import torch
import torch.nn as nn
from PIL import Image
import clip
import copy
from torch.autograd import Function
from typing import Union
from util.model_util import convert_state_dict, init_weights
from util.data_util import create_transform


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CSD_CLIP(nn.Module):
    def __init__(
            self,
            backbone,
            last_layer_style,
            last_layer_content,
            content_proj_head: str,
            prep_info: dict
    ):
        super().__init__()
        self.backbone = backbone
        self.last_layer_content = last_layer_content
        self.content_proj_head = content_proj_head
        self.last_layer_style = last_layer_style

        self.prep_info = prep_info
        self.prep_val = create_transform(is_train=False, **self.prep_info)
        self.prep_train = create_transform(is_train=True, **self.prep_info)

    def forward(self, x, alpha=None):
        feature = self.backbone(x)

        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
        else:
            reverse_feature = feature

        style_feature = feature @ self.last_layer_style
        style_feature = nn.functional.normalize(style_feature, dim=1, p=2)

        if self.content_proj_head == 'custom':
            content_feature = self.last_layer_content(reverse_feature)
        else:
            content_feature = reverse_feature @ self.last_layer_content
        content_feature = nn.functional.normalize(content_feature, dim=1, p=2)
        return feature, content_feature, style_feature

    @classmethod
    def load_model(cls, device, dtype=torch.float32, is_train=False, **kwargs):
        base_model_name = kwargs.get('base_model_name', 'vitl')
        base_model_dir = kwargs.get('base_model_dir', None)
        content_proj_head = kwargs.get('content_proj_head', 'default')
        model_dir = kwargs.get('model_dir', 'CSD-ViT-L-checkpoint.pth')
        prep_info = kwargs.get('preprocess', None)
        if base_model_name == 'vitl':
            if base_model_dir is None:
                clip_model, _ = clip.load('ViT-L/14', device=device)
            else:
                clip_model, _ = clip.load(base_model_dir, device=device)
            backbone = clip_model.visual
            embedding_dim = 1024
            feat_dim = 512
        elif base_model_name == 'vitb':
            if base_model_dir is None:
                clip_model, _ = clip.load('ViT-B/16', device=device)
            else:
                clip_model, _ = clip.load(base_model_dir, device=device)
            backbone = clip_model.visual
            embedding_dim = 768
            feat_dim = 512
        else:
            raise NotImplementedError(f'{base_model_name} is not implemented')
        backbone = backbone.to(dtype=dtype)

        last_layer_style = copy.deepcopy(backbone.proj)
        if content_proj_head == 'custom':
            last_layer_content = ProjectionHead(embedding_dim, feat_dim)
            last_layer_content.apply(init_weights)
        else:
            last_layer_content = copy.deepcopy(backbone.proj)
        last_layer_content = last_layer_content.to(device=device, dtype=dtype)
        backbone.proj = None

        csd_clip = cls(
            backbone=backbone,
            last_layer_style=last_layer_style,
            last_layer_content=last_layer_content,
            content_proj_head=content_proj_head,
            prep_info=prep_info,
        )
        checkpoint = torch.load(model_dir, map_location=device)
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        csd_clip.load_state_dict(state_dict)
        if not is_train:
            csd_clip.eval()
        return csd_clip

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
        _, _, style_embeds = self.forward(image_inputs)
        return style_embeds

