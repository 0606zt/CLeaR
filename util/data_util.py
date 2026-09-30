import torchvision.transforms as transforms
from timm.data.transforms import MaybeToTensor, str_to_interp_mode
import kornia.augmentation as kaugs
import random


def create_transform(is_train=False, **kwargs):
    resize_size = kwargs.get('resize_size', 256)
    crop_size = kwargs.get('crop_size', 224)
    interpolation_mode = kwargs.get('interpolation_mode', 'bilinear')
    norm_mean = kwargs.get('norm_mean', (0.485, 0.456, 0.406))
    norm_std = kwargs.get('norm_std', (0.229, 0.224, 0.225))
    augment_prob = kwargs.get('augment_prob', 0.7)
    augment_mode = kwargs.get('augment_mode', 'none')
    if is_train:
        preprocess = create_transform_train(
            resize_size=resize_size,
            interpolation_mode=interpolation_mode,
            norm_mean=norm_mean,
            norm_std=norm_std,
            augment_prob=augment_prob,
            augment_mode=augment_mode
        )
    else:
        preprocess = create_transform_val(
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation_mode=interpolation_mode,
            norm_mean=norm_mean,
            norm_std=norm_std
        )
    return preprocess


# note: kornia instead of transforms is used here because differentiable enhancement is required for training
def create_transform_train(
        resize_size=224,
        interpolation_mode='bicubic',
        norm_mean=(0.48145466, 0.4578275, 0.40821073),
        norm_std=(0.26862954, 0.26130258, 0.27577711),
        augment_prob=0.7,
        augment_mode='full'
):
    jitter = kaugs.RandomTranslate(
        translate_x=(-0.05, 0.05),
        translate_y=(-0.05, 0.05),
        p=augment_prob,
        same_on_batch=False
    )
    affine = kaugs.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        padding_mode='border',
        p=augment_prob,
        same_on_batch=False
    )
    color_jitter = kaugs.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=augment_prob,
        same_on_batch=False
    )
    resize = kaugs.Resize(
        size=resize_size,
        resample=interpolation_mode,
        align_corners=False,
        antialias=True
    )
    normalization = kaugs.Normalize(
        mean=norm_mean,
        std=norm_std
    )
    augment_options = [jitter, affine, color_jitter]
    if augment_mode == 'full':
        preprocess = kaugs.AugmentationSequential(
            jitter,
            affine,
            resize,
            color_jitter,
            normalization,
            data_keys=['image']
        )
    elif augment_mode == 'random':
        preprocess = kaugs.AugmentationSequential(
            random.choice(augment_options),
            resize,
            normalization,
            data_keys=['image']
        )
    elif augment_mode == 'none':
        preprocess = kaugs.AugmentationSequential(
            resize,
            normalization,
            data_keys=['image']
        )
    else:
        raise ValueError(f'{augment_mode} is not supported')
    return preprocess


def create_transform_val(
        resize_size=256,
        crop_size=224,
        interpolation_mode='bilinear',
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
):
    to_tensor = MaybeToTensor()
    resize = transforms.Resize(
        size=resize_size,
        interpolation=str_to_interp_mode(interpolation_mode)
    )
    crop = transforms.CenterCrop(crop_size)
    normalization = transforms.Normalize(
        mean=norm_mean,
        std=norm_std
    )
    preprocess = transforms.Compose([
        to_tensor,
        resize,
        crop,
        normalization
    ])
    return preprocess
