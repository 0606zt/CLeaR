import warnings
from diffusers.utils import logging
warnings.filterwarnings('ignore')
logging.disable_progress_bar()

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from util.config import config
from util.registry import model_classes, model_loss_funcs
from util.model_util import fix_seed, load_models
from style_extract import invert_train


def main(args):
    if args.seed != -1:
        fix_seed(args.seed)

    # load image generators
    print(f'Loading image generators: {args.image_generators}...')
    image_generator_devices = ['cuda:6'] * len(args.image_generators)
    image_generator_dtypes = [torch.float16] * len(args.image_generators)

    image_generators, _ = load_models(
        model_names=args.image_generators,
        devices=image_generator_devices,
        dtypes=image_generator_dtypes,
        model_configs=config,
        model_classes=model_classes,
        return_dict=True
    )
    sdxl = image_generators['sdxl']

    # load vision encoders
    print(f'Loading vision encoders: {args.vision_encoders}...')
    vision_encoder_devices = ['cuda:7'] * len(args.vision_encoders)
    vision_encoder_dtypes = [torch.bfloat16] * len(args.vision_encoders)

    vision_encoders, vision_encoder_loss_funcs = load_models(
        model_names=args.vision_encoders,
        devices=vision_encoder_devices,
        dtypes=vision_encoder_dtypes,
        model_configs=config,
        model_classes=model_classes,
        require_loss_func=True,
        model_loss_funcs=model_loss_funcs,
        return_dict=True
    )

    # generate a content image of the style image
    print('Generating the style content image...')
    negative_prompt = 'text, watermark, low res, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry'
    style_content_image = sdxl.text2image(
        prompt=args.style_content_prompt,
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_samples=1,
        guidance_scale=5.0,
        num_inference_steps=50
    )[0]
    style_content_image.save('style_content_image.png')

    # extract the style from the style image
    print('Extracting the style...')
    style_image = Image.open(args.style_image).convert('RGB')
    style_anchor_image, _, _ = invert_train(
        style_image=style_image,
        style_content_image=style_content_image,
        vision_encoders=vision_encoders,
        vision_encoder_loss_funcs=vision_encoder_loss_funcs,
        vision_weights=args.vision_weights,
        content_weight=args.content_weight,
        regular_weight=args.regular_weight,
        image_size=args.image_size,
        augment_batch=args.augment_batch,
        lr=args.invert_lr,
        epochs=args.invert_epochs
    )
    style_anchor_image = transforms.ToPILImage()(style_anchor_image.squeeze(0))
    style_anchor_image.save('style_anchor_image.png')

    # perform style transfer using the extracted style
    print('Performing style transfer...')
    sdxl.load_ip_adapter(**config.sdxl)
    style_anchor_image = style_anchor_image.resize(size=(512, 512), resample=Image.Resampling.BICUBIC)
    ip_adapter_scale = eval(args.ip_adapter_scale)

    for model_name, model_config in config._data.items():
        if 'preprocess' in model_config:
            model_config['preprocess']['augment_mode'] = 'none'

    for model in vision_encoders.values():
        model.to(torch.float32)

    style_trans_images = sdxl.ip_adapter_image2image(
        image=style_anchor_image,
        image_guidance_scale=ip_adapter_scale,
        prompt=args.style_trans_prompt,
        negative_prompt=negative_prompt,
        prompt_guidance_scale=5.0,
        height=1024,
        width=1024,
        num_samples=args.num_samples,
        num_inference_steps=50,
        edit_mode='calib_style',
        style_image=style_image,
        style_content_image=style_content_image,
        vision_encoders=vision_encoders,
        vision_encoder_loss_funcs=vision_encoder_loss_funcs,
        vision_weights=args.vision_weights,
        calib_guidance_scale=args.calib_scale,
        lr=args.calib_lr,
        epochs=args.calib_epochs,
        eps=args.eps
    )
    for i, style_trans_image in enumerate(style_trans_images):
        style_trans_image.save(f'style_trans_image_{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_generators', type=str, nargs='+', default=['sdxl'], choices=['sdxl'])
    parser.add_argument('--vision_encoders', type=str, nargs='+', default=['robust_clip', 'csd_clip', 'dino', 'inception', 'vgg'], choices=['clip', 'robust_clip', 'csd_clip', 'dino', 'inception', 'vgg'])
    parser.add_argument('--style_image', type=str, default='/docs/style_image_example.png')
    parser.add_argument('--style_content_prompt', type=str, default='Sky with celestial bodies, tall tree on left, hills in background, village with church steeple in foreground.')
    parser.add_argument('--style_trans_prompt', type=str, default='A cat.')
    parser.add_argument('--ip_adapter_scale', type=str, default="{'down': {'block_2': [0.0, 0.5]}, 'mid': 0.5, 'up': {'block_0': [0.0, 1.0, 0.0]}}")
    parser.add_argument('--vision_weights', type=float, nargs='+', default=None)
    parser.add_argument('--content_weight', type=float, default=None)
    parser.add_argument('--regular_weight', type=float, default=0.05)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--augment_batch', type=int, default=20)
    parser.add_argument('--invert_lr', type=float, default=1e-2)
    parser.add_argument('--invert_epochs', type=int, default=300)
    parser.add_argument('--calib_scale', type=float, default=0.1)
    parser.add_argument('--calib_lr', type=float, default=1e-2)
    parser.add_argument('--calib_epochs', type=int, default=5)
    parser.add_argument('--eps', type=float, default=2/255)
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()

    main(args)
