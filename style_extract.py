import warnings
from diffusers.utils import logging
warnings.filterwarnings('ignore')
logging.disable_progress_bar()

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
from util.loss_func import cal_tv_loss, cal_ortho_weight


def invert_train(
        style_image: Image.Image,
        style_content_image: Image.Image,
        vision_encoders: dict,
        vision_encoder_loss_funcs: dict,
        vision_weights: list[float] = None,
        content_weight: float = None,
        regular_weight: float = 0.05,
        image_size=224,
        augment_batch=12,
        lr=1e-2,
        epochs=500
):
    device = next(iter(vision_encoders.values())).device
    dtype = next(iter(vision_encoders.values())).dtype

    style_image_embeds = []
    style_content_image_embeds = []
    c_w = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        for model_name, model in vision_encoders.items():
            image_embeds = model.get_image_embeds(style_image, is_train=False)
            style_image_embeds.append(image_embeds)
            image_embeds = model.get_image_embeds(style_content_image, is_train=False)
            style_content_image_embeds.append(image_embeds)
            if model_name not in ['csd_clip', 'vgg']:
                if content_weight is None:
                    ortho_content_weight = cal_ortho_weight(style_image_embeds[-1], style_content_image_embeds[-1], is_pairwise=True)
                    c_w.append(ortho_content_weight.item())
                else:
                    c_w.append(content_weight)
            else:
                c_w.append(0.0)

    if vision_weights is None:
        v_w = torch.ones(len(vision_encoders), device=device) / len(vision_encoders)  # average weights
    else:
        assert len(vision_weights) == len(vision_encoders)
        v_w = torch.tensor(vision_weights, device=device)
    v_w = torch.softmax(v_w, dim=0)
    r_w = regular_weight

    z = torch.randn(1, 3, image_size, image_size, device=device, requires_grad=True)

    optimizer = torch.optim.AdamW([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    total_loss_record = []
    models_loss_record = {model_name: [] for model_name in vision_encoders}
    with torch.cuda.amp.autocast(dtype=dtype):
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                z.data = torch.clamp(z.data, 0.0, 1.0)
                z_batch = z.repeat(augment_batch, 1, 1, 1)
                total_loss = torch.tensor(0.0, device=device)

                for i, (model_name, model) in enumerate(vision_encoders.items()):
                    z_embeds = model.get_image_embeds(z_batch, is_train=True)

                    loss_func = vision_encoder_loss_funcs[model_name]
                    loss_scale = getattr(model, 'style_weight', 1.0)

                    vision_loss = loss_func(z_embeds, style_image_embeds[i], loss_scale=loss_scale, is_pairwise=False, is_neg=False)
                    if model_name not in ['csd_clip', 'vgg']:
                        content_loss = loss_func(z_embeds, style_content_image_embeds[i], loss_scale=loss_scale, is_pairwise=False, is_neg=True)
                    else:
                        content_loss = torch.tensor(0.0, device=device)

                    loss = vision_loss + content_loss * c_w[i]
                    total_loss += loss * v_w[i]
                    models_loss_record[model_name].append(loss.item())

                regular_loss = cal_tv_loss(z)
                total_loss += regular_loss * r_w

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix({'loss': total_loss.item()})
                total_loss_record.append(total_loss.item())
                # print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item():.4f}')

    return z.clamp(0.0, 1.0).detach().cpu(), total_loss_record, models_loss_record


if __name__ == '__main__':
    from util.model_util import fix_seed, load_models
    from util.registry import model_classes, model_loss_funcs
    from util.config import config

    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoders', type=str, nargs='+', default=['robust_clip', 'csd_clip', 'dino', 'inception', 'vgg'], choices=['clip', 'robust_clip', 'csd_clip', 'dino', 'inception', 'vgg'])
    parser.add_argument('--style_image', type=str, default='/docs/style_image_example.png')
    parser.add_argument('--style_content_image', type=str, default='/docs/style_content_image_example.png')
    parser.add_argument('--vision_weights', type=float, nargs='+', default=None)
    parser.add_argument('--content_weight', type=float, default=None)
    parser.add_argument('--regular_weight', type=float, default=0.05)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--augment_batch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    fix_seed(args.seed)

    print(f'Loading vision encoders: {args.vision_encoders}...')
    devices = ['cuda:0'] * len(args.vision_encoders)
    dtypes = [torch.bfloat16] * len(args.vision_encoders)

    vision_encoders, vision_encoder_loss_funcs = load_models(
        model_names=args.vision_encoders,
        devices=devices,
        dtypes=dtypes,
        model_configs=config,
        model_classes=model_classes,
        require_loss_func=True,
        model_loss_funcs=model_loss_funcs,
        return_dict=True
    )

    print('Extracting the style...')
    style_content_image = Image.open(args.style_content_image).convert('RGB')
    style_image = Image.open(args.style_image).convert('RGB')

    style_anchor_image, total_loss_record, models_loss_record = invert_train(
        style_image=style_image,
        style_content_image=style_content_image,
        vision_encoders=vision_encoders,
        vision_encoder_loss_funcs=vision_encoder_loss_funcs,
        vision_weights=args.vision_weights,
        content_weight=args.content_weight,
        regular_weight=args.regular_weight,
        image_size=args.image_size,
        augment_batch=args.augment_batch,
        lr=args.lr,
        epochs=args.epochs
    )

    style_anchor_image = transforms.ToPILImage()(style_anchor_image.squeeze(0))
    style_anchor_image.save(f'style_anchor_image,sample={round}.png')
    # torch.save(total_loss_record, 'total_loss_record.pt')
    # torch.save(models_loss_record, 'models_loss_record.pt')
