import torch
import torch.nn.functional as nnf
from torch_fidelity.metric_kid import kid_features_to_metric


def cal_ortho_weight(
        embeds1: torch.Tensor,
        embeds2: torch.Tensor,
        is_pairwise=True,
):
    embeds1_norm = embeds1 / embeds1.norm(p=2, dim=1, keepdim=True)
    embeds2_norm = embeds2 / embeds2.norm(p=2, dim=1, keepdim=True)
    ortho_weight = embeds1_norm @ embeds2_norm.T
    if is_pairwise:
        ortho_weight = torch.diagonal(ortho_weight)
    return ortho_weight


def cal_cos_sim_loss(
        embeds1: torch.Tensor,
        embeds2: torch.Tensor,
        loss_scale=1.0,
        is_pairwise=False,
        is_neg=False
):
    embeds1_norm = embeds1 / embeds1.norm(p=2, dim=1, keepdim=True)
    embeds2_norm = embeds2 / embeds2.norm(p=2, dim=1, keepdim=True)
    cos_sim = embeds1_norm @ embeds2_norm.T
    if is_pairwise:
        cos_sim = torch.diagonal(cos_sim)
    if is_neg:
        loss = cos_sim.mean()
    else:
        loss = 1 - cos_sim.mean()
    return loss * loss_scale


def cal_gram_loss(
        grams1: list[torch.Tensor],
        grams2: list[torch.Tensor],
        loss_scale=1.0,
        is_pairwise=False,
        is_neg=False
):
    loss = torch.tensor(0.0, device=grams1[0].device)
    for gram1, gram2 in zip(grams1, grams2):
        if not is_pairwise:
            gram2 = gram2.repeat(gram1.shape[0], 1, 1)
        loss += nnf.mse_loss(gram1, gram2)
    if is_neg:
        loss *= -1
    return loss * loss_scale


def cal_kid_loss(
        embeds1: torch.Tensor,
        embeds2: torch.Tensor,
):
    kid = kid_features_to_metric(
        embeds1,
        embeds2,
        kid_subsets=100,
        kid_subset_size=min(1000, embeds1.shape[0], embeds2.shape[0]),
        verbose=False
    )
    loss = kid['kernel_inception_distance_mean']
    return loss


def cal_tv_loss(images: torch.Tensor):
    b, ch, h, w = images.shape
    tv_h = torch.pow(images[:, :, 1:, :] - images[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(images[:, :, :, 1:] - images[:, :, :, :-1], 2).sum()
    loss = (tv_h + tv_w) / (b * ch * h * w)
    return loss
