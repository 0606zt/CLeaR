import torch
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict
from tqdm import tqdm


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)


def cal_gram_matrix(x):
    b, ch, h, w = x.shape
    embeds = x.view(b, ch, w * h)
    embeds_t = embeds.transpose(1, 2)
    gram = embeds.bmm(embeds_t) / (ch * h * w)
    return gram


def load_models(
        model_names: list[str],
        devices: list[str],
        dtypes: list[torch.dtype],
        model_configs: dict,
        model_classes: dict,
        require_loss_func=False,
        model_loss_funcs: dict = None,
        return_dict=True
):
    assert len(model_names) == len(devices) == len(dtypes)

    models = {}
    loss_funcs = {}
    for model_name, device, dtype in tqdm(zip(model_names, devices, dtypes), total=len(model_names)):
        cfg = getattr(model_configs, model_name)
        cls = model_classes[cfg['class_name']]
        model = cls.load_model(device=device, dtype=dtype, is_train=False, **cfg)
        models[model_name] = model

        if require_loss_func:
            loss_func = model_loss_funcs[cfg['loss_type']]
            loss_funcs[model_name] = loss_func

    if len(models) == 1 and not return_dict:
        if require_loss_func:
            return models[model_names[0]], loss_funcs[model_names[0]]
        else:
            return models[model_names[0]], loss_funcs
    else:
        return models, loss_funcs


class NoProgressBar:
    def __init__(self, total=None, desc=None):
        self.total = total
        self.desc = desc
        self.update = self._update

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _update(self, n=1):
        pass

    def close(self):
        pass
