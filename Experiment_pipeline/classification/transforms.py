import copy
import numpy as np
import torch
import torchvision

from transform import TransformWrapper, Mixup

class OneHotEncoding(TransformWrapper):
    
    PARAMS = dict(num_classes = 1000)
    
    def __init__(self, config_dict = {}, *args, **kwargs):
        self.num_classes = config_dict['num_classes']
    
    def transform(self, input):
        input['label'] = torch.nn.functional.one_hot(torch.tensor(input['label']).to(int), num_classes = self.num_classes).moveaxis(-1, 0)
        return input

class CutMix(TransformWrapper):
    
    PARAMS = dict(probability = 1.0, alpha = 1.0)
    
    num_datapoints = 2
    
    def __init__(self, config_dict = {}, seed = None, *args, **kwargs):
        self.p  = config_dict['probability']
        self.alpha = config_dict['alpha']
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, **input):
        (x1, x2), (t1, t2) = input['x'], input['label']
        input = {k: v[0] if isinstance(v, list) else v for k, v in input.items()}
        input['k'] = 1
        if self.rng.binomial(1, self.p):
            num_channels = len(x1.shape) - 2
            w, h = x1.shape[-2:]
            lda = self.rng.beta(self.alpha, self.alpha)
            scale = np.sqrt(1 - lda) / 2
            
            rx, ry = self.rng.integers(0, w), self.rng.integers(0, h)
            rw, rh = np.round(w * scale).astype(int), np.round(h * scale).astype(int)
            
            x_start, x_end = max(0, rx - rw), min(w, rx + rw)
            y_start, y_end = max(0, ry - rh), min(h, ry + rh)
            
            slices = (slice(None),) * num_channels + (slice(x_start, x_end), slice(y_start, y_end))
            x = copy.deepcopy(x1)
            x[slices] = x2[slices]
            
            mu = (x_end - x_start) * (y_end - y_start) / (h * w)
            input['x'] = x
            input['label'] = (1 - mu) * t1 + mu * t2
        return input

class MixupOrCutMix(TransformWrapper):
    
    PARAMS = dict(cutmix_params = CutMix.PARAMS, mixup_params = Mixup.PARAMS, switch_probability = 0.5)
    
    num_datapoints = 2
    
    @staticmethod
    def fill_kwargs(config_dict):
        config_dict['cutmix_params'].fill_with_defaults(CutMix.PARAMS)
        config_dict['mixup_params'].fill_with_defaults(Mixup.PARAMS)
    
    def __init__(self, config_dict = {}, seed = None, *args, **kwargs):
        self.cutmix = CutMix(config_dict['cutmix_params'], seed = seed, *args, **kwargs)
        self.mixup = Mixup(config_dict['mixup_params'], seed = seed, *args, **kwargs)
        self.switch_p = config_dict['switch_probability']
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, *args, **kwargs):
        if self.rng.binomial(1, self.switch_p):
            return self.mixup(*args, **kwargs)
        else:
            return self.cutmix(*args, **kwargs)

class RandAugment(torchvision.transforms.RandAugment):
    
    def __init__(self, num_ops = 2, magnitude = 9, num_magnitude_bins = 31, interpolation = 'NEAREST', fill = None):
        interpolation = getattr(torchvision.transforms.InterpolationMode, interpolation)
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        dtype = x.dtype
        scale = 1
        if torch.is_floating_point(x):
            if torch.any((x != 0) & (x != 1)):
                x = 255 * x
                scale = 255
        x = super().forward(x.to(dtype = torch.uint8)) / scale
        return x.to(dtype)
            