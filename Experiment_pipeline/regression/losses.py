from typing import Literal
import torch

import utils
from loss import Loss

def get_scaling_function(scale_name):
    SCALE_TYPES = {
        'linear': lambda x: x,
        'log': torch.log,
        'log1p': lambda x: torch.log(1 + x),
        'exponential': torch.exp
    }
    
    if not isinstance(scale_name, str):
        raise TypeError(f'Argument `scale` should be str, but got {type(scale_name)} ({scale_name}).')
    if scale_name in SCALE_TYPES:
        return SCALE_TYPES[scale_name]
    else:
        raise ValueError('Scaling should be one of ' + ', '.join([f'\'{s}\'' for s in list(SCALE_TYPES.keys())[:-1]]) + f', or \'{list(SCALE_TYPES.keys())[-1]}\', not \'{scale_name}\'.')

ELEMWISE_REDUCTIONS = {
    'max': lambda x: torch.max(x, dim = 1)[0],
    'mean': lambda x: torch.mean(x, dim = 1),
    'sum': lambda x: torch.sum(x, dim = 1),
    'none': lambda x: x}

def get_elemwise_reduction(reduction_str_or_dict, *args, **kwargs):
    if reduction_str_or_dict in ELEMWISE_REDUCTIONS:
        return ELEMWISE_REDUCTIONS[reduction_str_or_dict]
    
    if not isinstance(reduction_str_or_dict, utils.config_dict.ConfigDict):
        pass
    
    red_name, red_dict = reduction_str_or_dict.item()
    if red_name == 'smooth_max':
        alpha = red_dict.get('alpha', 1)
        def smooth_max(x):
            exps = torch.exp(alpha * x)
            return torch.sum(x * exps, dim = 1) / torch.sum(exps, dim = 1)
        return smooth_max
    
    if red_name == 'lp_norm':
        p = red_dict.get('p', 4)
        p_inv = 1 / p
        def norm(x):
            return torch.sum(torch.abs(x) ** p, dim = 1) ** p_inv
        return norm

class LpLoss(torch.nn.Module):
    def __init__(self, p = 2, take_pth_root = False,
                 scale : Literal['linear', 'log', 'log1p', 'exponential'] = 'linear',
                 reduction : Literal['mean', 'sum', 'none'] = 'mean',
                 elementwise_reduction : Literal['mean', 'sum', 'none', 'max'] = 'mean',
                 *args, **kwargs):
        super().__init__()
        self.p = p
        self.root = take_pth_root

        if self.root:
            self.p_inv = 1 / p
        
        if isinstance(scale, utils.config_dict.ConfigDict):
            scale = scale.key()
        self.name = f'l{p}_loss' if scale == 'linear' else f'l{p}_{scale}_loss'
        self.scale = get_scaling_function(scale)
        self.batchwise_reduce = Loss.REDUCTION_METHODS[reduction]
        self.elemwise_reduce = ELEMWISE_REDUCTIONS[elementwise_reduction]
        self.reduce = lambda x: self.batchwise_reduce(self.elemwise_reduce(x))
    
    def forward(self, prediction, label):
        prediction, label = self.scale(prediction), self.scale(label)
        value = self.reduce(torch.abs(prediction - label) ** self.p)
        if self.root:
            value = value ** self.p_inv
        return value

class RMSELoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, prediction, label):
        mse = super().forward(prediction, label)
        return torch.sqrt(mse)

class R2Loss(torch.nn.Module):
    def __init__(self, smoothing_term = 1e-10,
                 scale : Literal['linear', 'log', 'log1p', 'exponential'] = 'linear',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = smoothing_term

        if isinstance(scale, utils.config_dict.ConfigDict):
            scale = scale.key()
        self.name = 'r2_loss' if scale == 'linear' else f'r2_{scale}_loss'
        self.scale = get_scaling_function(scale)
    
    def forward(self, prediction, label):
        prediction, label = self.scale(prediction), self.scale(label)
        squared_error = torch.sum((prediction - label) ** 2)
        variance = torch.sum((label - label.mean()) ** 2)
        return (squared_error + self.eps) / (variance + self.eps)
        

class RelativeL1Loss(torch.nn.Module):

    def __init__(self, scale : Literal['linear', 'log', 'log1p', 'exponential'] = 'linear',
                 reduction : Literal['mean', 'sum', 'none'] = 'mean',
                 elementwise_reduction : Literal['mean', 'sum', 'none', 'max'] = 'mean',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(scale, utils.config_dict.ConfigDict):
            scale = scale.key()
        self.name = 'relative_l1_loss' if scale == 'linear' else f'relative_l1_{scale}_loss'
        self.scale = get_scaling_function(scale)
        self.batchwise_reduce = Loss.REDUCTION_METHODS[reduction]
        self.elemwise_reduce = ELEMWISE_REDUCTIONS[elementwise_reduction]
        self.reduce = lambda x: self.batchwise_reduce(self.elemwise_reduce(x))
    
    def forward(self, prediction, target, *args, **kwargs):
        return self.reduce(torch.nan_to_num(torch.abs(1 - self.scale(prediction)/self.scale(target))))

class AbsoluteOrRelativeLoss(torch.nn.Module):

    def __init__(self, relative_threshold = 0.05, absolute_threshold = 0.01, exponent = 1,
                 reduction : Literal['mean', 'sum', 'none'] = 'mean',
                 elementwise_reduction : Literal['mean', 'sum', 'none', 'max'] = 'mean',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_del = relative_threshold
        self.abs_del = absolute_threshold
        self.exp = exponent
        if exponent != 1:
            self.name = f'absolute_or_relative_l{exponent}_loss'
        self.batchwise_reduce = Loss.REDUCTION_METHODS[reduction]
        self.elemwise_reduce = ELEMWISE_REDUCTIONS[elementwise_reduction]
        self.reduce = lambda x: self.batchwise_reduce(self.elemwise_reduce(x))
    
    def forward(self, prediction, target, *args, **kwargs):
        scale = torch.maximum(self.rel_del * torch.abs(target), self.abs_del * torch.ones_like(target))
        loss = torch.abs(prediction - target) / scale
        return self.reduce(loss ** self.exp)

class CorrectnessRankingLoss(torch.nn.Module):
    
    PASS_ALL_INPUTS = True
    
    def __init__(self, reduction : Literal['mean', 'sum', 'none'] = 'mean',
                 elementwise_reduction = 'max', *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.reduce = Loss.REDUCTION_METHODS[reduction]
        self.rel_l1_loss = RelativeL1Loss(reduction = 'none', *args, **kwargs)
        
        if elementwise_reduction in ('max', 'min'):
            mix = getattr(torch, elementwise_reduction)
            self.aggregate_loss = lambda *args, **kwargs: mix(*args, **kwargs)[0]
        else:
            self.aggregate_loss = getattr(torch, elementwise_reduction)
    
    def pointwise_call(self, l1, l2, c1, c2):
        c_diff = c1 - c2
        l_diff = l2 - l1
        margin = torch.min(torch.ones_like(l_diff), torch.abs(l_diff))
        loss = -torch.sign(l_diff) * c_diff + margin
        return torch.max(torch.zeros_like(loss), loss)
    
    def forward(self, predictions, label, *args, **kwargs):
        pred, conf = predictions
        unreduced_rel_losses = self.rel_l1_loss(pred, label)
        unreduced_shape = unreduced_rel_losses.shape + (1,) * (unreduced_rel_losses.dim() == 1)
        rel_losses = self.aggregate_loss(unreduced_rel_losses.view(*unreduced_shape),
                                         dim = -1, keepdim = True)
        n = pred.shape[0]
        losses = [self.pointwise_call(rel_losses[i], rel_losses[j], conf[i], conf[j])
                  for i, j in zip(range(n), map(lambda i: (i + 1) % n, range(n)))]
        losses = torch.stack(losses)
        return self.reduce(losses)