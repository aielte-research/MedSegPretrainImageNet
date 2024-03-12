import torch
import utils
from .optims_dict import optimizers_dict

import importlib

class Optimizer(torch.optim.Optimizer):

    PARAMS = {
        'learning rate': 0.01
    }

    @staticmethod
    def fill_kwargs(config_dict):
        optim_name, optim_dict = config_dict.item()
        if optim_name in optimizers_dict:
            optim_dict.fill_with_defaults(optimizers_dict[optim_name]['arguments'])
        else:
            utils.fill_dict(config_dict)
            optim_dict.pop('lr', None)
        lr = Optimizer.PARAMS['learning rate']
        lr_dict = optim_dict.get_or_update('learning rate', 'constant', final = False)
        lr_dict.get_or_update(f'{lr_dict.key()}/base', lr)
        if lr_dict.key() != 'constant':
            utils.fill_dict(lr_dict)
    
    def __init__(self, config_dict, params):
        optim_name, optim_dict = config_dict.item()
        lr_dict = optim_dict['learning rate']
        lr = lr_dict[f'{lr_dict.key()}/base']

        if optim_name in optimizers_dict:
            optim_kwargs = config_dict.to_kwargs(optimizers_dict[optim_name]['arguments'])
            optim_const = optimizers_dict[optim_name]['init']

        else:
            module_name, class_name = '.'.join(optim_name.split('.')[:-1]), optim_name.split('.')[-1]
            optim_const = getattr(importlib.import_module(module_name), class_name)
            optim_kwargs = utils.get_kwargs(optim_const, optim_dict.mask('learning_rate'))
        
        self.optim = optim_const(lr = lr, params = params, **optim_kwargs)
        self.param_groups = self.optim.param_groups
    
    def zero_grad(self, *args, **kwargs):
        return self.optim.zero_grad(*args, **kwargs)
    
    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        return self.optim.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        self.optim.load_state_dict(*args, **kwargs)
        self.param_groups = self.optim.param_groups
    
    def add_param_group(self, *args, **kwargs):
        self.optim.add_param_group(*args, **kwargs)
        self.param_groups = self.optim.param_groups