import utils
from optim import Optimizer

import torch

class MeanTeacherOptimizer(torch.optim.Optimizer):
    
    @staticmethod
    def fill_kwargs(config_dict):
        Optimizer.fill_kwargs(config_dict['optimizer'])
        config_dict['optimizer'].value().pop('learning_rate', None)
    
    def __init__(self, params, lr, optimizer : utils.ConfigDict = 'torch.optim.Adam',
                 alpha = 0.99, *args, **kwargs):
        parameters, self.get_param_pairs = params
        optimizer.value().update({'learning_rate': {'constant': {'base': lr}}})
        self.base_optim = Optimizer(optimizer, parameters)
        self.param_groups = self.base_optim.param_groups
        
        self.alpha = alpha
        self.step_count = 1
        
        self.dict_keys = ('alpha', 'step_count')
    
    def step(self, *args, **kwargs):
        result = self.base_optim.step(*args, **kwargs)
        alpha = min(1 - 1 / self.step_count, self.alpha)
        for student_param, teacher_param in self.get_param_pairs():
            teacher_param.data.mul_(alpha).add_((1-alpha) * student_param.data)
        self.step_count += 1
        return result
    
    def zero_grad(self, *args, **kwargs):
        return self.base_optim.zero_grad(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        return {'base_optim': self.base_optim.state_dict(*args, **kwargs),
                **{k: getattr(self, k) for k in self.dict_keys}}
    
    def load_state_dict(self, state_dict):
        self.base_optim.load_state_dict(state_dict['base_optim'])
        for k in self.dict_keys:
            setattr(self, k, state_dict[k])
    
    def add_param_group(self, *args, **kwargs):
        raise NotImplementedError