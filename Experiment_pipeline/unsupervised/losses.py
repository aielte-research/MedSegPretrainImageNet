from typing import Tuple
import utils
from loss import Loss

import torch

class ConsistencyLoss(torch.nn.Module):
    
    PASS_ALL_INPUTS = True
    
    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['base'])
    
    def __init__(self, base = 'torch.nn.MSELoss', *args, **kwargs):
        super().__init__()
        base_loss = utils.create_object_from_dict(base, wrapper_class = Loss, *args, **kwargs)
        self.loss : torch.nn.Module = getattr(base_loss, 'calculator', base_loss)
        self.name = base_loss.name.rstrip('_loss') + '_consisteny_loss'
    
    def forward(self, predictions : Tuple[torch.Tensor], *args, **kwargs):
        y1, y2 = predictions
        return self.loss(y1, y2)
          