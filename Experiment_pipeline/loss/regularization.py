from typing import Union

import re
import torch
from math import log

class ConfidencePenalty(torch.nn.Module):
    """Regularises with the negative (natural base) entropy of the predictions."""

    PARAMS = {
        'clamping': 1e-10,
        'entropy threshold': {
            'argument name': 'gamma',
            'default': log(2)
            }
        }

    LOG_PATTERN = 'log[ (]*([\d.]+)'

    def __init__(self, clamping = 1e-10, gamma : Union[float, str] = log(2), *args, **kwargs):
        """
        Arguments:
            `clamping`: float; forces values to a (`clamping`, `1-clamping`) range in order to avoid infinite gradients.
            `gamma`: float or str; if the entropy is larger than `gamma`, then only `-gamma` penalty is applied; if str, it has to be of shape 'log x' or 'log(x)' where log is the natural logarithm
        """
        super().__init__()
        self.clamping = clamping

        # convert str like 'log 2' to float
        if isinstance(gamma, str):
            m = re.match(self.LOG_PATTERN, gamma)
            if m:
                gamma = log(float(m.group(1)))
            else:
                raise TypeError(f'Cannot convert {gamma} to float.')

        self.gamma = gamma
    
    def forward(self, prediction, *args, **kwargs):
        clamped_pred = torch.clamp(prediction, self.clamping, 1 - self.clamping)
        if clamped_pred.size(1) == 1:
            clamped_pred = torch.cat([1 - clamped_pred, clamped_pred], axis = 1)
        log_likelihood = torch.log(clamped_pred)
        neg_entropy = torch.sum(clamped_pred * log_likelihood, dim = 1)
        penalty = torch.maximum(neg_entropy, - self.gamma * torch.ones_like(neg_entropy))
        return penalty.mean()

class LabelNoisePenalty(torch.nn.Module):
    """Regularises with the average log likelihood of the prediction. This is equivalent with label smoothing for the crossentropy loss."""
    
    PARAMS = {'clamping': 1e-10}

    def __init__(self, clamping = 1e-10, *args, **kwargs):
        """`clamping`: forces values to a (`clamping`, `1-clamping`) range in order to avoid infinite gradients."""
        super().__init__()
        self.clamping = clamping

    def forward(self, prediction, *args, **kwargs):
        clamped_pred = torch.clamp(prediction, self.clamping, 1 - self.clamping)
        if clamped_pred.size(1) == 1:
            clamped_pred = torch.cat([1 - clamped_pred, clamped_pred], axis = 1)
        return -torch.log(clamped_pred).mean()