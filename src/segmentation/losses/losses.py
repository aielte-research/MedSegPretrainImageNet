from cProfile import label
import torch
from torch import nn
import numpy as np

from loss import Loss
from utils import config_dict



class DiceLoss(nn.Module):
    # based on: https://github.com/Beckschen/TransUNet/blob/d68a53a2da73ecb496bb7585340eb660ecda1d59/utils.py#L9
    
    """
    Smooth version of 1 - DSC, where the DSC (Dice similarity coefficient) is calculated as DSC(X, Y) = 2|X \cap Y| / (|X| + |Y|), and if X is a continuous tensor, |X| is replaced with X*X.
    """

    def __init__(self, batchwise = True, include_background = True, smoothing_term = 1e-5, apply_softmax = False, *args, **kwargs):
        """
        Arguments:
            `batchwise`: bool; if set to True, calculate the Dice scores over all datapoints in a batch; if set to False, calculate it over individual datapoints, then average them.
            `include_background`: bool; whether to include the 0th channel in the mean
            `smoothing_term`: float; term to add to the numerator and denominator to avoid division by 0
            `apply_softmax`: bool; whether to softmax the prediction
        """

        super().__init__()
        self.eps = smoothing_term
        self.axes_start = int(not batchwise)
        self.include_background = include_background
        self.classes_start = int(not include_background)
        self.softmax = apply_softmax
    
    def dice_index(self, y_hat, y, axes = (1, 2)):
        """Calculates the smooth Dice index for one class."""

        intersection = torch.sum(y * y_hat, axis = axes, keepdim = True)
        y_size = torch.sum(y, axis = axes, keepdim = True)
        y_hat_size = torch.sum(y_hat ** 2, axis = axes, keepdim = True)
        return (2 * intersection + self.eps) / (y_size + y_hat_size + self.eps)
    
    def forward(self, prediction, mask, *args, **kwargs):
        if self.softmax:
            prediction = torch.softmax(prediction, dim = 1)
        n_classes = prediction.shape[1]
        if n_classes == 1:
            if self.include_background:
                prediction = torch.concat([1 - prediction, prediction], axis = 1)
                n_classes = 2
            else:
                self.classes_start = 0
                mask = 1 - mask
        axes = tuple(range(self.axes_start, len(prediction.shape) - 1))
        mask = mask.view(-1, *prediction.shape[2:])

        dice_idcs = torch.concat([self.dice_index(prediction[:, i], mask == i, axes = axes)
                                  for i in range(self.classes_start, n_classes)])
        return 1 - dice_idcs.mean()



