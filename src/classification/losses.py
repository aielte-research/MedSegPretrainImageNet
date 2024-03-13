import torch
from loss import Loss

class BCELoss(torch.nn.Module):
    
    def __init__(self, reduction = 'mean', *args, **kwargs):
        super().__init__()
        self.reduce = Loss.REDUCTION_METHODS[reduction]
    
    def forward(self, prediction, label):
        return -self.reduce(label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction))

class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    
    def __init__(self, label_smoothing = 0.0, apply_softmax = True, *args, **kwargs):
        if label_smoothing >= 0.5:
            raise ValueError('Label smoothing value should be <0.5')
        super().__init__(label_smoothing = label_smoothing, *args, **kwargs)
        self.smooth = label_smoothing
        self.log_clamp = -100

        self.forward = self.forward_with_softmax if apply_softmax else self.forward_without_softmax
    
    def forward_with_softmax(self, prediction, label):
        return super().forward(prediction, label.squeeze(1).long())

    def forward_without_softmax(self, prediction, label, *args, **kwargs):
        n_classes = prediction.shape[1]

        log_pred = torch.log(prediction).view(*prediction.shape[:2], -1)
        log_pred = log_pred.nan_to_num().clamp(self.log_clamp) # clamp values at a minimum to avoid problems with log(0)

        bin_label = torch.nn.functional.one_hot(label.flatten(1).long(), num_classes = n_classes)
        bin_label = bin_label.moveaxis(-1, 1)
        if self.smooth:
            bin_label = torch.clamp(bin_label, self.smooth / n_classes, 1 - (self.smooth / n_classes))
        
        loss = -torch.sum(log_pred * bin_label, axis = 1)

        return loss.mean()