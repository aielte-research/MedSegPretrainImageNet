from cProfile import label
import torch
from torch import nn
import numpy as np

from loss import Loss
from utils import config_dict

class BCEWrapper(Loss):
    """Wrapper object for torch.nn.BCELoss"""

    PARAMS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(lambda **kwargs: None, config_dict.ConfigDict({}))
        loss = nn.BCELoss()
        def calculator(prediction, mask = None, label = None, **kwargs):
            y = mask if mask is not None else label
            return loss(prediction, y)
        self.calculator = calculator
        self.calculate = lambda batch: calculator(**batch)
        self.name = 'binary_crossentropy_loss'

class SensitivitySpecificityLoss(nn.Module):
    
    """One minus the weighted sum of continuous true positive rate and true negative rate."""

    PARAMS = {
        'weight of sensitivity': {
            'argument name': 'alpha',
            'default': 0.5
            },
        'square terms': {
            'argument name': 'square_terms',
            'default': False
            },
        'square confusion matrix entries': {
            'argument name': 'square_preds',
            'default': False
            },
        'batchwise': {
            'argument name': 'batchwise_loss',
            'default': False
            }
    }

    def __init__(self,
                 alpha : float = 0.5,
                 square_preds : bool = False,
                 square_terms : bool = False,
                 eps : float = 0.01,
                 batchwise_loss : bool = True,
                 *args, **kwargs):
        """
        Arguments:
            `alpha`: float; weight of sensitivity; weight of specificity will be `1 - alpha`
            `square_preds`: bool; if False, calculate eg. true positives as 1*y_hat, if True, as y_hat*y_hat
            `square_terms`: bool; if True, square the true positive and true negative rates
            `batchwise_loss`: bool; if False, calculate a different loss value for each record in the batch, and then average them
        """
        
        super().__init__()
        
        self.alpha = alpha
        self.square_preds = square_preds
        self.square_terms = square_terms
        self.eps = eps
        self.batchwise = batchwise_loss
    
    def forward(self, prediction, mask, *args, **kwargs):
        axes = tuple(range(len(prediction.shape))) if self.batchwise else tuple(range(1, len(prediction.shape)))
        tp, fp, fn, tn = get_tp_fp_fn_tn(prediction, mask, axes, None, self.square_preds)

        tpr = (tp + self.eps) / (tp + fn + self.eps)
        tnr = (tn + self.eps) / (tn + fp + self.eps)

        if self.square_terms:
            tpr, tnr = tpr ** 2, tnr ** 2
        
        sens_spec_loss = self.alpha * tpr + (1 - self.alpha) * tnr
        sens_spec_loss = sens_spec_loss.mean()
        
        return 1 - sens_spec_loss

class CrossEntropyLoss(nn.Module):
    
    def __init__(self, label_smoothing = 0.0, *args, **kwargs):
        super().__init__()
        if label_smoothing >= 0.5:
            raise ValueError('Label smoothing value should be <0.5')
        self.smooth = label_smoothing
        self.log_clamp = -100
    
    def forward(self, prediction, label, *args, **kwargs):
        n_classes = prediction.shape[1]

        log_pred = torch.log(prediction).view(*prediction.shape[:2], -1)
        log_pred = log_pred.nan_to_num().clamp(self.log_clamp) # clamp values at a minimum to avoid problems with log(0)

        bin_label = torch.nn.functional.one_hot(label.flatten(1).long(), num_classes = n_classes)
        bin_label = bin_label.moveaxis(-1, 1)
        if self.smooth:
            bin_label = torch.clamp(bin_label, self.smooth / n_classes, 1 - (self.smooth / n_classes))
        
        loss = -torch.sum(log_pred * bin_label, axis = 1)

        return loss.mean()


class CrossEntropySqueeze(nn.CrossEntropyLoss):
    def forward(self, input, target):
        # print(target.shape)
        if target.shape[1] == 1:
            target = torch.squeeze(target, dim = 1)
            # print('squeezes:', target.shape)
        return super(CrossEntropySqueeze, self).forward(input, target.long()).mean()






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



"""
Loss implementations are from https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
"""
    

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(1, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    tp = sum_tensor(tp, axes, keepdim = False)
    fp = sum_tensor(fp, axes, keepdim = False)
    fn = sum_tensor(fn, axes, keepdim = False)
    tn = sum_tensor(tn, axes, keepdim = False)

    return tp, fp, fn, tn

class TverskyLoss(nn.Module):

    PARAMS = {
        'batchwise': {
            'argument name': 'batch_dice',
            'default': False
            },
        'square confusion matrix entries': {
            'argument name': 'square',
            'default': False
            },
        'weight of false positives': {
            'argument name': 'alpha',
            'default': 0.5
            },
        'weight of false negatives': {
            'argument name': 'beta',
            'default': 0.5
            },
        'include_background': {
            'argument name': 'do_bg',
            'default': True
        }
    }

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1.,
                 square=False, alpha = 0.3, beta = 0.7):
        # TODO: `do_bg` argument has to be True for one class, and False for multiclass; this should be changed
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, mask, loss_mask=None, *args, **kwargs):
        shp_x = prediction.shape
        len_shp_x = len(shp_x)

        if self.batch_dice:
            axes = [0] + list(range(2, len_shp_x))
        else:
            axes = list(range(2, len_shp_x))

        if self.apply_nonlin is not None:
            prediction = self.apply_nonlin(prediction)

        tp, fp, fn, _ = get_tp_fp_fn_tn(prediction, mask, axes, loss_mask, self.square)


        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

        binary = len_shp_x == 3 or shp_x[1] == 1
        if not binary and not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """

    PARAMS = {
        'focal gamma': {
            'argument name': 'gamma',
            'default': 0.75
            },
        **TverskyLoss.PARAMS
    }

    def __init__(self, gamma=0.75, **tversky_kwargs):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, prediction, mask, *args, **kwargs):
        tversky_loss = self.tversky(prediction, mask)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    PARAMS = {
        'weight of positives': {
            'argument name': 'alpha',
            'default': 0.5
            },
        'focal gamma': {
            'argument name': 'gamma',
            'default': 0.0
            },
        'label smoothing': {
            'argument name': 'smooth',
            'default': 1e-5
            }
        }

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=1, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, prediction, mask, *args, **kwargs):
        if self.apply_nonlin is not None:
            prediction = self.apply_nonlin(prediction)

# Ez a sor kezeli a bináris esetet. Visszavezetem az általánosra.
        if prediction.size(1)==1:
            prediction=torch.cat([1-prediction, prediction], dim=1)
        
        num_class = prediction.shape[1]

        if prediction.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prediction = prediction.view(prediction.size(0), prediction.size(1), -1)
            prediction = prediction.permute(0, 2, 1).contiguous()
            prediction = prediction.view(-1, prediction.size(-1)) # -> (N*d1*d2*...,C)
        mask = torch.squeeze(mask, 1)
        mask = mask.view(-1, 1) # -> (N*d1*d2*...,1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != prediction.device:
            alpha = alpha.to(prediction.device)

        idx = mask.cpu().long()

        one_hot_key = torch.FloatTensor(mask.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != prediction.device:
            one_hot_key = one_hot_key.to(prediction.device)


# Itt valamelyik smoothing szerintem nem kell...
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * prediction).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp






class BoundaryLoss(nn.Module): # TODO: add PARAMS
    """
    Based on: https://arxiv.org/abs/1812.07032
    Implementation of the Boundary loss, that takes the average of the distance from the boundary of the target area
    over the pixels weighted with the predictions from the model.
    apply_nonlin: apply nonlinear function for raw output if neccessery
    dist_transform_function: function to modify the distance values if neccessary
    weighting_method: the way to combine boundary distance values with predictions
    """
    
    PARAMS = {
        'distance map clamping': {
            'argument name': 'distance_map_clamping',   # list or tuple in form: [min, max]
            'default': None
            },
        'distance map rescale': {
            'argument name': 'distance_map_rescale',    # number to downscale distnaces with
            'default': None
            }
        # TODO: weighting method
        }
    
    def __init__(self, apply_nonlin=None, weighting_method = (lambda x,y : x*y), distance_map_clamping=None, distance_map_rescale=None, **kwargs):
        super(BoundaryLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.clamping = distance_map_clamping
        self.rescale = distance_map_rescale
        self.weighting_method=weighting_method
    
    def forward(self, prediction, distance_map, *args, **kwargs):
        if self.apply_nonlin is not None:
            prediction = self.apply_nonlin(prediction)
        if self.clamping:
            distance_map = torch.clamp(distance_map, self.clamping[0], self.clamping[1])
        if self.rescale:
            distance_map = distance_map/self.rescale
        
        multiplied = self.weighting_method(prediction, distance_map)
        loss = multiplied.mean()

        return loss



