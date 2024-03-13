from typing import Callable, Dict
import torch
import utils
import re

from utils.config_dict import ConfigDict

class Loss(object):
    """
    Wrapper for loss functions that handles gradient accumulation.

    Parameters:
        name: str; name of the loss function that will appear on plots
        loss_fn: callable object that returns an instance of the loss function
        accumulate: bool; whether to use gradient accumulation; default: True
        args, kwargs: positional and keyword arguments that will be passed onto loss_fn
        label_name: name of the label that the loss should be applied to (eg. 'mask', 'distance map', 'label')
    """

    PARAMS = {'label type': 'mask'}
    
    REDUCTION_METHODS : Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {'mean': torch.mean,
                                                                             'sum': torch.sum,
                                                                             'none': lambda x : x}

    @staticmethod
    def convert_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def __init__(self, loss_fn : Callable[..., torch.nn.Module],
                 config_dict : ConfigDict, accumulate = True, *args, **kwargs):
        # calculator: loss function wrapped
        # loss: loss with gradient
        # value: value of loss accumulated over multiple batches; no gradient

        loss_kwargs = utils.get_kwargs(loss_fn, config_dict.mask('label_type', 'train_prediction_index'))
        self.calculator = loss_fn(*args, **kwargs, **loss_kwargs)
        self.name = getattr(self.calculator, 'name', self.convert_to_snake(loss_fn.__name__))
        self.value = 0
        self.num_batches = 0
        
        self.label_type = config_dict.get('label type', self.PARAMS.get('label type', Loss.PARAMS.get('label type', None)))
        pred_idx = config_dict.get('train_prediction_index', None)
        self.PASS_ALL_INPUTS = config_dict.get('pass_all_inputs', getattr(self.calculator, 'PASS_ALL_INPUTS', False))
        if self.label_type is None or self.PASS_ALL_INPUTS:
            calculate = lambda batch: self.calculator(**{k: v for k, v in batch.items() if k != 'x'})
        elif pred_idx is None:
            def calculate(batch):
                pred = batch['prediction']
                target = batch[self.label_type]
                return self.calculator(pred, target)
        else:
            def calculate(batch):
                pred = batch['predictions'][pred_idx]
                target = batch[self.label_type]
                return self.calculator(pred, target)
        self.calculate : Callable[[Dict[str, torch.Tensor]], torch.Tensor] = calculate
        
        self.accumulate = accumulate
        if accumulate:
            # initialise a counter for batch fragments
            # (images loaded into memory at once that do not make up a full batch)
            self.num_batch_fragments = 0
            self.acc_value = 0
        
        self.train = True
    
    def calculate_batch(self, batch : Dict[str, torch.Tensor], cumulate = True, train = True,
                        average = True, accumulation_scale = 1, last = False) -> Dict[str, float]:
        """
        Calculates loss value at a batch fragment.

        Parameters:
            `batch`: the dictionary containing the batched data
            `cumulate`: whether to add the current loss to a moving average (over either batch fragments or epoch)
            `train`: whether the model is in training mode; if True, gradient will be computed for backward
            `average`: whether to use averaging over the batch fragments
            `accumulation_scale`: the virtual batch size / actual batch size ratio
        """
        self.train = train
        loss = self.calculate(batch)
        if average:
            loss = loss / accumulation_scale
        value = loss.item()
        if train and not last:
            loss.backward()
        if cumulate:
            if self.accumulate:
                self.acc_value += value
                self.num_batch_fragments += 1
            else:
                self.value += value
                self.num_batches += 1
        return {self.name: value}
    
    def evaluate_batch(self, *args, cumulate = True, flush = True, **kwargs) -> Dict[str, float]:
        """Function to be called at the last batch fragment of the batch."""
        value = self.acc_value if self.accumulate else self.value
        if flush:
            self.num_batch_fragments = 0
            self.acc_value = 0
        if cumulate:
            self.value += value
            self.num_batches += 1
        return {self.name: value}

    def evaluate_epoch(self, *args, flush = True, average = True, **kwargs) -> Dict[str, float]:
        """Function to be called at the and of the epoch."""
        value = self.value
        if average and self.num_batches > 0:
            value = value / self.num_batches
        if flush:
            self.value, self.num_batches = 0, 0
        return {self.name: value}