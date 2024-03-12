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

class CombinedLoss(Loss):
    """Wrapper object for using a linear combination of multiple loss functions and regularisations."""

    PARAMS = {}

    @staticmethod
    def fill_kwargs(config_dict : ConfigDict):
        """Fill a ConfigDict with the default hyperparameters of the different losses and the weight update rule."""
        for key, loss_dict in config_dict.items():
            if key == 'weight update rule':
                continue
            utils.fill_dict(loss_dict, class_path = key)
            loss_dict.get_or_update('initial weight', 1.0)
        update_rule = config_dict.get_or_update('weight update rule', 'constant', final = False)
        if update_rule.key() != 'constant':
            utils.fill_dict(update_rule)

    def __init__(self, config_dict : ConfigDict, accumulate = True, *args, **kwargs):
        """Initialises a CombinedLoss instance from a ConfigDict containing all the losses and their corresponding hyperparameters."""

        self.name = 'combined_loss'
        self.num_batches = 0
        self.value = 0
        
        self.accumulate = accumulate
        if accumulate:
            # initialise a counter for batch fragments
            # (images loaded into memory at once that do not make up a full batch)
            self.num_batch_fragments = 0
            self.acc_value = 0
        
        self.train = True
        
        losses, weights, names, label_types, state_updates = [], [], [], [], []

        # create the individual losses
        for key, loss_dict in config_dict.items():
            if key in ('weight update rule', 'weight_update_rule'):
                continue
            label_type = loss_dict.get('label type')
            label_types.append(label_type)
            loss = utils.create_object_from_dict(config_dict[key].mask('initial_weight'), wrapper_class = Loss, class_path = key)
            losses.append(getattr(loss, 'calculate', loss))
            names.append(getattr(loss, 'name', Loss.convert_to_snake(type(loss).__name__)))
            weights.append(loss_dict['initial weight'])
            state_updates.append(getattr(loss, 'update_state', lambda *args, **kwargs: None))

        self.names = names
        self.calculators = losses
        self.weights = weights
        self.label_types = label_types
        self.state_updates = state_updates
        update_rule_dict = config_dict['weight update rule']
        if isinstance(update_rule_dict, str):
            update_rule_dict = utils.config_dict.ConfigDict({update_rule_dict: {}})
        if update_rule_dict.key() == 'constant':
            self.update_rule = None
        else:
            self.update_rule = utils.create_object_from_dict(update_rule_dict, loss_names = self.names)

        self.num_epochs = 0

        self.values = [0 for _ in names]
        if accumulate:
            self.acc_values = [0 for _ in names]
    
    def calculate(self, batch):
        return sum(weight * calc(batch) for weight, calc in zip(self.weights, self.calculators))

    def calculate_batch(self, batch, cumulate = True, train = True,
                        average = True, accumulation_scale = 1, last = False) -> Dict[str, float]:
        """
        Calculates loss value at a batch fragment. Returns a dictionary with their combined values, as well as the value of the individual losses.

        Parameters:
            `batch`: the dictionary containing the batched data
            `cumulate`: whether to add the current loss to a moving average (over either batch fragments or epoch)
            `train`: whether the model is in training mode; if True, gradient will be computed for backward
            `average`: whether to use averaging over the batch fragments
            `accumulation_scale`: the virtual batch size / actual batch size ratio
        """

        self.train = train
        losses = []
        for calc in self.calculators:
            losses.append(calc(batch))
            
        if average:
            losses = [loss / accumulation_scale for loss in losses]
        loss = sum(self.weights[i] * loss for i, loss in enumerate(losses))
        value, losses = loss.item(), [loss_term.item() for loss_term in losses]
        if train and not last:
            loss.backward()
        if cumulate:
            if self.accumulate:
                self.acc_values = [acc_value + losses[i] for i, acc_value in enumerate(self.acc_values)]
                self.acc_value += value
                self.num_batch_fragments += 1
            else:
                self.values = [curr_value + losses[i] for i, curr_value in enumerate(self.values)]
                self.value += value
                self.num_batches += 1
        values_dict = {self.names[i]: losses[i] for i in range(len(losses))}
        values_dict.update({self.name: value})
        return values_dict

    def evaluate_batch(self, cumulate = True, flush = True, *args, **kwargs) -> Dict[str, float]:
        """Function to be called at the last batch fragment of the batch."""
        value = self.acc_value if self.accumulate else self.value
        values = self.acc_values if self.accumulate else self.values
        if flush:
            self.num_batch_fragments = 0
            self.acc_value = 0
            self.acc_values = [0 for _ in self.names]
        if cumulate:
            self.value += value
            self.values = [curr_value + values[i] for i, curr_value in enumerate(self.values)]
            self.num_batches += 1
        values_dict = {self.names[i]: value for i, value in enumerate(values)}
        values_dict.update({self.name: value})
        [update_state() for update_state in self.state_updates]
        return values_dict

    def evaluate_epoch(self, flush = True, average = True, last = False, *args, **kwargs) -> Dict[str, float]:
        """Function to be called at the and of the epoch."""
        value = self.value
        values = self.values
        if average:
            value = value / self.num_batches
            values = [value / self.num_batches for value in values]
        if flush:
            self.value, self.num_batches = 0, 0
            self.values = [0 for _ in self.names]
        
        if self.train and self.update_rule and not last:
            self.num_epochs += 1
            self.weights = self.update_rule(self.num_epochs, self.weights)

        values_dict = {self.names[i]: value for i, value in enumerate(values)}
        values_dict.update({self.name: value})
        return values_dict

class ConstantLoss(torch.nn.Module):
    
    def __init__(self, value = 0, *args, **kwargs):
        super().__init__()
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.value = value
    
    def forward(self, prediction, *args, **kwargs):
        value = torch.clone(self.value)
        return value.requires_grad_(True)

class CutOffLoss(Loss):

    PARAMS = {'loss': 'regression.losses.AbsoluteOrRelativeLoss', 'min': 0, 'max': None}

    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['loss'])
        reduction = config_dict['loss'].value().pop('reduction', WeightedLoss.PARAMS['reduction'])
        config_dict.get_or_update('reduction', reduction)

    def __init__(self, config_dict : ConfigDict, *args, **kwargs):
        loss_dict = config_dict['loss']
        super().__init__(*utils.get_class_constr_and_dict(loss_dict), reduction = 'none', *args, **kwargs)
        
        self.min, self.max = config_dict['min'], config_dict['max']
        if self.min is None:
            self.min = -torch.inf 
        if self.max is None:
            self.max = torch.inf
        
        self.reduce = Loss.REDUCTION_METHODS[config_dict['reduction']]
        unreduced_calc = self.calculate
        def reduced_calc(batch):
            unreduced_loss = unreduced_calc(batch)
            zeros = torch.zeros_like(unreduced_loss)
            min_cut_loss = torch.maximum(unreduced_loss, self.min + zeros)
            cut_off_loss = torch.minimum(min_cut_loss, self.max + zeros)
            return self.reduce(cut_off_loss)
        self.calculate = reduced_calc
        
        self.base_name = self.name
        self.name = f'cut_off_{self.base_name}'    
     
class WeightedLoss(Loss):
    """
    Wrapper for loss functions that handles gradient accumulation.

    Parameters:
        name: str; name of the loss function that will appear on plots
        loss_fn: callable object that returns an instance of the loss function
        accumulate: bool; whether to use gradient accumulation; default: True
        args, kwargs: positional and keyword arguments that will be passed onto loss_fn
        label_name: name of the label that the loss should be applied to (eg. 'mask', 'distance map', 'label')
    """

    PARAMS = {'loss': 'regression.losses.AbsoluteOrRelativeLoss', 'norm': False, 'reduction': 'mean'}

    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['loss'])
        reduction = config_dict['loss'].value().pop('reduction', WeightedLoss.PARAMS['reduction'])
        config_dict.get_or_update('reduction', reduction)

    def __init__(self, config_dict : ConfigDict, *args, **kwargs):
        # calculator: loss function wrapped
        # loss: loss with gradient
        # value: value of loss accumulated over multiple batches; no gradient

        loss_dict = config_dict['loss'].copy()
        if 'label_type' in config_dict:
            loss_dict.value()['label_type'] = config_dict['label_type']
        loss_dict.value()['reduction'] = config_dict['reduction']
            
        self.base_loss = utils.create_object_from_dict(loss_dict, wrapper_class = Loss, *args, **kwargs)
        self.weighted_loss = utils.create_object_from_dict(loss_dict.mask(f'{loss_dict.key()}/reduction'), wrapper_class = Loss, reduction = 'none',*args, **kwargs)
        
        
        self.reduce = Loss.REDUCTION_METHODS[config_dict['reduction']]
        unreduced_calc = self.weighted_loss.calculate
        def reduced_calc(batch):
            unreduced_loss = unreduced_calc(batch)
            weights = batch['weight']
            reduced = self.reduce(weights * unreduced_loss)
            if config_dict['norm']:
                return reduced / weights.sum()
            else:
                return reduced
        self.weighted_loss.calculate = reduced_calc
            
        
        self.base_name = getattr(self.base_loss, 'name', Loss.convert_to_snake(type(self.base_loss).__name__))
        self.name = f'weighted_{self.base_name}'
        
    def calculate_batch(self, batch, train = True, *args, **kwargs) -> Dict[str, float]:
        values = {}
        with torch.no_grad():
            values.update(
                self.base_loss.calculate_batch(batch, train = False, *args, **kwargs)
            )
        values.update({
            f'weighted_{loss_name}': value for loss_name, value in self.weighted_loss.calculate_batch(batch, train = train, *args, **kwargs).items()
        })
        
        return values
    
    def evaluate_iter(self, eval_function, *args, **kwargs):
        values = {
            **getattr(self.base_loss, eval_function)(*args, **kwargs),
            **{f'weighted_{loss_name}': value for loss_name, value in getattr(self.weighted_loss, eval_function)(*args, **kwargs).items()}
        }
        
        return values
    
    def evaluate_epoch(self, *args, **kwargs):
        return self.evaluate_iter('evaluate_epoch', *args, **kwargs)
    
    def evaluate_batch(self, *args, **kwargs):
        return self.evaluate_iter('evaluate_batch', *args, **kwargs)

class IgnoreNaNLoss(Loss):

    PARAMS = {'loss': 'torch.nn.BCELoss', 'reduction': 'mean'}

    @staticmethod
    def fill_kwargs(config_dict : ConfigDict):
        utils.fill_dict(config_dict['loss'])
        reduction = config_dict['loss'].value().pop('reduction', IgnoreNaNLoss.PARAMS['reduction'])
        reduction = config_dict.get_or_update('reduction', reduction)
        if isinstance(reduction, ConfigDict):
            reduction = reduction.key()
        if reduction == 'mean':
            config_dict.get_or_update('true_mean', False)
        elif reduction == 'none':
            config_dict.get_or_update('nan_value', 0.)

    def __init__(self, config_dict : ConfigDict, *args, **kwargs):
        loss_dict = config_dict['loss']
        if isinstance(loss_dict, str):
            loss_dict = ConfigDict({loss_dict: {}})
        if 'label_type' in config_dict:
            loss_dict.value()['label_type'] = config_dict['label_type']
            
        loss_fn = utils.create_object_from_dict(loss_dict, wrapper_class = Loss, reduction = 'none', *args, **kwargs)
        super().__init__(lambda *args, **kwargs: loss_fn.calculator, config_dict, *args, **kwargs)
        
        unreduced_calc = loss_fn.calculate
        
        reduction = config_dict.get_str('reduction')
        if reduction == 'none':
            self.reduce = lambda x, _: x
        elif reduction == 'sum':
            self.reduce = lambda x, mask: torch.sum(x[mask])
        elif reduction == 'mean':
            if config_dict['true_mean']:
                self.reduce = lambda x, mask: torch.mean(x[mask])
            else:
                self.reduce = lambda x, mask: torch.sum(x[mask]) / x.numel()
        else:
            raise ValueError(f'Argument `reduction` should be one of \'none\', \'sum\', or \'mean\', but got \'{reduction}\'.')
        
        nan_value = config_dict.get('nan_value') or 0
        def reduced_calc(batch): # TODO: when label_type is None or batch has 'predictions' key
            target = batch[self.label_type]
            mask = ~torch.isnan(target)
            unreduced = unreduced_calc({**batch, self.label_type: torch.nan_to_num(target, nan = nan_value)})
            return self.reduce(unreduced, mask)
        self.calculate = reduced_calc
        
        self.name = loss_fn.name


class LossWithConfidence(Loss):
    
    PARAMS = {'main_loss': 'regression.losses.AbsoluteOrRelativeLoss',
              'confidence_loss_initial_weight': 1.0,
              'hint_probability': 0.5,
              'budget_parameter': 0.1,
              'weight_adjustion_scale': 0.01
              }
    
    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['main_loss'])

    def __init__(self, config_dict : ConfigDict, accumulate = True, *args, **kwargs):
        # calculator: loss function wrapped
        # loss: loss with gradient
        # value: value of loss accumulated over multiple batches; no gradient

        loss_dict = config_dict['main_loss'].copy()
        if 'label_type' in config_dict:
            loss_dict.value()['label_type'] = config_dict['label_type']
        
        self.label_type = config_dict.get('label_type', Loss.PARAMS['label type'])
            
        loss = utils.create_object_from_dict(loss_dict, wrapper_class = Loss, *args, **kwargs)
        self.loss = getattr(loss, 'calculator', loss)
        
        self.base_name = getattr(loss, 'name', Loss.convert_to_snake(type(loss).__name__))
        self.main_name = f'{self.base_name}_with_confidence_hints'
        self.name = 'confidence_regulated_loss'
        self.conf_name = 'confidence_loss'
        
        
        self.p = config_dict['hint_probability']
        self.lda = config_dict['confidence_loss_initial_weight']
        
        beta = config_dict['budget_parameter']
        delta = config_dict['weight_adjustion_scale']
        up_delta = 1 - delta
        down_delta = 1 + delta
        
        def update_lda(confidence_loss):
            if confidence_loss < beta:
                self.lda = self.lda / down_delta
            else:
                self.lda = self.lda / up_delta
        
        self.update_lambda = update_lda if delta != 0 else lambda _: _
        
        self.base_value = 0
        self.main_value = 0
        self.conf_loss_value = 0
        self.conf_value = 0
        self.value = 0
        self.num_batches = 0
        
        self.accumulate = accumulate
        if accumulate:
            # initialise a counter for batch fragments
            # (images loaded into memory at once that do not make up a full batch)
            self.num_batch_fragments = 0
            self.acc_main_value = 0
            self.acc_base_value = 0
            self.acc_conf_loss_value = 0
            self.acc_conf_value = 0
            self.acc_value = 0
        
        self.train = True
        self.curr_conf_loss = 0
    
    def calculate(self, batch):
        pred, conf = batch['predictions']
        ones = torch.ones_like(conf)
        hints = torch.bernoulli(self.p * ones).to(bool)
        coeffs = torch.where(hints, conf, ones) if self.train else ones
        
        target = batch[self.label_type]
        c_pred = coeffs * pred + (1 - coeffs) * target
        
        confidence_loss = -torch.log(conf).mean()
        main_loss = self.loss(c_pred, target)
        
        loss = main_loss + self.lda * confidence_loss
        self.curr_conf_loss = confidence_loss
        
        return loss
        
       
    def calculate_batch(self, batch, train = True, average = True, accumulation_scale = 1, cumulate = True, last = False, *args, **kwargs) -> Dict[str, float]:
        
        self.train = train
        
        pred, conf = batch['predictions']
        ones = torch.ones_like(conf)
        hints = torch.bernoulli(self.p * ones).to(bool)
        coeffs = torch.where(hints, conf, ones) if train else ones
        
        target = batch[self.label_type]
        c_pred = coeffs * pred + (1 - coeffs) * target
        
        with torch.no_grad():
            base_value = self.loss(pred, target).item()
        
        confidence_loss = -torch.log(conf).mean()
        main_loss = self.loss(c_pred, target)
        conf_value = torch.mean(conf).item()
        
        if average:
            confidence_loss = confidence_loss / accumulation_scale
            main_loss = main_loss / accumulation_scale
            base_value = base_value / accumulation_scale
            conf_value = conf_value / accumulation_scale
        
        loss = main_loss + self.lda * confidence_loss
            
        value = loss.item()
        main_value = main_loss.item()
        confidence_value = confidence_loss.item()
        
        if train and not last:
            loss.backward()
        if cumulate:
            if self.accumulate:
                self.acc_value += value
                self.acc_main_value += main_value
                self.acc_base_value += base_value
                self.acc_conf_loss_value += confidence_value
                self.acc_conf_value += conf_value
                self.num_batch_fragments += 1
            else:
                self.value += value
                self.main_value += main_value
                self.base_value += base_value
                self.conf_loss_value += confidence_value
                self.conf_value += conf_value
                self.num_batches += 1
        
        return {self.name: value,
                self.main_name: main_value,
                self.base_name: base_value,
                self.conf_name: confidence_value,
                'confidence': conf_value,
                'weight_of_confidence_loss': self.lda}
    
    def evaluate_batch(self, cumulate = True, flush = True, last = False, *args, **kwargs) -> Dict[str, float]:
        """Function to be called at the last batch fragment of the batch."""
        if self.accumulate:
            value = self.acc_value
            main_value = self.acc_main_value
            base_value = self.acc_base_value
            conf_loss_value = self.acc_conf_loss_value
            conf_value = self.acc_conf_value
        else:
            value = self.value
            main_value = self.main_value
            base_value = self.base_value
            conf_loss_value = self.conf_loss_value
            conf_value = self.conf_value
            
        if flush:
            self.num_batch_fragments = 0
            self.acc_value = 0
            self.acc_main_value = 0
            self.acc_base_value = 0
            self.acc_conf_loss_value = 0
            self.acc_conf_value = 0
        if cumulate:
            self.value += value
            self.main_value += main_value
            self.base_value += base_value
            self.conf_loss_value += conf_loss_value
            self.conf_value += conf_value
            self.num_batches += 1
        
        if self.train and not last:
            self.update_lambda(conf_loss_value)
        
        return {self.name: value,
                self.main_name: main_value,
                self.base_name: base_value,
                self.conf_name: conf_loss_value,
                'confidence': conf_value,
                'weight_of_confidence_loss': self.lda}
    
    def update_state(self):
        self.update_lambda(self.curr_conf_loss)

    def evaluate_epoch(self, flush = True, average = True, *args, **kwargs) -> Dict[str, float]:
        """Function to be called at the and of the epoch."""
        value = self.value
        main_value = self.main_value
        base_value = self.base_value
        conf_loss_value = self.conf_loss_value
        conf_value = self.conf_value
        
        if average:
            value = value / self.num_batches
            main_value = main_value / self.num_batches
            base_value = base_value / self.num_batches
            conf_loss_value = conf_loss_value / self.num_batches
            conf_value = conf_value / self.num_batches
        if flush:
            self.value, self.num_batches = 0, 0
            self.main_value = 0
            self.base_value = 0
            self.conf_loss_value = 0
            self.conf_value = 0

        return {self.name: value,
                self.main_name: main_value,
                self.base_name: base_value,
                self.conf_name: conf_loss_value,
                'confidence': conf_value}