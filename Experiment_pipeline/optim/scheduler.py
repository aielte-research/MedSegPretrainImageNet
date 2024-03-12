import inspect

from typing import Dict, Iterable, Literal, Optional, Union
from torch import optim
from optim.optimizer import Optimizer

import utils

class SchedulerWrapper(optim.lr_scheduler._LRScheduler):
    
    """Wrapper class for all learning rate scheduler objects. Do not initialise this class directly."""
    
    ITERATION_UNIT = 'epoch'

    @staticmethod
    def fill_kwargs(config_dict):
        config_dict.get_or_update('iteration_unit', SchedulerWrapper.ITERATION_UNIT)
    
    def __init__(self, scheduler_const, config_dict, optimizer = None,
                 num_epochs = None, batches_per_epoch = 1, *args, **kwargs):
        self.optimizer = optimizer
        scheduler_kwargs = utils.get_kwargs(scheduler_const, config_dict)
        if getattr(scheduler_const, 'LENGTH_DEPENDENT', False):
            scheduler_kwargs.update({'num_epochs': num_epochs, 'batches_per_epoch': batches_per_epoch})
        if 'base' not in inspect.signature(scheduler_const).parameters:
            scheduler_kwargs.pop('base', None)
        if 'iteration_unit' not in inspect.signature(scheduler_const).parameters:
            scheduler_kwargs.pop('iteration_unit', None)
        self.scheduler = scheduler_const(optimizer = optimizer, **scheduler_kwargs)
        iter_unit = config_dict.get('iteration_unit') or self.ITERATION_UNIT
        if isinstance(iter_unit, utils.config_dict.ConfigDict):
            iter_unit = iter_unit.key()
        self.batch_update = iter_unit == 'batch'
        self.epoch_update = iter_unit == 'epoch'

    def step(self, *args, **kwargs):
        self.scheduler.step()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)

class ConstantLR(optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer : optim.Optimizer, base : Optional[int] = None, *args, **kwargs):
        """Sets the learning rate for all parameters to `base`, and does not change them. If `base` is not give or is None, uses the current learning rate parameters."""
        
        super().__init__(optimizer = optimizer)
        if base is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = base
        
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def step(self, *args, **kwargs):
        return
    
class SequentialLR(optim.lr_scheduler.SequentialLR): # TODO: step (maybe you want to step at every batch at first, but later only at every epoch)
    # as a first step you could just make sure every `iteration_unit` is the same as with the super scheduler
    # also check out poly lr scheduling, that doesn't work with this scheduler
    
    """Strings together several different learning rate schedulers. At each milestone, it switches to the next scheduler."""
    
    LENGTH_DEPENDENT = True
    
    @staticmethod
    def fill_kwargs(config_dict):
        for scheduler_dict in config_dict.elements_of('schedulers'):
            utils.fill_dict(scheduler_dict)
            scheduler_dict.pop('iteration_unit', None)
    
    def __init__(self,
                 optimizer : optim.Optimizer,
                 schedulers : Iterable[utils.config_dict.ConfigDict],
                 milestones : Iterable[int],
                 iteration_unit : Literal['batch', 'epoch'] = 'batch',
                 milestones_unit : Literal['batch', 'epoch'] = 'epoch',
                 *args, **kwargs):
        
        """
        Arguments:
            `schedulers`: tuple of ConfigDicts; each dictionary should contain the specification of a learning rate scheduler
            `milestones`: tuple of ints; milestones when to switch to the next scheduler; should be of length `len(schedulers) - 1`
            `iteration_unit`: one of 'batch' or 'epoch'; at what iteration should the scheduler be stepped
            `milestones_unit`: one of 'batch' or 'epoch'; specifies the unit of measurement used to give the milestones
            `args`, `kwargs`: positional and keyword arguments that will be passed onto the scheduler constructors
        """
        
        if not isinstance(milestones_unit, str):
            msg = f'Argument `milestones_unit` should be str, not {type(milestones_unit)} ({milestones_unit}).'
            raise TypeError(msg)
        if milestones_unit not in ('batch', 'epoch'):
            msg = f'Argument `milestones_unit` should be either \'batch\' or \'epoch\', not \'{milestones_unit}\'.'
            raise ValueError(msg)
        
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        if not isinstance(milestones, (list, tuple)):
            milestones = [milestones]
        
        if milestones_unit == 'epoch':
            epoch_milestones = milestones
            if iteration_unit == 'batch':
                batches_per_epoch = kwargs['batches_per_epoch']
                milestones = [milestone * batches_per_epoch for milestone in milestones]
        else:
            batches_per_epoch = kwargs['batches_per_epoch']
            epoch_milestones = [milestone // batches_per_epoch for milestone in milestones]
        
        
        default_end = epoch_milestones[-1] + 1
        end = kwargs.pop('num_epochs', default_end) or default_end
        starts, ends = [0, *epoch_milestones], [*epoch_milestones, end]
        
        schedulers = [utils.create_object_from_dict(scheduler_dict, wrapper_class = SchedulerWrapper,
                                                    optimizer = optimizer, num_epochs = end - start,
                                                    *args, **kwargs)
                      for scheduler_dict, start, end in zip(schedulers, starts, ends)]
        schedulers = [getattr(scheduler, 'scheduler', scheduler) for scheduler in schedulers]
        
        super().__init__(optimizer, schedulers, milestones)
        

class WarmUpScheduler(object):
    """
    Wrapper object for learning rate schedulers.
    
    Arguments:
        `lr`: float; initial learning rate
        `optimizer`: torch.optim.Optimizer; optimizer object
        `warmup`: bool; whether to do a constant (usually large) warmup learning rate for a first few epochs
        `warmup_length`: int; how long the warmup should be
        `warmup_lr`: float; learning rate during the warmup
        `main_scheduler`: config dict describing learning rate scheduler to be called after warmup is done
    """

    LENGTH_DEPENDENT = True

    PARAMS = {
        'warmup': {
            'learning_rate': 0.1,
            'length': 1
            },
        'base': {
            'default': 0.01,
            'argument name': 'lr'
        },
        'main_scheduler': None,
        'iteration_unit': 'epoch'
    }
    
    STATE_DICT_ENTRIES = ('in_warmup_phase', 'warmup_length', 'warmup_lr', 'base_lr', 'last_step')

    @staticmethod
    def fill_kwargs(config_dict : utils.ConfigDict):
        main_scheduler = config_dict.get('main_scheduler')
        if main_scheduler is not None:
            utils.fill_dict(main_scheduler)
        config_dict['warmup'].fill_with_defaults(WarmUpScheduler.PARAMS['warmup'])
        config_dict.get_or_update('iteration_unit', SchedulerWrapper.ITERATION_UNIT)

    def __init__(self,
                 lr : float,
                 optimizer : Optimizer,
                 warmup : Union[Literal[False], utils.ConfigDict] = False,
                 main_scheduler : Optional[utils.ConfigDict] = None,
                 iteration_unit : Literal['epoch', 'batch'] = 'epoch',
                 num_epochs = None, batches_per_epoch = 1,
                 *args, **kwargs):
        
        if warmup:
            self.warmup_length, self.warmup_lr = warmup['length'], warmup['learning_rate']
        else:
            self.warmup_length, self.warmup_lr = 0, lr

        
        if main_scheduler:
            if iteration_unit == 'epoch':
                num_epochs = num_epochs and num_epochs - warmup['length']
            elif iteration_unit == 'batch':
                num_epochs = num_epochs and num_epochs - (warmup['length'] // batches_per_epoch)
            wrapper = utils.create_object_from_dict(main_scheduler, wrapper_class = SchedulerWrapper,
                                                    optimizer = optimizer,
                                                    num_epochs = num_epochs,
                                                    batches_per_epoch = batches_per_epoch)
            self.main_scheduler = getattr(wrapper, 'scheduler', wrapper)
        else:
            self.main_scheduler = None
        
        self.optim = optimizer
        self.last_step = 0
        self.base_lr = lr
        
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.warmup_lr
        
        self.in_warmup_phase = True
        
    
    def step(self, *args, **kwargs):
        if self.last_step == self.warmup_length:
            self.in_warmup_phase = False
            for param_group in self.optim.param_groups:
                param_group['lr'] = self.base_lr
        if not self.in_warmup_phase and self.main_scheduler:
            self.main_scheduler.step(*args, **kwargs)
        self.last_step += 1

    def state_dict(self):
        state_dict = {kw: getattr(self, kw) for kw in self.STATE_DICT_ENTRIES}
        if self.main_scheduler:
            state_dict['main_scheduler_dict'] = self.main_scheduler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        for kw in self.STATE_DICT_ENTRIES:
            setattr(self, state_dict[kw])
        self.main_scheduler.load_state_dict(state_dict['main_scheduler_dict'])

class LinearLR(optim.lr_scheduler.LinearLR):
    
    """Acts the same as torch.optim.lr_scheduler.LinearLR. If the `total_iters` argument is set to 'auto', it will calculate the number of iterations based on the `num_epochs`, `batches_per_epoch`, and `iteration_unit` arguments."""
    
    LENGTH_DEPENDENT = True
    
    def __init__(self,
                 optimizer : Optimizer,
                 start_factor : float = 1.0e-6,
                 end_factor : float = 1.0,
                 total_iters : Union[int, Literal['auto']] = 'auto',
                 iteration_unit : Literal['epoch', 'batch'] = 'batch',
                 *args, **kwargs
                 ):
        if isinstance(total_iters, utils.config_dict.ConfigDict):
            total_iters = total_iters.key()
        
        if total_iters == 'auto':
            num_epochs = kwargs['num_epochs']
            if not isinstance(num_epochs, int):
                raise TypeError(f'For poly learning rate scheduling, number of epochs must be integer, not {num_epochs}.')
            if iteration_unit == 'batch':
                total_iters = num_epochs * kwargs['batches_per_epoch']
            elif iteration_unit == 'epoch':
                total_iters = num_epochs
            else:
                raise ValueError(f'Iteration unit must be either \'batch\' or \'epoch\', not \'{iteration_unit}\'.')
        elif not isinstance(total_iters, int):
            msg = f'Argument `total_iters` must be int or \'auto\', but got {type(total_iters)} ({total_iters}).'
            raise TypeError(msg)
        
        super().__init__(optimizer, start_factor, end_factor, total_iters)

class LinearWarmupScheduler(optim.lr_scheduler._LRScheduler):
    
    PARAMS = {'warmup_length': 20, 'main_scheduler': None, 'iteration_unit': 'epoch'}
    
    @staticmethod
    def fill_kwargs(config_dict : utils.ConfigDict):
        main_scheduler = config_dict.get('main_scheduler')
        if main_scheduler is not None:
            utils.fill_dict(main_scheduler)
    
    def __init__(self,
                 optimizer : Optimizer,
                 warmup_length : int = 20,
                 main_scheduler : Optional[utils.ConfigDict] = None,
                 iteration_unit : Literal['epoch', 'batch'] = 'epoch',
                 num_epochs = None, batches_per_epoch = None,
                 *args, **kwargs):
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer = optimizer,
                                                       start_factor = 1.0e-6, end_factor = 1,
                                                       total_iters = warmup_length)

        
        if main_scheduler:
            if iteration_unit == 'epoch':
                num_epochs = num_epochs and num_epochs - warmup_length
            elif iteration_unit == 'batch':
                num_epochs = num_epochs and num_epochs - (warmup_length // batches_per_epoch)
                
            wrapper = utils.create_object_from_dict(main_scheduler, wrapper_class = SchedulerWrapper,
                                                    optimizer = optimizer,
                                                    num_epochs = num_epochs,
                                                    batches_per_epoch = batches_per_epoch)
            
            main_scheduler = getattr(wrapper, 'scheduler', wrapper)
            
            schedulers = [warmup_scheduler, main_scheduler]
            milestones = [warmup_length]
            self.scheduler = optim.lr_scheduler.SequentialLR(optimizer = optimizer,
                                                             schedulers = schedulers,
                                                             milestones = milestones)
        else:
            self.scheduler = warmup_scheduler
    
    def step(self, *args, **kwargs):
        self.scheduler.step()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)

class StepLearningRateScheduler(optim.lr_scheduler._LRScheduler): # TODO: test
    """
    Changes the learning rate at certain epochs, then keeps them constant until the next change comes up.
    Arguments:
        `init_lr`: float; initial learning rate
        `learning_rates_dict`: dict[int, float]; dictionary that contains the index of epochs when the learning rate should be changed, and the learning rate starting from that epoch
    """
    PARAMS = {
        'base': {
            'default': 0.01,
            'argument name': 'init_lr'
        },
        'learning_rates_dict': {}
    }

    def __init__(self,
                 optimizer : optim.Optimizer,
                 init_lr : float,
                 learning_rates_dict : Dict[int, float],
                 *args, **kwargs):
        
        self.optimizer = optimizer
        self.base_lr = init_lr
        self.milestones = learning_rates_dict
        self.step_count = 0
        
    def step(self, *args, **kwargs):
        self.step_count += 1
        if self.step_count in self.milestones:
            lr = self.milestones[self.step_count]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class PolyLearningRateDecay(optim.lr_scheduler._LRScheduler):
    """
    Decays the learning rate in the ith iteration by `(1 - i / number_of_iterations) ** exponent`.
    Arguments:
        `base`: float; learning rate in the 0th iteration
        `number_of_iterations`: int or 'auto'; total number of iterations. If set to 'auto', the number of iterations will be automatically computed
        `exponent`: float
        `minimum`: optional int; minimal learning rate; if set to `None`, the learning rate decays indefinitely (in the last iteration it will reach `1 / number_of_iterations ** exponent`)
        `iteration_unit`: 'batch' or 'epoch'; what counts as one step for the scheduler
        `num_epochs`: optional int; used to calculate the total number of iterations if `number_of_iterations` is set to 'auto'
        `batches_per_epoch`: optional int; used to calculate the total nuber of iterations if `number_of_iterations` is set to 'auto' and `iteration_unit` is set to 'batch'
    """

    LENGTH_DEPENDENT = True

    def __init__(self,
                 optimizer : optim.Optimizer,
                 number_of_iterations : Union[int, Literal['auto']] = 'auto',
                 exponent : float = 0.9,
                 minimum : Optional[int] = None,
                 iteration_unit : Literal['batch', 'epoch'] = 'epoch',
                 last_epoch = -1,
                 *args, **kwargs):

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        
        self.min_lr = minimum or 0
        self.cap_decay = False
        self.gamma = exponent

        if number_of_iterations == 'auto':
            num_epochs = kwargs['num_epochs']
            if not isinstance(num_epochs, int):
                raise TypeError(f'For poly learning rate scheduling, number of epochs must be integer, not {num_epochs}.')
            if iteration_unit == 'batch':
                number_of_iterations = num_epochs * kwargs['batches_per_epoch']
            elif iteration_unit == 'epoch':
                number_of_iterations = num_epochs
            else:
                raise ValueError(f'Iteration unit must be either \'batch\' or \'epoch\', not \'{iteration_unit}\'.')
        
        self.num_iters = number_of_iterations
        
        self._last_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        self.base_lr = [param_group.get('initial_lr', param_group.get('lr', 0)) for param_group in self.optimizer.param_groups]
    
    def step(self, *args, **kwargs):
        if self.cap_decay:
            return
        cap_decay = True
        self._last_lr = []
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lr):
            new_lr = ((1 - self.last_epoch / self.num_iters) ** self.gamma) * base_lr
            lr = max(new_lr, self.min_lr)
            param_group['lr'] = lr
            self._last_lr.append(lr)
            self.last_epoch += 1
            cap_decay = cap_decay and (lr == self.min_lr)
        self.cap_decay = cap_decay
            

class CosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    
    """Acts the same as torch.optim.lr_scheduler.CosineAnnealingLR. If the `T_max` argument is set to 'auto', it will calculate the number of iterations based on the `num_epochs`, `batches_per_epoch`, and `iteration_unit` arguments."""
    
    LENGTH_DEPENDENT = True
    
    def __init__(self, optimizer, T_max = 'auto', eta_min = 0, iteration_unit = 'batch', *args, **kwargs):
        if T_max == 'auto':
            num_epochs = kwargs['num_epochs']
            if not isinstance(num_epochs, int):
                raise TypeError(f'For cosine annealing learning rate scheduling, number of epochs must be integer, not {num_epochs}.')
            if iteration_unit == 'batch':
                T_max = num_epochs * kwargs['batches_per_epoch']
            elif iteration_unit == 'epoch':
                T_max = num_epochs
            else:
                raise ValueError(f'Iteration unit must be either \'batch\' or \'epoch\', not \'{iteration_unit}\'.')
        
        super().__init__(optimizer, T_max, eta_min)
        
class ExponentialLR(optim.lr_scheduler.ExponentialLR):
    
    """Acts the same as torch.optim.lr_scheduler.ExponentialLR. If the `gamma` argument is set to 'auto', a `min_scale` should be specified (the scaling factor of the learning rate after the last epoch), and it will calculate the number of iterations based on the `num_epochs`, `batches_per_epoch`, `iteration_unit`, and `min_scale` arguments."""
    
    LENGTH_DEPENDENT = True
    
    DEFAULT_MIN_SCALE = 0.001
    
    @staticmethod
    def fill_kwargs(config_dict):
        if config_dict.get_str('gamma') == 'auto':
            config_dict.get_or_update('min_scale', ExponentialLR.DEFAULT_MIN_SCALE)
    
    def __init__(self, optimizer, gamma = 'auto', iteration_unit = 'batch', min_scale = None, *args, **kwargs):
        
        if gamma == 'auto':
            num_epochs = kwargs['num_epochs']
            if not isinstance(num_epochs, int):
                pass
            if iteration_unit == 'batch':
                num_iters = num_epochs * kwargs['batches_per_epoch']
            elif iteration_unit == 'epoch':
                num_iters = num_epochs
            else:
                raise ValueError(f'Iteration unit must be either \'batch\' or \'epoch\', not \'{iteration_unit}\'.')
            gamma = min_scale ** (1 / num_iters)
        
        super().__init__(optimizer, gamma)

class GaussianLRDecay(optim.lr_scheduler._LRScheduler):
    
    """Sets the learning rate at iteration `k` to `base_lr * gamma ** (k**2). Parameter `gamma` is calculated so that by the end of training the optimizer reaches `min_scale`."""
    
    LENGTH_DEPENDENT = True
    
    def __init__(self,
                 optimizer : optim.Optimizer,
                 min_scale : float = 0.001,
                 number_of_iterations : Union[Literal['auto'], int] = 'auto',
                 iteration_unit : Literal['batch', 'epoch'] = 'batch',
                 *args, **kwargs):
        
        if number_of_iterations == 'auto':
            num_epochs = kwargs['num_epochs']
            if not isinstance(num_epochs, int):
                raise TypeError(f'For Gaussian learning rate scheduling, number of epochs must be integer, not {num_epochs}.')
            if iteration_unit == 'batch':
                number_of_iterations = num_epochs * kwargs['batches_per_epoch']
            elif iteration_unit == 'epoch':
                number_of_iterations = num_epochs
            else:
                raise ValueError(f'Iteration unit must be either \'batch\' or \'epoch\', not \'{iteration_unit}\'.')

        self.gamma = min_scale ** (1 / number_of_iterations ** 2)
        
        self.scale = self.gamma
        self.gamma_sq = self.gamma ** 2
        self.iter_count = 0
        self.total_iters = number_of_iterations
        self.stop_decay = False

        self.optimizer = optimizer
    
    def step(self):
        self.iter_count += 1
        if self.stop_decay:
            return
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.scale
        
        self.scale = self.scale * self.gamma_sq
        self.stop_decay = self.iter_count == self.total_iters