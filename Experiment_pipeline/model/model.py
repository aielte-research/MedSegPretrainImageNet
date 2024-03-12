import importlib
import inspect
import os
import types
from typing import Callable, Iterable, List, Literal, Optional, OrderedDict, Union
import warnings
from exception_handling import handle_exception
import torch

import fvcore.nn

import data
from model import weight_init

import utils
from utils.config_dict import ConfigDict

class Model(torch.nn.Module):
    """Wrapper object for models."""

    @staticmethod
    def fill_weight_init_kwargs(config_dict):
        def fill_scheme_kwargs(init_dict):
            if not os.path.isfile(init_dict.key()):
                try:
                    utils.fill_dict(init_dict)
                except ValueError:
                    return

        config_dict.expand()
        for key in ('weight_initialisation', 'weight_init'):
            if key not in config_dict:
                continue
            random = f'{key}/random' in config_dict
            if random:
                key = f'{key}/random'
            if isinstance(config_dict[key], (tuple, list)):
                for init_dict in config_dict.elements_of(key):
                    fill_scheme_kwargs(init_dict)
            elif len(config_dict[key].keys()) == 1:
                if config_dict[key].key() != 'weights':
                    fill_scheme_kwargs(config_dict[key])
            else:
                for class_dict in config_dict[key].values():
                    utils.fill_dict(class_dict)
                    
    def __init__(self,
                 model_const : Callable[..., torch.nn.Module] = None,
                 config_dict : utils.ConfigDict = None,
                 *args, **kwargs):
        """Initialises model from constructor, with pretrained weights loaded if specified in `config_dict`."""

        super().__init__()

        if model_const is None:
            return
            
        model_kwargs = utils.get_kwargs(model_const, config_dict)
        for kw in ('weight_init', 'weight_initialisation'):
            if kw not in inspect.signature(model_const).parameters:
                model_kwargs.pop(kw, None)
        self.model = model_const(*args, **kwargs, **model_kwargs)
        
        self.PASS_ALL_INPUTS = config_dict.get('pass_all_input', getattr(self.model, 'PASS_ALL_INPUTS', False))
        if torch.cuda.device_count() <= 1:
            if self.PASS_ALL_INPUTS:
                # useful for custom models where there is more than one input
                self.forward = lambda *args, **kwargs: self.model(*args, **kwargs)
            else:
                self.forward = lambda x, *args, **kwargs: self.model(x)
        else:
            if self.PASS_ALL_INPUTS:
                self.forward = self.forward_all
            else:
                self.forward = self.forward_x
    
    def forward_all(self, *args, **kwargs):
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = 'cuda'
        args = [arg.to(device) for arg in args if hasattr(arg, 'device')]
        kwargs = {kw: arg.to(device) for kw, arg in kwargs.items() if hasattr(arg, 'device')}
        return self.model(*args, **kwargs)
    
    def forward_x(self, x, *args, **kwargs):
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = 'cuda'
        return self.model(x.to(device))
            
    def init_weight(self, config_dict : utils.config_dict.ConfigDict, *args, **kwargs):
        config_dict.expand()
        weight_init : utils.ConfigDict = config_dict.get('weight_initialisation', config_dict.get('weight_init', None))
        
        if weight_init is None:
            return
        
        is_list = isinstance(weight_init, (tuple, list))
        if not is_list:
            single_key = len(weight_init.keys()) == 1
            if single_key:
                key = weight_init.key()
                is_path = os.path.isfile(key) or key[-3:] == '.pt'
            else:
                is_path = False
        else:
            single_key, is_path = True, False
            
        if 'weights' in weight_init or is_path:
            pretrained_init = weight_init.get('weights', weight_init).trim()
            strict = weight_init.get('strict', True)
        else:
            pretrained_init = None
        
        if 'random' in weight_init or 'weights' not in weight_init:
            if is_list:
                random_init = weight_init
            else:
                random_init = weight_init.trim().get('random', weight_init)
            if isinstance(random_init, (tuple, list, str)):
                random_init = utils.config_dict.ConfigDict({'otherwise': random_init})
            elif len(random_init.keys()) == 1:
                key_list = random_init.key().split('.')
                module_name, attr_name = '.'.join(key_list[:-1]), key_list[-1]
                module = importlib.import_module(module_name)
                attr = getattr(module, attr_name)
                if isinstance(attr, types.FunctionType):
                    random_init = utils.config_dict.ConfigDict({'otherwise': random_init})
        else:
            random_init = None
        
        non_init_modules = []
        
        if random_init is not None:    
            keys = list(random_init.keys())
            if 'otherwise' in keys:
                keys = [*filter(lambda k: k != 'otherwise', keys), 'otherwise']
            init_dict = {}
            
            def init_weights_w_scheme(module, init_func, init_kwargs, bias_init, is_function):
                if is_function:
                    init_func(getattr(module, 'weight', module), **init_kwargs)
                    if getattr(module, 'bias', None) is not None:
                        module.bias.data.fill_(bias_init)
                else:
                    init_func(module)
            
            def init_submodule_weights(module, init_items_list):
                for init_items in init_items_list:
                    try:
                        return init_weights_w_scheme(module, *init_items)
                    except (ValueError, AttributeError):
                        pass
                    except Exception as e:
                        msg = f'An unforeseen exception occured while trying to initialise the weights of {module._get_name()}.'
                        handle_exception(e, msg)
                non_init_modules.append(module._get_name())
            
            init_items_lists = [[] for _ in random_init.keys()]
            for layer_name, init_items_list in zip(random_init.keys(), init_items_lists):
                for init_func_dict in random_init.elements_of(layer_name):
                    try:
                        constr_or_func, c_dict = utils.get_class_constr_and_dict(init_func_dict)
                        init_kwargs = utils.get_kwargs(constr_or_func, c_dict.mask('bias_init'))
                        bias_init = c_dict.get('bias_init', 0)
                        is_function = isinstance(constr_or_func, types.FunctionType)
                        if is_function:
                            init_func = constr_or_func
                        else:    
                            if 'bias_init' in inspect.signature(constr_or_func).parameters.keys():
                                init_kwargs['bias_init'] = bias_init
                            init_func = constr_or_func(**init_kwargs)
                        init_items_list.append((init_func, init_kwargs, bias_init, is_function))
                    except ValueError:
                        msg = f'Did not recognise \'{init_func_dict.key()}\' as path to weights or random weight initialisation scheme. Ignoring entry.'
                        warnings.warn(msg)
                
                if layer_name == 'otherwise':
                    layer_type = object
                else:
                    layer_path = layer_name.split('.')
                    layer_type = getattr(importlib.import_module('.'.join(layer_path[:-1])), layer_path[-1])
                init_dict[layer_type] = init_items_list
            
            def init_weight(module):
                for layer_type, init_items_list in init_dict.items():
                    if isinstance(module, layer_type):
                        return init_submodule_weights(module, init_items_list)
                non_init_modules.append(module._get_name())
            
            self.apply(init_weight)
            if len(non_init_modules) > 0:
                non_init_modules = list(set(non_init_modules))
                non_init_modules.sort()
                modules = ', '.join(non_init_modules)
                warnings.warn(f'None of the given initialisation methods could initialise the following modules: {modules}. Leaving the module with default weights. If the module has children, they might be initialised according to a given scheme.')        
        
        if pretrained_init is not None:
            def init_weight(module, weights_path):
                state_dict = torch.load(weights_path)
                missing, _ = module.load_state_dict(state_dict, strict = False)
                for key in missing:
                    shortened_key = key.replace('.model.', '.')
                    if shortened_key in state_dict:
                        state_dict[key] = state_dict.pop(shortened_key)
                missing, unexpected = module.load_state_dict(state_dict, strict = strict)
                if len(missing) > 0:
                    missing_keys = ', '.join(missing)
                    msg = f'The following keys were missing from the state dictionary {weights_path}: {missing_keys}. The corresponding submodules will be left with randomly initialised weights.'
                    warnings.warn(msg)
                if len(unexpected) > 0:
                    unexpected_keys = ', '.join(unexpected)
                    msg = f'The following keys in state dictionary {weights_path} were not in the state dictionary of the model: {unexpected_keys}. The corresponding submodules will be left randomly initialised.'
                    warnings.warn(msg)
                print(f'Loaded model weights of {module._get_name()} from `{weights_path}`.')
            
            if isinstance(pretrained_init, ConfigDict) and len(pretrained_init.keys()) == 1 and os.path.isfile(pretrained_init.get_str()):
                pretrained_init = pretrained_init.get_str()
            if isinstance(pretrained_init, str):
                init_weight(self, pretrained_init)
            else:
                for attr_name, weights_path in pretrained_init.items():
                    init_weight(getattr(self, attr_name), weights_path)
        
        getattr(getattr(self, 'model', self), 'init_weights', lambda *args, **kwargs: None)(config_dict)    

    def freeze_and_unfreeze(self, config_dict : utils.config_dict.ConfigDict, *args, **kwargs):
        submodules_to_freeze = config_dict.get_str_tuple('weight_init/freeze_weights')
        submodules_to_unfreeze = config_dict.get_str_tuple('weight_init/unfreeze_weights')
        
        for req_grad, submodules_list in zip((False, True), (submodules_to_freeze, submodules_to_unfreeze)):
            for submodule_path in submodules_list:
                submodule = self
                if submodule_path != 'all':
                    submodule_names_list = submodule_path.split('.')
                    for i, subsubmodule_name in enumerate(submodule_names_list):
                        if hasattr(submodule, subsubmodule_name):
                            submodule = getattr(submodule, subsubmodule_name)
                        elif hasattr(submodule, 'model'):
                            submodule = getattr(submodule.model, subsubmodule_name)
                        else:
                            subpath = '.'.join(submodule_names_list[:i+1])
                            raise AttributeError(f'Model has not attribute {subpath}.')
                submodule.requires_grad_(req_grad)
    
    def state_dict(self, *args, **kwargs):
        return getattr(self, 'model', super()).state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict : OrderedDict[str, torch.Tensor], strict: bool = True):
        return getattr(self, 'model', super()).load_state_dict(state_dict, strict)
    
    def parameters(self, recurse = True):
        return getattr(self, 'model', super()).parameters(recurse)
    
    @torch.no_grad()
    def get_number_of_flops(self, ds : data.BalancedDataLoader):
        if hasattr(getattr(self, 'model', None), 'get_number_of_flops'):
            return self.model.get_number_of_flops(ds)
        all_inputs = next(iter(ds))
        signature = inspect.signature(getattr(self, 'model', self).forward)
        input = tuple(all_inputs[name][:1] for name in signature.parameters if name in all_inputs)
        if len(input) == 0:
            input = all_inputs['x'][:1]
        return fvcore.nn.FlopCountAnalysis(self, input).total()
    
    def get_num_params(self, trainable_only = False):
        if hasattr(getattr(self, 'model', None), 'get_num_params'):
            return self.model.get_num_params(trainable_only = trainable_only)
        # code source: https://stackoverflow.com/a/62764464
        if trainable_only:
            condition = lambda p: p.requires_grad
        else:
            condition = lambda _: True
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters() if condition(p)).values())
    
class CompoundModel(torch.nn.Module):
    
    @staticmethod
    def fill_kwargs(config_dict : ConfigDict):
        for layer_dict in config_dict.elements_of('submodels'):
            utils.fill_dict(layer_dict)
            if 'weight initialisation' in layer_dict:
                init_name, init_dict = layer_dict['weight initialisation'].item()
                init_default_dict = weight_init.inits_dict[init_name]['arguments']
                init_dict.fill_with_defaults(init_default_dict)
    
    def __init__(self, submodels : Iterable[ConfigDict], *args, **kwargs):
        """Creates a torch.nn.Sequential instansce from a list of layers and their hyperparameters."""
        super().__init__()

        if submodels is None:
            submodels = []
        if not isinstance(submodels, (tuple, list)):
                submodels = [submodels]
                
        self.layers = torch.nn.ModuleList()
        self.pass_all_inputs : List[bool] = []
        for layer_dict in submodels:
            layer = utils.create_object_from_dict(layer_dict, wrapper_class = Model)
            if 'weight initialisation' in layer_dict:
                init_func = utils.config_dict.initialise_object_from_dict(
                                                config_dict = layer_dict['weight initialisation'],
                                                classes_dict = weight_init.inits_dict
                                                )
                init_func(layer)
            self.layers.append(layer)
            self.pass_all_inputs.append(getattr(layer, 'PASS_ALL_INPUTS', False))

        self.PASS_ALL_INPUTS = any(self.pass_all_inputs)
            
    
class ModifiedModel(Model):
    """Base class for models that have a base (eg. a torchvision pretrained model) that is then slighlty modified (eg. its final layer is changed)."""

    PARAMS = {
        'base':  'torchvision.models.video.r3d_18',
        'modify': {
            'stem.0': {
                'torch.nn.Conv3d': {
                    'in_channels': 11,
                    'out_channels': 64,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 'same',
                    'bias': False
                    }
                },
            'fc': {
                'torch.nn.Linear': {
                    'in_features': 512,
                    'out_features': 8
                    }
                }
            }
    }
    
    @staticmethod
    def fill_kwargs(config_dict : utils.ConfigDict):
        """Fills the config dict with the hyperparameters of the model and all its layers."""
        utils.fill_dict(config_dict['base'])
        if 'modify' not in config_dict:
            return
        for layer_dict in config_dict['modify'].values():
            if layer_dict is None:
                continue
            utils.fill_dict(layer_dict)
            if 'weight initialisation' in layer_dict:
                init_name, init_dict = layer_dict['weight initialisation'].item()
                init_default_dict = weight_init.inits_dict[init_name]['arguments']
                init_dict.fill_with_defaults(init_default_dict)

    
    def create_architecture(self,
                            base : utils.ConfigDict,
                            modify : Optional[Union[utils.ConfigDict, Literal[False]]] = None,
                            *args, **kwargs):
        """
        Helper function that creates the model architecture but doesn't load pretrained weights.
        Arguments:
            `base_model_dict`: a ConfigDict specifying the hyperparameters of the base model
            `modify`: ConfigDict or False; if a ConfigDict, its keys should be layer names in the base model that should be modified, and their corresponding values the modified layers. If the value is a list or tuple of layers, then they will be gathered in an nn.Sequential module.
        """
        model : Model = utils.create_object_from_dict(base, wrapper_class = Model)
        model.init_weight(base.value())
        if not modify:
            return model
        for submodel_to_modify, replacement_layers in modify.items():
            replacement_model = FeedForwardModel(replacement_layers)
                
            submodel = getattr(model, 'model', model)
            attr_names = submodel_to_modify.split('.')
            for attr_name in attr_names[:-1]:
                if isinstance(attr_name, str):
                    submodel = getattr(submodel, attr_name)
                else:
                    submodel = submodel[attr_name]
            setattr(submodel, attr_names[-1], replacement_model)
        return model
    
    def __init__(self, config_dict : utils.ConfigDict, *args, **kwargs):
        """`config_dict` should have a 'base' key that specifies the base model. If it has a 'modify' key (and it's not `False` or `None`) then it should be a dict with keys that are layer names in the original model, and their corresponding values are layers or a list/tuple of layers that the base layers should be replaced to."""
        super().__init__(self.create_architecture, config_dict, *args, **kwargs)

class FeedForwardModel(CompoundModel):
    
    @staticmethod
    def fill_kwargs(config_dict : ConfigDict):
        for layer_dict in config_dict.elements_of('layers'):
            utils.fill_dict(layer_dict)
            if 'weight initialisation' in layer_dict:
                init_name, init_dict = layer_dict['weight initialisation'].item()
                init_default_dict = weight_init.inits_dict[init_name]['arguments']
                init_dict.fill_with_defaults(init_default_dict)
    
    def __init__(self, layers : Iterable[ConfigDict], *args, **kwargs):
        """Creates a torch.nn.Sequential instansce from a list of layers and their hyperparameters."""
        super().__init__(layers, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        for pass_all, layer in zip(self.pass_all_inputs, self.layers):
            layer_args = args if pass_all else ()
            layer_kwargs = kwargs if pass_all else {}
            x = layer(x, *layer_args, **layer_kwargs)
        return x

class ParallelModel(CompoundModel):
    
    @staticmethod
    def fill_kwargs(config_dict : ConfigDict):
        for layer_dict in config_dict.elements_of('threads'):
            utils.fill_dict(layer_dict)
            if 'weight initialisation' in layer_dict:
                init_name, init_dict = layer_dict['weight initialisation'].item()
                init_default_dict = weight_init.inits_dict[init_name]['arguments']
                init_dict.fill_with_defaults(init_default_dict)
    
    def __init__(self, threads, *args, **kwargs):
        super().__init__(threads, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        outputs = []
        for pass_all, layer in zip(self.pass_all_inputs, self.layers):
            layer_args = args if pass_all else ()
            layer_kwargs = kwargs if pass_all else {}
            outputs.append(layer(x, *layer_args, **layer_kwargs))
        return outputs