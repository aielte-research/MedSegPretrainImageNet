import math
import signal
import string
from typing import Callable, Optional, Tuple, Union, Any

import importlib
import inspect
import warnings

from matplotlib import pyplot as plt

from .config_dict import ConfigDict
from exception_handling import handle_exception

def timeout_handler(*args, **kwargs):
    raise TimeoutError

signal.signal(signal.SIGALRM, timeout_handler)

def timeout(seconds : int, final : Optional[Callable] = None):
    """Decorator that times out the function or method after `seconds` seconds. If a `final` callable is given, it will be executed regardless of whether the function execution times out or not."""
    def decorate(f):
        def timedout_f(*args, **kwargs):
            signal.alarm(seconds)
            try:
                result = f(*args, **kwargs)
                signal.alarm(0)
                return result
            except TimeoutError:
                warnings.warn(f'Execution of function {f.__name__} has timed out, continuing.')
            finally:
                if final is not None:
                    final()
        return timedout_f
    return decorate

def get_class_constr(class_path : str):
    """
    Get the class constructor from a string specifying its absoulte path.
    
    Eg. from the str 'torch.nn.Module' it should return the torch.nn.Module class constructor.
    """
    if isinstance(class_path, ConfigDict):
        class_path = class_path.key()
        
    path_list = class_path.split('.')
    module_path, class_name = '.'.join(path_list[:-1]), path_list[-1] # separate module name and the final class name
    class_constr = getattr(importlib.import_module(module_path), class_name)
    return class_constr

def get_class_constr_and_dict(config_dict : ConfigDict, key : Optional[str] = None) -> Tuple[Any, ConfigDict]:
    """From a ConfigDict structure {class_constr_path: class_dict} return the constructor object of the class and the class dict."""
    value = config_dict[key]
    if isinstance(value, str):
        class_path, class_dict = value, ConfigDict()
    else:
        class_path, class_dict = config_dict[key].item()
    class_constr = get_class_constr(class_path)

    return class_constr, class_dict


def fill_dict(config_dict : ConfigDict, key : Union[str, None] = None, fill_with_init_params : bool = True,
              class_path : Union[str, None] = None):
    """
    Fills in a ConfigDict with the default hyperparameters of a class if they are missing.
    
    Arguments:
        `config_dict`: ConfigDict; either the dict with the hyperparameters of the class (class dict), or a dict with one key-value pair, where the key is the path to the class constructor, and the value is the class dict
        `key`: optional str; if `config_dict` is not of the form specified above, `config_dict[key]` should be
        `fill_with_init_params`: bool; if True the class does not have a `PARAMS` attribute, the config dict will be filled with the default arguments of the `__init__` method of the class constructor; default: True
        `class_path`: optional str; if `config_dict` does not contain a class path as its key, it should be specified in this variable
    """
    if not isinstance(config_dict, ConfigDict):
        return

    if class_path is None:
        class_constr, class_dict = get_class_constr_and_dict(config_dict, key)
    else:
        class_constr, class_dict = get_class_constr(class_path), config_dict
    if hasattr(class_constr, 'PARAMS'):
        defaults : dict = class_constr.PARAMS
    elif fill_with_init_params:
        # if the class has no `PARAMS` attribute, the default values of its `__init__` method will be considered
        if not isinstance(class_constr, type):
            init_func = class_constr
        else:
            init_func = getattr(class_constr, '__init__', class_constr)
        if getattr(init_func, '__defaults__', False):
            defaults = {k: v.default for k, v in inspect.signature(class_constr).parameters.items()
                        if v.default not in (inspect.Parameter.empty, None)}
        else:
            defaults = {}
    else:
        defaults = {}

    class_dict.fill_with_defaults(defaults)

    if hasattr(class_constr, 'fill_kwargs'):
        class_constr.fill_kwargs(class_dict)

    return config_dict

def expects_kwarg(callable, kwarg):
    sign_params = inspect.signature(callable).parameters
    return kwarg in sign_params.keys()

def accepts_kwarg(callable, kwarg):
    sign_params = inspect.signature(callable).parameters
    return kwarg in sign_params.keys() or any(param.kind == inspect.Parameter.VAR_KEYWORD
                                              for param in sign_params.values())

def create_object_from_dict(config_dict : ConfigDict, key : Union[str, None] = None,
                            class_path = None, wrapper_class = None, convert_to_kwargs = False,
                            filter_kwargs = False, *args, **kwargs):
    """
    Creates an object from a ConfigDict containing a class constructor and hyperparameters.
    Arguments:
        `config_dict`: ConfigDict; either the dict with the hyperparameters of the class (class dict), or a dict with one key-value pair, where the key is the path to the class constructor, and the value is the class dict
        `key`: optional str; if `config_dict` is not of the form specified above, `config_dict[key]` should be
        `class_path`: optional str; if `config_dict` does not contain a class path as its key, it should be specified in this variable
        `wrapper_class`: if given, and the class constructor is not a subclass of the wrapper class, it will be wrapped in the wrapper class
        `args`, `kwargs`: any number of positional and keyword arguments that will be passed onto the class constructor
    """
    if not isinstance(config_dict, ConfigDict):
        config_dict = ConfigDict({config_dict: {}})
    if class_path is None:
        class_constr, class_dict = get_class_constr_and_dict(config_dict, key)
    else:
        class_constr, class_dict = get_class_constr(class_path), config_dict
    
    if filter_kwargs:
        kwargs = {kw: value for kw, value in kwargs.items() if accepts_kwarg(class_constr, kw)}
    
    if wrapper_class is not None and wrapper_class not in getattr(class_constr, '__mro__', []):
        return wrapper_class(class_constr, class_dict, *args, **kwargs)

    if not convert_to_kwargs:
        return class_constr(class_dict, *args, **kwargs)

    else:
        class_kwargs = get_kwargs(class_constr, class_dict)
        return class_constr(*args, **kwargs, **class_kwargs)

def get_kwargs(class_constr, config_dict: Union[ConfigDict, dict] = {}):
    """
    Returns a dictionary of keyword arguments that can be passed onto the class constructor.
    Arguments:
        `class_constr`: callable that returns an instance of a class
        `config_dict`: dict-like; its keywords should be hyperparameters. If a keyword in `config_dict` is not under the same name as it should be passed onto `class_constr`, its corresponding value should be a dict-like object that has an 'argument name' key
    """
    if hasattr(class_constr, 'PARAMS'):
        kwargs = {}
        for arg_name, arg in class_constr.PARAMS.items():
            if isinstance(arg, dict):
                kwargs[arg.get('argument name', arg_name)] = config_dict[arg_name]
            else:
                kwargs[arg_name] = config_dict[arg_name]
    elif isinstance(config_dict, dict):
        kwargs = config_dict
    else:
        kwargs = config_dict.trim().config_dict
    
    return kwargs

def get_extensions(extensions):
    return [extension.lower().strip('.') for extension in extensions]

@timeout(300, final = lambda: plt.close('all'))
def export_plot(plotter, extensions, *args, **kwargs):
    export_funcs = {
        'json': plotter.export_json,
        'png': lambda *args, **kwargs: plotter.export_matplotlib(extension = 'png', *args, **kwargs),
        'svg': lambda *args, **kwargs: plotter.export_matplotlib(extension = 'svg', *args, **kwargs),
        'html': plotter.export_bokeh
    }
    
    for extension in get_extensions(extensions):
        try:
            export_funcs[extension](*args, **kwargs)
        except Exception as e:
            handle_exception(e, f'An exception occured while trying to export plot as a .{extension} file.')

alphabet = ''.join(str(i) for i in range(10)) + string.ascii_letters + '!?'

def base64_str(num, num_digits = None):
    output = ''
    output_len = num and math.floor(math.log(num, 64))
    curr_magn = 64 ** output_len
    curr_num = int(num)
    while curr_magn >= 1:
        output += alphabet[curr_num // curr_magn]
        curr_num = curr_num % curr_magn
        curr_magn = curr_magn // 64
    if num_digits is not None:
        if num_digits <= output_len:
            raise ValueError(f'{output} (decimal {num}) has more than {num_digits} digits.')
        output = '0' * (num_digits - output_len - 1) + output
    return output