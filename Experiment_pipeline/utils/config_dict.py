from typing import Optional, Tuple, Union, Any

import yaml
from copy import deepcopy
from .default_dict import default_dict

class ConfigDict(object):
    """
    Wrapper class for dict of dicts objects describing the hyperparameters of a training scheme, that can have default values not specified in the dict.
    All keys should be str. The class can deal with nested dictionaries of any depth. D['k1']['k2']...['kn'] can be accessed and modified as D['k1/k2/.../kn'].
    """

    SLASH_SUBSTITUTE = '___SLASH___'
    PROTECTED = ['meta/technical/name_fields'] # TODO

    @staticmethod
    def from_yaml(path):
        """Builds a ConfigDict object from a .yaml file."""
        with open(path, 'r') as config_file:
            config_dict = yaml.full_load(config_file)
        return ConfigDict(config_dict)

    def __init__(self, config_dict : dict = {}):
        """Converts a dict of dicts to a ConfigDict object."""
        def convert(value):
            if isinstance(value, dict):
                if all(isinstance(key, str) for key in value.keys()):
                    return ConfigDict(value)
                else:
                    return value
            
            if isinstance(value, (list, tuple)):
                return type(value)(map(convert, value))

            if isinstance(value, str):
                return value.replace('/', self.SLASH_SUBSTITUTE)
            
            return value

        self.config_dict = {key: convert(value) for key, value in config_dict.items()}
    
    def __getitem__(self, key_seq : str) -> Any:
        """
        Returns an element of the config dict. You can access a value inside a subdictionary within the main dict, or at any level of depth recursively, by specifying a path delimited by /'s. So D['k1/k2/.../kn'] is equivalent to D['k1']['k2']...['kn'].
        D[None] return D.
        """
        if key_seq is None:
            return self

        keys_list = key_seq.split('/')
        key = keys_list[0]

        value = self.config_dict.get(key.replace('_', ' '),
                                     self.config_dict.get(key.replace(' ', '_'),
                                                          self.config_dict.get(key)))
        
        if len(keys_list) == 1:
            if isinstance(value, str):
                value = value.replace(self.SLASH_SUBSTITUTE, '/')
            return value

        if isinstance(value, (tuple, list)) and len(value) == 1:
            value_ = value[0]
            if isinstance(value_, ConfigDict):
                value = value_
        
        if not isinstance(value, (tuple, list)):
            value = value['/'.join(keys_list[1:])]
        else:
            key = keys_list[1]
            for possible_dict in value:
                if isinstance(possible_dict, ConfigDict) and possible_dict.key() == key:
                    if len(keys_list) == 2:
                        return possible_dict.value()
                    return possible_dict.value()['/'.join(keys_list[2:])]
        if isinstance(value, str):
            value = value.replace(self.SLASH_SUBSTITUTE, '/')
        return value

    def get_str(self, key_seq : Optional[str] = None) -> str:
        value = self[key_seq]
        if isinstance(value, ConfigDict):
            value = value.key()
        return value.replace(self.SLASH_SUBSTITUTE, '/')

    def get_tuple(self, key_seq : str, default = []) -> tuple:
        value = self.get(key_seq, default = default)
        if not isinstance(value, (list, tuple)):
            value = [value]
        return tuple(value)
    
    def get_str_tuple(self, key_seq : str, default = []) -> Tuple[str]:
        values = self.get_tuple(key_seq, default = default)
        out_list = []
        for value in values:
            if isinstance(value, ConfigDict):
                value = value.key()
            out_list.append(value)
        return tuple(out_list)
    
    def __setitem__(self, key_seq : str, value : Any):
        """
        Sets the value of an element of the config dict. You can access elements inside a subdictionary within the main dict, or at any level of depth recursively, by specifying a path delimited by /'s. So if D['k1'] == {'k2': {}}, then setting D['k1/k2/k3'] = 'k4' is equivalent to D['k1']['k2']['k3'] = 'k4'.
        D['k1/k2/.../kn'] = 'k' creates subdictionaries 'k3', 'k3/k4', 'k3/k4/k5', etc. if they do not exist already, and adds a 'kn' key to the last subdict with value 'k'.
        """
        keys_list = key_seq.split('/')
        key = keys_list[0]

        if len(keys_list) == 1:
            if isinstance(value, dict):
                if all(isinstance(key, str) for key in value.keys()):
                    value = ConfigDict(value)
            self.config_dict[key] = value
            return
        
        if not key in self.config_dict:
            key = key.replace(' ', '_')
        if not key in self.config_dict:
            key = key.replace('_', ' ')
        if not key in self.config_dict:
            self.config_dict[key] = ConfigDict({})
        
        if isinstance(value, str):
            value = value.replace('/', self.SLASH_SUBSTITUTE)

        self.config_dict[key]['/'.join(keys_list[1:])] = value
    
    def __str__(self):
        return f'ConfigDict({self.to_dict()})'
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if not isinstance(other, ConfigDict):
            return False
        return self.config_dict == other.config_dict

    def to_dict(self, lists_to_tuples = True, space_to_underscore = True):
        """Converts the ConfigDict object to a python dictionary."""
        def convert(value):
            if isinstance(value, ConfigDict):
                return value.to_dict()
            
            if isinstance(value, (list, tuple)):
                casting_type = tuple if lists_to_tuples else type(value)
                return casting_type(map(convert, value))
            
            if isinstance(value, str):
                return value.replace(self.SLASH_SUBSTITUTE, '/')

            return value

        if space_to_underscore:
            def convert_key(key):
                return key.replace(self.SLASH_SUBSTITUTE, '/').replace(' ', '_')
        else:
            def convert_key(key):
                return key.replace(self.SLASH_SUBSTITUTE, '/')
        return {convert_key(key): convert(value) for key, value in self.items()}
    
    #def __dict__(self):
    #    return self.to_dict()
        
    def has(self, key_seq : str):
        """Returns True if the dictionary has the specified key sequence."""
        keys_list = key_seq.split('/')
        key = keys_list[0]

        if key not in self.config_dict and key.replace(' ', '_') not in self.config_dict and key.replace('_', ' ') not in self.config_dict:
            return False

        if len(keys_list) == 1:
            return True

        value = self[key]
        if isinstance(value, (tuple, list)) and len(value) == 1:
            value_ = value[0]
            if isinstance(value_, ConfigDict):
                value = value_
        if isinstance(value, (tuple, list)):
            key = keys_list[1]
            for possible_dict in value:
                if isinstance(possible_dict, ConfigDict) and possible_dict.key() == key:
                    if len(keys_list) == 2:
                        return True
                    return possible_dict.value().has('/'.join(keys_list[2:]))
            return False

        if isinstance(value, ConfigDict):
            return value.has('/'.join(keys_list[1:]))
        
        return False
    
    def __contains__(self, key_seq : str):
        return self.has(key_seq)
    
    def has_key(self, key_seq : str):
        return self.has(key_seq)
    
    def get(self, key_seq : str, default = None):
        """Analogous to the `dict.get()` method."""
        if self.has(key_seq):
            return self[key_seq]
        return default

    def get_or_update(self, key_seq : str, default : Union[dict, Any] = default_dict,
                      final = True, keep_key_seq = False):
        """
        Returns the value of a specified key, or a default value that it looks up from a dictionary. If the key is not present in the original dict, it is added with the default value as its value. 
        Arguments:
            `key_seq`: str; sequence of keys whose value should be looked up
            `default`: dict of dicts or any; default value to return; if dict, it should have a `default[key]['default']` value, which will be set if the dictionary does not have a value for the key (`key` being the actual key looked up, so if `key_seq = 'k1/k2/.../kn'`, then `key == kn`). If not a dict, then the value `default` will be returned.
            `final`: bool, default: True; if False, insted of `D[key_seq] = value`, `D[key_seq] = {value: {}}` will be set, so the dictionary can be continued to be built up.
        """
        if self.has(key_seq):
            value = self[key_seq]
        else:
            if keep_key_seq:
                key = key_seq
            else:
                keys_list = key_seq.split('/')
                key = keys_list[-1]
            if isinstance (default, dict):
                default_value_or_dict = default[key]
                if isinstance(default_value_or_dict, dict):
                    value = default_value_or_dict.get('default', default_value_or_dict)
                else:
                    value = default_value_or_dict
            else:
                value = default
        if not final and isinstance(value, str):
            value = {value.replace('/', self.SLASH_SUBSTITUTE): {}}
        if isinstance(value, dict):
            if all(isinstance(key, str) for key in value.keys()):
                value = ConfigDict(value)
        self[key_seq] = value
        return value
    
    def fill_with_defaults(self, default_dict : dict = default_dict, final = False, keep_key_seq = True):
        for key in default_dict.keys():
            self.get_or_update(key, default_dict, final, keep_key_seq)
        return self
    
    def update(self, new_dict : dict):
        """Updates the values of the ConfigDict from a dictionary."""
        for key, value in new_dict.items():
            if not isinstance(value, dict):
                self[key] = value
            elif not self.has(key):
                self[key] = ConfigDict(value)
            else:
                self[key].update(value)
    
    def mask(self, *key_seqs):
        """Returns a copy of the dictionary with the specified key sequences deleted."""
        copied_dict = deepcopy(self)
        for key_seq in key_seqs:
            if key_seq in self:
                copied_dict.pop(key_seq)
        return copied_dict

    def to_kwargs(self, default_dict : dict = default_dict, key_seq : Union[str, None] = None):
        """
        Creates a dict of keyword arguments that can be passed on to the constructor of an object.
        Arguments:
            `key_seq`: str or None; sequence of keys pointing to a dictionary; if None, then the original dictionary should only have one key
            `default_dict`: dict; dictionary of hyperparameters that the object constructor takes; every key should be a hyperparameter, and its corresponding entry a dict with 'argument name' and 'default' entries
        """
        if key_seq is None:
            keys = list(self.keys())
            if len(keys) == 1:
                key_seq = keys[0]
            else:
                raise ValueError('If `key_seq` is None, then the dictionary should only contain one entry.')
        keys_list = key_seq.split('/')
        sup_seq = '/'.join(keys_list[:-1])
        key = keys_list[-1]
        sup_dict = self if sup_seq == '' else self[sup_seq]
        if not isinstance(sup_dict, ConfigDict):
            sup_dict = ConfigDict({key: {}})
        curr_dict = self[key_seq]
        return {subdict.get('argument name', key): curr_dict.get_or_update(key, default_dict)
                for key, subdict in default_dict.items()}
    
    def elements_of(self, key_seq : str):
        """Returns a generator that iterates over the elements of `self[key_seq]`."""
        if not self.has(key_seq):
            return ()
        value = self[key_seq]
        if not isinstance(value, (list, tuple)):
            self[key_seq] = [value]
        else:
            self[key_seq] = list(value)
        for i, x in enumerate(self[key_seq]):
            if not isinstance(x, ConfigDict):
                x = ConfigDict({x: {}})
                self[key_seq][i] = x
            yield x
               
    def trim(self):
        """
        Creates a ConfigDict from the original where `key: {}` pairs are reduced to `key`, and one-long lists and tuples are reduced to their only element.
        
        Example: `ConfigDict({k1: {v1: {}}, k2: [v2]}) -> ConfigDict({k1: v1, k2: v2})`
        """
        def convert(value):
            if isinstance(value, ConfigDict):
                keys = list(value.keys())
                if len(keys) == 1 and isinstance(value[value.key()], ConfigDict) and value[value.key()].config_dict == {}:
                    return value.key()
                return value.trim()
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return convert(value[0])
                return type(value)(map(convert, value))
            return value

        def is_protected(key): # TODO
            for protected_key in self.PROTECTED:
                keys_list = protected_key.split('/')
                for i in range(len(keys_list)):
                    if key == '/'.join(keys_list[i:]):
                        return True
            return False
        
        return ConfigDict({key: convert(value) if not is_protected(key) else value for key, value in self.items()})
    
    def expand(self):
        for key, value in self.items():
            if isinstance(value, str) and '/' not in value:
                self[key] = ConfigDict({value: {}})
            elif isinstance(value, ConfigDict):
                value.expand()

    def key(self) -> str:
        """If the ConfigDict has only one key, it is returned."""
        keys = list(self.keys())
        if len(keys) == 1:
            return keys[0]
        
        raise ValueError(f'Method `key` requires the dict to have only one key, but {self} has several.')

    def value(self) -> 'ConfigDict':
        """If the ConfigDict has only one key, its corresponding value is returned."""
        values = list(self.values())
        if len(values) == 1:
            return values[0]
        
        raise ValueError(f'Method `value` requires the dict to have only one key, but {self} has several.')
    
    def item(self):
        """If the ConfigDict has only one key, it is returned with its corresponding value."""
        try:
            return self.key(), self.value()
        except ValueError:
            raise ValueError(
                f'Method `item` requires the dict to have only one key, but {self} has several.'
                )

    def items(self):
        """An analogous function to dict.items()"""
        return self.config_dict.items()
    
    def keys(self):
        """An analogous function to dict.keys()"""
        return self.config_dict.keys()
    
    def values(self):
        """An analogous function to dict.values()"""
        return self.config_dict.values()
    
    def depth(self):
        """Returns the length of the longest possible key sequence in the dictionary."""
        def depth(value):
            if isinstance(value, ConfigDict):
                return value.depth()
            return 0
        if len(list(self.values())) == 0:
            return 0
        return 1 + max(map(depth, self.values()))
    
    def pop(self, key_seq : str, *args):
        args_len = len(args)
        if args_len > 1:
            raise TypeError('`ConfigDict.pop()` takes only one default value.')
        if args_len == 1:
            default = args[0]
            has_default = True
        else:
            has_default = False

        keys_list = key_seq.split('/')
        key = keys_list[0]
        if len(keys_list) == 1:
            had = False
            for variant in (key, key.replace(' ', '_'), key.replace('_', ' ')):
                if variant in self.config_dict:
                    output = self.config_dict.pop(variant, *args)
                    had = True
            if had:
                return output
        if key not in self:
            if has_default:
                return default
            raise KeyError(f'Key sequence \'{key_seq}\' is not in ConfigDict.')
        return self[key].pop('/'.join(keys_list[1:]), *args)
    
    def popitem(self):
        return self.config_dict.popitem()
    
    def clear(self):
        self.config_dict.clear()
    
    def copy(self):
        return deepcopy(self)
    
    def __iter__(self):
        return iter(self.config_dict)
    
    def __len__(self):
        return len(self.config_dict)

    
def initialise_object_from_dict(config_dict : ConfigDict, classes_dict : dict,
                                class_name : Union[str, None] = None, key_seq : Union[str, None] = None, **kwargs):
    """
    Takes a ConfigDict and a dictionary, and initialises an instance of a class specified in that dictionary, with the hyperparameters found in the ConfigDict.
    Arguments
        `config_dict`: ConfigDict; dictionary of hyperparameters; every parameter not specified in this dict will be automatically set to its default value
        `classes_dict`: dict[str, dict[str, callable | dict[str, any]]]; dictionary where entries correspond to classes, and their corresponding keys are dictionaries, with an 'init' and an 'arguments' key. The 'init' value should be a callable that takes key word arguments specified in 'arguments'; 'arguments' should be a dict of dicts, with arguments as keys, and every dictionary value having a 'default' and 'argument name' key
        `class_name`: str or None, default: None; optional name of the class to look for; if None, `config_dict` is expected to have only one key, and the inferred class name will be `config_dict.key()`
        `key_seq`: str or None, default: None; if not None, instead of `config_dict`, `config_dict[key_seq]` will be parsed
        `kwargs`: optional key word arguments that will be directly passed onto the class constructor 
    """
    if isinstance(config_dict, str):
        config_dict = ConfigDict({config_dict: {}})
    if class_name is None:
        class_name = config_dict.key()
    class_dict : ConfigDict = classes_dict[class_name]
    key_seq = key_seq + '/' + class_name if key_seq is not None else class_name
    arguments = config_dict.to_kwargs(default_dict = class_dict['arguments'], key_seq = key_seq)
    return class_dict['init'](**arguments, **kwargs)