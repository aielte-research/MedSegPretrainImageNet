from typing import Callable, Generator, Iterable, List, Literal, Optional, Set, Union, Tuple, Dict, Any
import numpy as np

from torch.utils.data import DataLoader
from .utils import BalancedDataset, DataIterator
import utils

class Dataset(object):
    """Wrapper class for callables that return (train_data, val_data) tuples."""
    def __init__(self, ds_constr : Callable[..., Tuple[Dict[str, Iterable], Dict[str, Iterable]]],
                 ds_dict : utils.ConfigDict, *args, **kwargs):
        ds_kwargs = utils.get_kwargs(ds_constr, ds_dict)
        self.train, self.val = ds_constr(*args, **kwargs, **ds_kwargs)

class MixedDataset(Dataset):
    
    """
    Class for a union of several datasets. It will produce samples by alternating between the datasets, with a give frequency.
    
    Parameters:
        `datasets`: Tuple[ConfigDict]; each entry specifies a subdataset
        `switch_frequency`: int; every consequent `switch_frequency` sample will be from the same dataset
        `preserve_order`: bool; if set to True, then the datasets will always change according to the cyclic order they were specified in in `datasets`; if set to True, the order will be random
        `balancing_strategy`: Dict[Literal['train', 'val'], Literal['none', 'under', 'over']]; specifies how to deal with differing dataset sizes in the train and validation set separately; possible values:
            `'none'`: pool together all datasets as are
            `'under'`: in each epoch, only sample from each dataset a number of records equal to the size of the smallest dataset
            `'over'`: in each epoch, oversample all datasets so that their size equals the largest one's
        `preload_data`: bool; if set to False, then the Dataset objects will be reinitialised each time the sampling switches to a different dataset
        
    Notes:
        * If `preserve_order` is False, then it is not guaranteed that the same dataset can't appear twice in a row. Let's say we have three datasets, 0, 1, and 2, and we have `switch_frequency` set to 2. Then with `preserve_order == True`, the datasets will follow as 0, 0, 1, 1, 2, 2, 0, 0...; if it is False, however, then an order such as 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1... is possible.
        * `preload_data` only sets whether the Dataset object, as it would be loaded if that was the only dataset you use, is stored when it's not the active dataset. If you only use that dataset to store information that will then be fed to a load_function, then that function will be called at each iteration. It is recommended to only turn this parameter off if the sheer number of datapoints you will be working with is too much.
        * Even if `balancing_strategy == 'none'`, a few records might be dropped in each epoch, as the number of records in an epoch will always be divisible by `switch_frequency`.
        * If `preserve_order == True` and `balancing_strategy == 'none'`, then, if for example dataset 0 has fewer records than dataset 1, then the end of the epoch will look like ...0, 0, 1, 1, 2, 2, 1, 1, 2, 2... If you want to make sure that 1 always follows 0, switch to a different balancing strategy.
        * The dataset's dictionaries will contain all keywords that appear in at least one subdataset. If a keyword does not appear in a dataset, its value will be set to NaN for all entries.
    
    Apart from the keywords present in the datasets, there are three additional keywords introduced. `train_counter` and `val_counter` are booleans indicating whether the current record is a train or a validation sample; `ds_idx` is an int that specifies the index of the current subdataset.
    """
    
    PARAMS = dict(datasets = tuple(), # list of dicts specifying datasets
                  switch_frequency = 1, # how many images you should load before switching to the next dataset
                  preserve_order = False, # whether to always load the next dataset per their defined order
                  balancing_strategy = {'train': 'none', 'val': 'under'}, # one of 'none', 'under', or 'over'
                  preload_data = True # whether to load all of the data in advance
                  )
    
    @staticmethod
    def fill_kwargs(config_dict):
        for ds_dict in config_dict.elements_of('datasets'):
            utils.fill_dict(ds_dict)
        config_dict['balancing_strategy'].fill_with_defaults(MixedDataset.PARAMS['balancing_strategy'])
    
    def __init__(self, config_dict : utils.ConfigDict, seed : Optional[int] = None, *args, **kwargs):
        
        self.dataset_configs : Tuple[utils.ConfigDict] = config_dict.get_tuple('datasets')
        
        self.rng = np.random.default_rng(seed)
        if seed is None:
            seed = self.rng.integers(0, 1e12)
        
        self.seed : int = seed
        self.args, self.kwargs = args, kwargs
        self.preload_data : bool = config_dict['preload_data']
        
        keys : Set[str] = set()
        
        self.train_ds_sizes : List[int] = []
        self.val_ds_sizes : List[int] = []
        
        self.dss : List[Dataset] = []
        self.load_functions : List[Dict[str, Callable]] = []
        
        def default_load_function(data_to_load, load_function):
            if isinstance(data_to_load, str):
                return load_function(data_to_load)
            else:
                return data_to_load
        
        for ds_idx, ds_dict in enumerate(self.dataset_configs):
            ds = utils.create_object_from_dict(ds_dict, wrapper_class = Dataset, seed = seed, *args, **kwargs)
            train_ds, val_ds = ds.train, getattr(ds, 'val', {})
            setattr(ds, 'val', val_ds)
            keys.update(train_ds.keys())
            
            train_len = len(next(iter(train_ds.values())))
            val_len = len(next(iter(val_ds.values()))) if len(val_ds) > 0 else 0
            
            if not self.preload_data:
                setattr(ds, 'train', {kw: np.arange(train_len) for kw in train_ds.keys()})
                setattr(ds, 'val', {kw: np.arange(val_len) for kw in val_ds.keys()})
            
            ds.train['ds_idx'] = [ds_idx] * train_len
            ds.val['ds_idx'] = [ds_idx] * val_len
            
            self.dss.append(ds)
            load_function = getattr(ds, 'load_function', lambda _: _)
            if not isinstance(load_function, (dict, utils.ConfigDict)):
                load_function = {kw: lambda x: default_load_function(x, load_function) for kw in train_ds.keys()}
            self.load_functions.append(load_function)
            
            self.train_ds_sizes.append(train_len)
            self.val_ds_sizes.append(val_len)
        
        for key in keys:
            for load_function, ds, train_len, val_len in zip(self.load_functions, self.dss, self.train_ds_sizes, self.val_ds_sizes):
                if key not in load_function:
                    load_function[key] = lambda _: _
                for kw, length in zip(('train', 'val'), (train_len, val_len)):
                    data_dict : Dict[str, Iterable] = getattr(ds, kw)
                    if key not in data_dict:
                        data_dict[key] = [np.nan] * length
        
        self.switch_frequency : int = config_dict['switch_frequency']
        self.preserve_order : bool = config_dict['preserve_order']
        for kw in ('train', 'val'):
            setattr(self, f'{kw}_balancing_strategy', config_dict.get_str(f'balancing_strategy/{kw}'))
        
        self.num_datasets = len(self.dataset_configs)
        
        for ds_kw, balancing_strategy, ds_sizes in zip(('train', 'val'),
                                                       (self.train_balancing_strategy, self.val_balancing_strategy),
                                                       (self.train_ds_sizes, self.val_ds_sizes)):
            
            if balancing_strategy == 'none':
                full_len = sum(ds_size - (ds_size % self.switch_frequency) for ds_size in ds_sizes)
            elif balancing_strategy == 'over':
                max_size = max(ds_sizes)
                full_len = (max_size - (max_size % self.switch_frequency)) * self.num_datasets
            elif balancing_strategy == 'under':
                min_size = min(ds_sizes)
                full_len = (min_size - (min_size % self.switch_frequency)) * self.num_datasets
            
            setattr(self, f'full_{ds_kw}_len', full_len)
        
        keys.add('ds_idx')
        
        self.train = {'train_counter': [True] * self.full_train_len,
                      'val_counter': [False] * self.full_train_len,
                      **{kw: [True] * self.full_train_len for kw in keys}}
        
        self.val = {'train_counter': [False] * self.full_val_len,
                    'val_counter': [True] * self.full_val_len,
                    **{kw: [False] * self.full_val_len for kw in keys}}
        
        self.load_function = {'train_counter': self.count_train,
                              'val_counter': self.count_val}
        
        def make_load_fn(keyword):
            def load_fn(train):
                return self.load(keyword, train)
            return load_fn
        
        self.load_function = {**{kw: make_load_fn(kw) for kw in keys},
                              'train_counter': self.count_train,
                              'val_counter': self.count_val,
                              'ds_idx': self.load_ds_idx}
        
        self.reinit_train_ds_idcs() # initialises `self.train_ds_idcs`
        self.reinit_val_ds_idcs()   # initialises `self.val_ds_idcs
        
        self.curr_train_ds, self.curr_val_ds = 0, 0
        
        # initialise train indices
        self.train_record_idcs : List[Generator[int]] = [iter(self.rng.permutation(ds_size)) for ds_size in self.train_ds_sizes]
        
        # initialise validation indices
        self.val_record_idcs : List[Generator[int]] = [iter(range(ds_size)) for ds_size in self.val_ds_sizes]
    
    def load_next_train_record_idx(self, ds_idx : int):
        try:
            next_idx = next(self.train_record_idcs[ds_idx])
        except StopIteration:
            self.train_record_idcs[ds_idx] = iter(self.rng.permutation(self.train_ds_sizes[ds_idx]))
            next_idx = next(self.train_record_idcs[ds_idx])
        self.curr_train_record = next_idx
    
    def load_next_val_record_idx(self, ds_idx : int):
        try:
            next_idx = next(self.val_record_idcs[ds_idx])
        except StopIteration:
            self.val_record_idcs[ds_idx] = iter(range(self.val_ds_sizes[ds_idx]))
            next_idx = next(self.val_record_idcs[ds_idx])
        self.curr_val_record = next_idx
    
    def count(self, ds_type : Literal['train', 'val'], counter : bool):
        if not counter:
            return False
        try:
            new_idx = next(getattr(self, f'{ds_type}_ds_idcs'))
        except StopIteration:
            getattr(self, f'reinit_{ds_type}_ds_idcs')()
            new_idx = next(getattr(self, f'{ds_type}_ds_idcs'))
        if new_idx != getattr(self, f'curr_{ds_type}_ds'):
            if not self.preload_data:
                getattr(self, f'load_{ds_type}_ds')(getattr(self, f'curr_{ds_type}_ds'), new_idx)
            setattr(self, f'curr_{ds_type}_ds', new_idx)
        return True
    
    def count_train(self, train_counter):
        counter = self.count('train', train_counter)
        self.load_next_train_record_idx(self.curr_train_ds)
        return counter
    
    def count_val(self, val_counter):
        counter = self.count('val', val_counter)
        self.load_next_val_record_idx(self.curr_val_ds)
        return counter
    
    def load_ds_idx(self, is_train : bool):
        if is_train:
            return self.curr_train_ds
        else:
            return self.curr_val_ds
    
    def load(self, keyword : str, is_train : bool):
        ds_type = 'train' if is_train else 'val'
        ds_idx = getattr(self, f'curr_{ds_type}_ds')
        record_idx = getattr(self, f'curr_{ds_type}_record')
        unloaded_data = getattr(self.dss[ds_idx], ds_type)[keyword][record_idx]
        return self.load_functions[ds_idx][keyword](unloaded_data)
    
    def load_new_ds(self, old_idx : int, new_idx : int, is_train : bool):
        ds_type = 'train' if is_train else 'val'
        setattr(self.dss[old_idx], ds_type, {})
        self.dss[new_idx] = utils.create_object_from_dict(self.dataset_configs[new_idx],
                                                          wrapper_class = Dataset,
                                                          seed = self.seed,
                                                          *self.args, **self.kwargs)
        
    def load_train_ds(self, old_idx : int, new_idx : int):
        return self.load_new_ds(old_idx, new_idx, is_train = True)
    
    def load_val_ds(self, old_idx : int, new_idx : int):
        return self.load_new_ds(old_idx, new_idx, is_train = False)
    
    def reinit_ds_idcs(self, ds_type = 'train', preserve_order = True):
        full_len = getattr(self, f'full_{ds_type}_len')
        if getattr(self, f'{ds_type}_balancing_strategy') in ('under', 'over'):
            if preserve_order:
                idcs = np.tile(np.arange(self.num_datasets), full_len // self.switch_frequency).repeat(self.switch_frequency)
            else:
                idcs = self.rng.permutation(np.tile(np.arange(self.num_datasets), full_len // self.switch_frequency)).repeat(self.switch_frequency)
        else: # if self.train_balancing_strategy == 'none'
            if preserve_order:
                idcs = []
                curr_idx = 0
                num_idcs_left = [num_records for num_records in getattr(self, f'{ds_type}_ds_sizes')]
                num_idcs_processed = 0
                while num_idcs_processed < full_len:
                    bundle_size = min(self.switch_frequency, num_idcs_left[curr_idx])
                    if bundle_size == self.switch_frequency:
                        idcs += [curr_idx] * bundle_size
                        num_idcs_processed += bundle_size
                    num_idcs_left[curr_idx] -= bundle_size
                    curr_idx = (curr_idx + 1) % self.num_datasets
            else:
                idcs = self.rng.permutation(sum([[ds_idx] * (ds_size // self.switch_frequency) for ds_idx, ds_size in enumerate(getattr(self, f'{ds_type}_ds_sizes'))], start = [])).repeat(self.switch_frequency)
        setattr(self, f'{ds_type}_ds_idcs', iter(idcs))
    
    # sets `self.train_ds_idcs` to be an iterator that will go over all the dataset indices in the train set
    def reinit_train_ds_idcs(self):
        self.reinit_ds_idcs('train', self.preserve_order)
        
    # same as `self.reinit_train_idcs` but for the validation set
    def reinit_val_ds_idcs(self):
        self.reinit_ds_idcs('val', preserve_order = True)

class BalancedDataLoader(object):
    """Wrapper class that creates an iterable batched dataset from raw data."""

    PARAMS = {
        'epoch': {
            'argument name': 'epoch_samling_method',
            'default': 'uniform'
            },
        'batch': {
            'argument name': 'batch_sampling_method',
            'default': 'uniform'
            },
        'sort_by': None,
        'pad_with': None,
        'relative_size': 1.0
    }

    
    BATCH_SIZE = 8 # default batch size

    SAMPLING_METHODS = {
        'oversampling': 'over',
        'undersampling': 'under',
        'positives only': 'pos_only',
        'uniform': 1
    }

    SAMPLING_PARAMS = {
        'ratio of positives': 0.5
    }

    REPLACE = True

    @staticmethod
    def fill_kwargs(config_dict : utils.ConfigDict):
        """
        Fill a ConfigDict with the default data sampling parameters.

        ConfigDict should have 'train' and 'val' keys.
        """

        config_dict.get_or_update('batch size', BalancedDataLoader.BATCH_SIZE)

        for ds_type in ('train', 'val'):
            curr_dict = config_dict.get_or_update(ds_type, {ds_type: {'default': {}}}, final = False)
            curr_dict.fill_with_defaults(BalancedDataLoader.PARAMS)

            epoch_sampling = curr_dict['epoch']
            if epoch_sampling.key() in ('oversampling', 'undersampling'):
                epoch_sampling.value().get_or_update('ratio of positives',
                                                     BalancedDataLoader.SAMPLING_PARAMS['ratio of positives'])
        
    def __init__(self, data : Dict[str, Any], config_dict : utils.ConfigDict,
                 bs : int = 1, actual_bs : int = 1, num_workers : int = 0,
                 transforms : Union[Callable, None] = None,
                 datapoints_per_sample : Optional[int] = 1,
                 *args, **kwargs):
        """
        Arguments:
            `data`: dict
            `config_dict`: ConfigDict; a config dict specifying the data sampling hyperparameters
            `bs`: int; (virtual) batch size
            `actual_bs`: int; actual batch size; can be smaller than `bs` in case of gradient accumulation
            `num_workers`: number of data loader workers for `torch.utils.data.DataLoader`
            `transforms`: callable; transforms to be applied to the data
        """
        epoch_sampling_method = config_dict['epoch'].key()
        datapoints_per_sample = datapoints_per_sample or 1
        with_replacement = config_dict.get('extra_datapoints_with_replacement', True)
        dataset = BalancedDataset(data,
                                  balanced = self.SAMPLING_METHODS.get(epoch_sampling_method.replace('_', ' '), epoch_sampling_method),
                                  pos_ratio_in_ds = config_dict['epoch'].get(
                                                                epoch_sampling_method + '/ratio of positives', 1
                                                                ),
                                  transforms = transforms,
                                  extra_datapoints = datapoints_per_sample - 1,
                                  sample_with_replacement = with_replacement, 
                                  relative_size = config_dict['relative_size'],
                                  **kwargs
                                  )
        batch_sampling = config_dict['batch']
        sort_by, pad_with = config_dict['sort_by'], config_dict['pad_with']
        if not data or any(len(values) == 0 for values in data.values()):
            self.dataloader = []
        elif 'uniform' in batch_sampling and sort_by is None and pad_with is None:
            self.dataloader = DataLoader(dataset, batch_size = actual_bs, shuffle = True, num_workers = num_workers)
        else:
            self.dataloader = DataIterator(
                                    dataset,
                                    min_pos_ratio = batch_sampling.get('min ratio of positives', 0.0),
                                    min_neg_ratio = batch_sampling.get('min ratio of negatives', 0.0),
                                    bs = bs,
                                    loaded_bs = actual_bs,
                                    sort_by = sort_by,
                                    pad_with = pad_with
            )
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        return next(self.dataloader)


