import math
import random
from typing import Any, Callable, Dict, Literal, Optional, Union

import numpy as np
import torch

from utils.config_dict import ConfigDict

def calc_balance_factor(pos_len, neg_len, pos_ratio_in_dataset, **kwargs) -> float:
    # pos_ratio_in_dataset: ratio of positive elements in the final dataset after over/under sampling
    balance_factor = pos_ratio_in_dataset * neg_len / ((1-pos_ratio_in_dataset) * pos_len)
    
    return balance_factor       # A visszaadott balance_factor nem feltétlenül egész szám!


class BalancedDataset(torch.utils.data.Dataset):
    """
    A custom dataset object, that can handle various data-balancing methods.
    
    Parameters:
        `images`: array-like object containing the image files in channels-last format
        `labels`: array-like object containing the masks (as two-dimensional arrays)
        `sources`: array-like object that where sources[:,1] is '1' if the corresponding image contains positive pixels
        `balanced`: possible values are 'over', 'under', 'pos_only' and positive integers
                if 'over': includes postive examples multiple times to achive positive ratio in the dataset defined by the pos_ratio_in_dataset parameter
                if 'under': excludes some negative examples to achive positive ratio in the dataset defined by the pos_ratio_in_dataset parameter
                if 'pos_only': only includes positive patches in the final dataset
                if integer: includes positive examples this many times
        `transforms`: optional callable object that transforms the images and masks
        `load_function`: optional callable; if given, this will be used to load data poits from disk (eg. `np.load`)
    """
    def __init__(self,
                 data : Dict[str, Any],
                 balanced : Union[int, Literal['over', 'under', 'pos_only']] = 'over',
                 relative_size : float = 1,
                 transforms : Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 load_function : Optional[Callable[[str], Any]] = None,
                 pos_ratio_in_ds : float = 1,
                 partition_count : Optional[int] = None,
                 extra_datapoints : int = 0,
                 seed : Optional[int] = None,
                 sample_with_replacement : bool = True,
                 *args, **kwargs):
        super(BalancedDataset, self).__init__()
        
        self.rng = np.random.default_rng(seed)
        
        full_len = len(list(data.values())[0])
        self.len_data = int(full_len * relative_size)
        idcs = np.arange(full_len)[self.rng.permutation(full_len) < self.len_data]
        self.data = {key: [value[idx] for idx in idcs] for key, value in data.items()}
        
        self.pos_only = balanced == 'pos_only'
        
        self.k = extra_datapoints       # NOTE: only works for cases, where everything is labeled as positive
        self.replacement = sample_with_replacement   
        
        self.load_data = load_function is not None

        if load_function is not None:
            if isinstance(load_function, (dict, ConfigDict)):
                def load(data):
                    for name, array_paths in data.items():  # NOTE: I call it array_paths, but it can be either a list of paths or just one path
                        if name in load_function:
                            if self.k > 0:
                                data[name] = [load_function[name](array_path) for array_path in array_paths]
                            else:
                                data[name] = load_function[name](array_paths)
                    return data
            else:
                def load(data):        
                    for name, array_paths in data.items():
                        if self.k > 0:
                            if isinstance(array_paths[0], str):
                                data[name] = [load_function(array_path) for array_path in array_paths]
                        else:                            
                            if isinstance(array_paths, str):
                                data[name] = load_function(array_paths)
                    return data
            self.load = load

        #separating postitve and negative patches
        if 'positives' in data:
            positives = self.data.pop('positives')
        elif 'label' in data and all(label in (0, 1) for label in data['label']):
            positives = self.data['label']
        else:
            # if indices are not explicitly given, all data is treated as positive
            # (will not work together with balanced sampling methods)
            positives = np.ones(self.len_data, dtype = bool)
        positives = np.array(positives).astype(bool)
        negatives = ~positives

        all_idcs = np.arange(self.len_data)
        self.positive_idcs = all_idcs[positives]
        self.negative_idcs = all_idcs[negatives]

        self.pos_len, self.neg_len = positives.sum(), negatives.sum()

        if balanced=='pos_only':
            self.data = {data_type: np.array(tensor)[positives] for data_type, tensor in self.data.items()}
            self.balance_factor = 1
        
        elif balanced=='over':
            # The multiplier of positive example is always an integer, but it may ruin the intended ratio a little  bit
            # We save the spare amount and deal with it later (not now in order to let every positive sample be included in the dataset the same number of times)
            self.balance_factor=calc_balance_factor(self.pos_len, self.neg_len, pos_ratio_in_ds)
            self.cut = {'class':'pos', 'amount': (math.ceil(self.balance_factor) - self.balance_factor) * self.pos_len}       
            
        elif balanced=='under':
            # In case of undersampling we need to drop some negative examples. 
            # We save the number to drop and do it later for similar reasons as in the oversampling case
            self.cut_factor = 1 / calc_balance_factor(self.pos_len, self.neg_len, pos_ratio_in_ds)
            self.balance_factor = 1
            self.cut = {'class' : 'neg', 'amount' : self.neg_len * (1-self.cut_factor)}
                
        else:
            self.cut = {'amount': 0}
            self.balance_factor=balanced        

        self.transforms=transforms
        self.partition_count = partition_count if partition_count else 1

    @torch.no_grad()
    def __getitem__(self, index):
        # TODO: load each datapoint from memory if needed
        idx = int(index / self.partition_count)

        if self.pos_only:
            pass
        elif index < self.neg_len:
            idx = self.negative_idcs[idx]
        else:
            idx = self.positive_idcs[(idx - self.neg_len) % self.pos_len]

        if self.k > 0:
            possible_idcs = np.arange(self.pos_len)
            if not self.replacement:
                possible_idcs = possible_idcs[possible_idcs != idx]
            extra_idxs = self.rng.choice(possible_idcs, self.k, replace = self.replacement)
            index = [index, *extra_idxs]
            data = {name: [array[i] for i in [idx, *extra_idxs]] for name, array in self.data.items()}
        else:
            extra_idxs = []
            data = {name: array[idx] for name, array in self.data.items()} 
        
        if self.load_data:
            data = self.load(data)
        
        # NOTE: at this point every value in the data dictionary is either an array of the selected datapoints (if k>0) or just one datapoint (if k=0)

        # Applying transforms
        if self.transforms:
            data = self.transforms(**data, _index = index, k = self.k + 1)
            data.pop('_index', None), data.pop('k', None)
        
        if self.k > 0:
            for name, values in data.items():
                if isinstance(values, list):
                    data[name] = values[0]
        
        return data

    def __len__(self):
        if self.pos_only:
            real_length = self.pos_len
        else:
            real_length = self.neg_len + math.ceil(self.balance_factor) * self.pos_len
        
        return self.partition_count * real_length
          
def create_index_list(ds_len, nr_pos, reuse, p, n, bs, cut):
    '''
    A függvény, ami megadja az indexek listáját, ami szerint végig kell haladni az adathalmaz elemein az adott epochban.
    ds_len: Az adathalmaz mérete.
    nr_pos: A datasetben a pozitívan annotált elemek száma.
    reuse: A lefixált pozitív és negatív adatpontokat újrahasználjuk-e mikor feltöltjük a batchekben maradt helyet véletlenszerűen. Opciók: "no" (ezek jelenleg nem: "all", "pos")
    p, n: A batchekben lefixált pozitív/negatív adatpontok száma
    bs: annak a felosztásnak a mérete, ami szerint szeretnénk, hogy biztos legyen benne fix mennyiségű poz vagy neg elem
    '''
    
    index_list=list(range(ds_len))
    new_index_list=[]
    
    # Calculates number of batches in an epoch with the given configuration
    def calculate_batch_nr(ds_len, cut, bs):
        return int((ds_len-cut['amount']) / bs)
    
    # Cuts spare positive or negative examples depending on sampling method
    def make_cut(pos, neg, cut):
        cut_size=int(cut['amount'])
        if cut_size == 0:
            return pos[:] + neg[:]
        if cut['class']=='pos':
            rest = pos[:-cut_size] + neg[:]
        elif cut['class']=='neg':
            rest = pos[:] + neg[:-cut_size]  
        return rest        

    batch_nr=calculate_batch_nr(ds_len, cut, bs)
    s=bs-p-n
    
    nr_neg=ds_len-nr_pos
    
    # Shuffles all negative and positive samples
    shuf_pos=random.sample(index_list[nr_neg:], nr_pos)
    shuf_neg=random.sample(index_list[:nr_neg], nr_neg)

# ----
#     if reuse=="all":
#         rest=make_cut(shuf_pos, shuf_neg, cut)
#         shuffled_indices=random.sample(rest, len(rest))
# #         print("Újrahasználjuk")
#     elif reuse=="pos":
#         rest=make_cut(shuf_pos, shuf_neg[n*batch_nr:], cut)
#         shuffled_indices=random.sample(rest, len(rest))
# #         print("Pozitívakat használjuk újra")
# ----

    if reuse=="no":
        # Cuts spare images from those which are used to fill the not fixed positive or negative positions
        rest=make_cut(shuf_pos[p*batch_nr:], shuf_neg[n*batch_nr:], cut)
        shuffled_indices=random.sample(rest, len(rest))

    # Fills every batch with the requested amount of pos and neg example and with the rest, then shuffles it
    for i in range(batch_nr):
        fix_pos=shuf_pos[i*p:(i+1)*p]
        fix_neg=shuf_neg[i*n:(i+1)*n]
        rest_of_batch=shuffled_indices[i*s:(i+1)*s]
        batch=fix_pos + fix_neg + rest_of_batch
        random.shuffle(batch)
        new_index_list = new_index_list + batch 
        
    return batch_nr, new_index_list

  
# TODO (or to consider): padding now only considers the actual loaded batch,
# so using padding with or without gradient accumulation is not equivalent
def get_batch(dataset, idx_en, batch_size, pad_with = None):
    '''
    Megkreálja a batchet.
    dataset: Balanceddataset osztályú adathalmaz
    idx_en: enumerate object, meghatározza milyen sorrendben kérjük az adathalmaz elemeit
    batch_size: batch_size (amit egyszerre betöltünk)
    '''
    batch = {}
    
    for _ in range(batch_size):
        # Picks the next element from the dataset. The index is defined by the current element of idx_en.
        state, idx=next(idx_en)
        datapoint = dataset[idx]
        for key, value in datapoint.items():
            if key not in batch:
                batch[key] = [value]
            else:
                batch[key].append(value)
    
    for key, values in batch.items():
        if pad_with is not None:
            shapes = [value.shape[-1] for value in values]
            max_len = max(shapes)
            for i, value in enumerate(values):
                values[i] = torch.concat([value, pad_with * torch.ones((*value.shape[:-1], max_len - value.shape[-1]))], axis = -1)
        stacked_values = np.stack(values)
        batch[key] = torch.tensor(stacked_values)
    
    return state, batch



class DataIterator:
    '''Egy iterable object, amin végigiterálva megkapjuk a batcheket. 
    
    dataset: Mindenképp a Balanceddataset objektum kell legyen a dataset.
    (nem használjuk) reuse: A lefixált pozitív és negatív adatpontokat újrahasználjuk-e mikor feltöltjük a batchekben maradt helyet véletlenszerűen. Opciók: "no", "all", "pos"
    min_pos_ratio, min_neg_ratio: A batchekben lefixált pozitív/negatív adatpontok arány a batchmérethez (bs) képest
    bs: annak a felosztásnak a mérete, ami szerint szeretnénk, hogy biztos legyen benne fix mennyiségű poz vagy neg elem
    loaded_bs: A valódi (egyszerre betöltött) batchek mérete
    sort_by: a kulcs, ami szerint rendezni akarjuk az adathalmazt
    pad_with: None, vagy az az érték, amivel padelni akarunk az utolsó tengely szerint, ha egy batchben nem csupa ugyanolyan hosszú adat van
    '''
    def __init__(self, dataset, min_pos_ratio, min_neg_ratio, bs, loaded_bs,
                 reuse = 'no', sort_by = None, pad_with = None):
        self.loaded_batch_size=loaded_bs    #Ekkora batchet ad valójában
        self.dataset=dataset
        self.ds_len=len(self.dataset)
        self.reuse = reuse
        self.p = int(min_pos_ratio * bs)
        self.n = int(min_neg_ratio * bs)
        self.bs = bs
        self.cut = self.dataset.cut
        
        self.sort = sort_by is not None
        if self.sort:
            if not isinstance(sort_by, str):
                sort_by = sort_by.key()
            self.sort_keys = self.dataset.data[sort_by] if sort_by else None
            self.sort_keys = [float(key) for key in self.sort_keys]
        self.pad_with = pad_with
        
        self.nr_pos=self.dataset.pos_len * math.ceil(self.dataset.balance_factor)
        self.load_next_epoch()
        
    def load_next_epoch(self):
        self.batch_nr, self.idx_list=create_index_list(
            self.ds_len, self.nr_pos, self.reuse, self.p, self.n, self.bs, self.cut
            )

        if self.sort:
            self.idx_list.sort(key = self.sort_keys.__getitem__)

        self.idx_list_len=len(self.idx_list)
        self.idx_en=enumerate(self.idx_list)
        
        self.index=0
        

    def __iter__(self):
        return(self)
    
    # Amíg van még elég elem az idx_list-ben, addig betölt egy új batch-et
    def __next__(self):
        if self.index < self.idx_list_len - self.loaded_batch_size:
            state, batch=get_batch(self.dataset, self.idx_en, self.loaded_batch_size, pad_with = self.pad_with)
            self.index=state
            return(batch)
        self.load_next_epoch()
        raise StopIteration
    
    def __len__(self):
        return math.ceil(self.idx_list_len / self.loaded_batch_size)
