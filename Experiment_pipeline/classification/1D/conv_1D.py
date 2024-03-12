import torch.nn as nn
import utils

from typing import Literal, Optional, Tuple, Union

class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm
    def forward(self, x):
        return x.permute(*self.perm)

class My_Conv1d(nn.Sequential):
    def __init__(
        self,
        conv1d_kwargs,
        batch_norm : bool = True,
        dropout : float = 0,
        spatial_dropout : bool = False,
        activation : utils.ConfigDict = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}}
    ):
        conv = nn.Conv1d(**conv1d_kwargs, bias = not batch_norm)
        
        seq = [conv,utils.create_object_from_dict(utils.ConfigDict(activation), convert_to_kwargs = True)]
        if dropout>0:
            if spatial_dropout:
                seq.append(Permute(perm = (0, 2, 1)))
                seq.append(nn.Dropout2d(p = dropout))
                seq.append(Permute(perm = (0, 2, 1)))
            else:
                seq.append(nn.Dropout(p = dropout))  
        if batch_norm:
            seq.append(nn.BatchNorm1d(num_features = conv1d_kwargs["out_channels"]))

        super().__init__(*seq)

class MLP(nn.Sequential):
    def __init__(
        self,
        channel_sizes : Tuple[int] = (64,1),
        batch_norm : bool = False,
        dropout : float = 0,
        activation : utils.ConfigDict = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}},
        bias : bool = True
    ):
        seq = []
        for ic,oc in zip(channel_sizes[:-2],channel_sizes[1:-1]):
            seq.append(nn.Linear(ic,oc, bias = bias))
            seq.append(utils.create_object_from_dict(utils.ConfigDict(activation), convert_to_kwargs = True))
            if dropout>0:
                seq.append(nn.Dropout(p = dropout))
            if batch_norm:
                seq.append(nn.BatchNorm1d(num_features = oc))
            
        seq.append(nn.Linear(*channel_sizes[-2:], bias = bias))

        super().__init__(*seq)

class Net(nn.Module):
    def __init__(
        self, 
        kernel_size : int = 3,
        conv1D_channels : Tuple[int] = (16,64),
        mlp_channels : Tuple[int] = (64,1),
        aggregation : str = "torch.nn.AdaptiveMaxPool1d",
        activation : utils.ConfigDict = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}},
        spatial_dropout : bool = False,
        dropout : float = 0.2,
        batch_norm : bool = True,
        padding_mode : str = "circular"
        #*args, **kwargs
    ):
        super().__init__()
                
        self.is_training = True

        ### SEQ2SEQ ### 

        if len(conv1D_channels)<2:      
            self.seq2seq = nn.Identity()
        else:
            seq = []
            for ic,oc in zip(conv1D_channels[:-1],conv1D_channels[1:]):
                conv1d_kwargs = {
                    'in_channels': ic,
                    'out_channels': oc,
                    'kernel_size': kernel_size,
                    'padding': 'same',
                    "padding_mode": padding_mode
                }
                seq.append(My_Conv1d(conv1d_kwargs, batch_norm = batch_norm, dropout=dropout, spatial_dropout=spatial_dropout, activation=activation))
            self.seq2seq = nn.Sequential(*seq)
        
        ### PERMUTATION INVARIANT AGGREGATION ###

        self.aggregation = utils.create_object_from_dict(config_dict=utils.ConfigDict({aggregation:1}))

        ### REGRESSION ###

        if len(mlp_channels)<2:      
            self.fc = nn.Identity()
        else:
            self.fc = MLP(channel_sizes=mlp_channels, batch_norm = batch_norm, dropout=dropout, activation=activation)
        
    def forward(self, batch):
        #print("batch",batch.shape)
        
        seq = self.seq2seq(batch)
        #print("seq = self.seq2seq(batch)",seq.shape)

        vec = self.aggregation(seq).squeeze(dim = 2)
        #print("vec = self.aggregation(seq).squeeze(dim = 2)",vec.shape)
        
        pred = self.fc(vec)
        #print("pred = self.fc(vec)",pred.shape)
        #input()
        return pred

    def get_nbr_of_params(self):
        return sum(p.numel() for p in self.parameters())
