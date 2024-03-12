import torch
import torch.nn as nn
from utils import ConfigDict,create_object_from_dict
def get_object(x):
    if isinstance(x, ConfigDict) and isinstance(list(x.values())[0], ConfigDict):
        return create_object_from_dict(ConfigDict(x), convert_to_kwargs = True)
    if isinstance(x, str):
        return create_object_from_dict(ConfigDict({x:{}}), convert_to_kwargs = True) 
    return create_object_from_dict(config_dict=ConfigDict(x))        

from typing import Literal, Optional, Tuple, Union

import logging

logger = logging.getLogger("soundy")
logger.setLevel(10)

streamHandler = logging.StreamHandler()
logger.addHandler(streamHandler)
streamHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))

# fileHandler = logging.FileHandler(f"last_run.log", mode='w')
# logger.addHandler(fileHandler)
# fileHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))

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
        activation : Union[str,ConfigDict] = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}}
    ):
        seq = [
            nn.Conv1d(**conv1d_kwargs, bias = not batch_norm),
            get_object(activation)
        ]
        
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

class MLC(nn.Sequential):
    def __init__(
        self,
        channel_sizes : Tuple[int] = (64,1),
        kernel_size: int = 3,
        batch_norm : bool = False,
        dropout : float = 0,
        activation : Union[str,ConfigDict] = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}},
        padding_mode : str = "circular",
        spatial_dropout : bool = False,
    ):
        seq = []

        for ic,oc in zip(channel_sizes[:-1],channel_sizes[1:]):
            conv1d_kwargs = {
                'in_channels': ic,
                'out_channels': oc,
                'kernel_size': kernel_size,
                'padding': 'same',
                'padding_mode': padding_mode
            }
            seq.append(My_Conv1d(
                conv1d_kwargs,
                batch_norm = batch_norm,
                dropout = dropout,
                spatial_dropout = spatial_dropout,
                activation = activation
            ))

        super().__init__(*seq)

class MLP(nn.Sequential):
    def __init__(
        self,
        channel_sizes : Tuple[int] = (64,1),
        batch_norm : bool = False,
        dropout : float = 0,
        activation : Union[str,ConfigDict] = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}},
        bias : bool = True
    ):
        seq = []
        for ic,oc in zip(channel_sizes[:-2],channel_sizes[1:-1]):
            seq.append(nn.Linear(ic,oc, bias = bias))
            seq.append(get_object(activation))
            if dropout>0:
                seq.append(nn.Dropout(p = dropout))
            if batch_norm:
                seq.append(nn.BatchNorm1d(num_features = oc))
            
        seq.append(nn.Linear(*channel_sizes[-2:], bias = bias))

        super().__init__(*seq)

class LSTM(nn.Module):
    def __init__(
        self, 
        input_size : int = 64,
        hidden_size : int = 64,
        num_layers : int = 2,
        bidirectional : bool = False,
        residual : bool = False
        #*args, **kwargs
    ): 
        super().__init__()

        self.residual=residual

        if residual:
            self.lstms = nn.ModuleList([nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional) for i in range(num_layers)])
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)

        #the lstm weights need to be initialized manually as by default they are suboptimal
        # for name, param in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)   

    def forward(self, x):
        x=x.permute(2, 0, 1)
        if self.residual:
            for lstm in self.lstms:
                x=x+lstm(x)[0]
            return x.permute(1, 2, 0)
        else:
            return self.lstm(x)[0].permute(1, 2, 0)

class Net(nn.Module):
    def __init__(
        self, 
        mlp_channels : Tuple[int] = (64,1),
        aggregation : Union[str,ConfigDict] = {'torch.nn.AdaptiveMaxPool1d':1},
        activation :  Union[str,ConfigDict] = {'torch.nn.LeakyReLU': {'negative_slope': 0.2}},
        dropout : float = 0.2,
        batch_norm : bool = True,
        seq2seq : Union[str,ConfigDict] = "torch.nn.Identity",
        model_state_dict_path : Union[str,None] = None,
        discr_dims : Tuple[int] = (6,),
        discr_embs : Tuple[int] = (3,),
        pass_all_input: bool = True
        #*args, **kwargs
    ):                
        super().__init__()
        self.is_training = True           
                
        ### EMBEDDING ###
        if not isinstance(discr_dims,tuple):
            discr_dims = (discr_dims,)
        if not isinstance(discr_embs,tuple):
            discr_embs = (discr_embs,)
        self.embedding = nn.ModuleList([nn.Embedding(d,e) for d,e in zip(discr_dims, discr_embs)])
    
        ### PLACEHOLDER seq2seq ###
        self.seq2seq = get_object(seq2seq)

        ### PERMUTATION INVARIANT AGGREGATION ###
        self.aggregation = get_object(aggregation)

        ### REGRESSION ###
        if len(mlp_channels)<2:      
            self.fc = nn.Identity()
        else:
            self.fc = MLP(channel_sizes=mlp_channels, batch_norm = batch_norm, dropout=dropout, activation=activation)
        
        if model_state_dict_path!=None:
            self.load_state_dict(torch.load(model_state_dict_path))
        
    def forward(self, x0, x1=None, **kwargs):
        if x1 is not None:
            x1=x1.long()
            # print("x1=x1.long()",x1.shape)
            cols = [torch.squeeze(x,1) for x in torch.tensor_split(x1, x1.shape[1], dim=1)]
            # print("[torch.squeeze(x,1) for x in torch.tensor_split(x1, x1.shape[1], dim=1)]",[x.shape for x in cols])     
            embedded = torch.cat([x0]+[emb(col).permute(0,2,1) for col,emb in zip(cols,self.embedding)], 1)            
        else:
            embedded = x0
        #print("embedded",embedded.shape)
        seq = self.seq2seq(embedded)
        #print("seq = self.seq2seq(emb)",seq.shape)
        vec = self.aggregation(seq).squeeze(dim = 2)
        #print("vec = self.aggregation(seq).squeeze(dim = 2)",vec.shape)
        pred = self.fc(vec)
        #print("pred = self.fc(vec)",pred.shape)
        #input()
        return pred

    def get_nbr_of_params(self):
        return sum(p.numel() for p in self.parameters())
