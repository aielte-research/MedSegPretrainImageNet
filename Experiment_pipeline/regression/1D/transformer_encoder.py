import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LayerNorm(nn.Module):
    "Layer normalization in the TF style (epsilon inside the square root)."
    def __init__(
        self,
        dim : int = 16,
        variance_epsilon : float = 1e-12
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

# GELU Activation: https://arxiv.org/abs/1606.08415
# gelu = lambda x : x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(
        self,
        dim : int = 16,
        dim_ff : int = 16
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

# class MultiHeadedSelfAttention(nn.Module):
#     """ Multi-Headed Dot Product Attention """
#     def __init__(
#         self,
#         dim : int = 16,
#         n_heads : int = 4,
#         dropout : float = 0
#     ):
#         super().__init__()
#         self.proj_q = nn.Linear(dim, dim)
#         self.proj_k = nn.Linear(dim, dim)
#         self.proj_v = nn.Linear(dim, dim)
#         self.drop = nn.Dropout(dropout)
#         self.scores = None # for visualization
#         self.n_heads = n_heads

#     def forward(self, x):
#         """
#         x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
#         mask : (B(batch_size) x S(seq_len))
#         * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
#         """
#         # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
#         q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
#                    for x in [q, k, v])
#         # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
#         scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
#         scores = self.drop(F.softmax(scores, dim=-1))
#         # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
#         h = (scores @ v).transpose(1, 2).contiguous()
#         # -merge-> (B, S, D)
#         h = merge_last(h, 2)
#         self.scores = scores
#         return h

class MultiHeadedSelfAttention(nn.MultiheadAttention):
    """ Multi-Headed Dot Product Attention """
    def __init__(
        self,
        dim : int = 16,
        n_heads : int = 4,
        dropout : float = 0
    ):
        super().__init__(dim, n_heads, dropout)

    def forward(self, x):
        return super().forward(x,x,x, need_weights=False)[0]

class Block(nn.Module):
    """ Transformer Block """
    def __init__(
        self,
        dim : int = 16,
        dim_ff : int = 16,
        dropout : float = 0,
        n_heads : int = 4
    ):
        super().__init__()
        
        self.attn = MultiHeadedSelfAttention(dim, n_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim,dim_ff)
        self.norm2 = LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.attn(x)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h

class Transformer_Encoder(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(
        self,
        dim : int = 16,
        dim_ff : int = 16,
        dropout : float = 0,
        n_heads : int = 4,
        n_layers : int = 2
    ):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim,dim_ff,dropout,n_heads) for _ in range(n_layers)])

    def forward(self, x):
        h=x.permute(2, 0, 1)
        for block in self.blocks:
            h = block(h)
        return h.permute(1, 2, 0)
    
    def get_nbr_of_params(self):
        return sum(p.numel() for p in self.parameters())

    