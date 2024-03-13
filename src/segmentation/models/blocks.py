from typing import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import torch
from torch import nn
import model
import utils


class ConcatLinearBlock(nn.Module):
    def __init__(self, x_channels, x_up_channels, skip_channels, out_channels, *args, **kwargs):
        super().__init__()
        
        self.linear_layer = nn.Linear(x_channels, out_channels)
        
    def forward(self, x, x_up, skip_val):
        x = torch.cat((x_up, skip_val), dim=2)
        x = self.linear_layer(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size=4, bias=True, dilation=1, groups=1, drop_rate=0.):
        super().__init__()
        
        self.in_chans = in_channels
        self.embed_dim = out_channels

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias, dilation=dilation, groups=groups)
        self.norm_layer = nn.LayerNorm(out_channels)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm_layer(x)
        x = self.pos_drop(x)
        
        return x
    
    
class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer='torch.nn.LayerNorm'):
        super().__init__()
        
        assert 2*in_channels == out_channels, f'Incorrect in_channels and outchannels ({in_channels}, {out_channels})'
        
        dim = in_channels
        
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        norm_layer = utils.get_class_constr(norm_layer)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(torch.sqrt(torch.tensor(L)))
        
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        
        return x
    
    
class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels, dim_scale=2, norm_layer='torch.nn.LayerNorm'):
        super().__init__()
        
        assert in_channels == 2*out_channels, f'Incorrect in_channels and outchannels ({in_channels}, {out_channels})'
        
        dim = in_channels
        
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        norm_layer = utils.get_class_constr(norm_layer)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H = W = int(torch.sqrt(torch.tensor(x.shape[1])))
        x = self.expand(x)
        B, _, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=torch.div(C, 4, rounding_mode='floor'))
        x = x.view(B,-1,torch.div(C, 4, rounding_mode='floor'))
        x = self.norm(x)
        
        return x
    
    
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, in_channels, out_channels, dim_scale=4, norm_layer='torch.nn.LayerNorm'):
        super().__init__()
        
        dim = in_channels
        
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        norm_layer = utils.get_class_constr(norm_layer)
        self.norm = norm_layer(self.output_dim)
        
        self.output = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.init_norm = norm_layer(in_channels)

    def forward(self, x):
        x = self.init_norm(x)
        
        B0, L0, _ = x.shape
        H = W = int(torch.sqrt(torch.tensor(L0)))
        x = self.expand(x)
        B, _, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=torch.div(C, self.dim_scale**2, rounding_mode='floor'))
        x = x.view(B,-1,self.output_dim)

        x = self.norm(x)
        
        # last steps
        x = x.view(B0,4*H,4*W,-1)
        x = x.permute(0,3,1,2) #B,C,H,W
        x = self.output(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, act_layer, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='floor')).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(torch.div(B_, nW, rounding_mode='floor'), nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, img_size, patch_size,
                 num_heads_layers,
                 in_channels, out_channels = None,
                 activations = 'gelu',
                 embed_dim = 96,
                 window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer='torch.nn.LayerNorm',
                 *args, **kwargs):
        super().__init__()
        
        out_channels = out_channels or in_channels
        if in_channels != out_channels:
            msg = f'Swin transformer block should not change channel size, but got in channel size {in_channels} and out channel size {out_channels}.'
            raise ValueError(msg)
        
        i = int(torch.log2(torch.tensor(in_channels//embed_dim)))
        
        num_heads = num_heads_layers[i]
        
        self.dim = in_channels
        embed_img_size = img_size // patch_size
        self.input_resolution = (int(embed_img_size // 2**i),) * 2
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = (kwargs['position'] % 2) * window_size // 2
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        def window_partition(x, window_size):
            B, H, W, C = x.shape
            HW = torch.div(H, window_size, rounding_mode='floor')
            WW = torch.div(W, window_size, rounding_mode='floor')
            x = x.view(B, HW, window_size, WW, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows

        def window_reverse(windows, window_size, H, W):
            B = int(windows.shape[0] / (H * W / window_size / window_size))
            HW = torch.div(H, window_size, rounding_mode='floor')
            WW = torch.div(W, window_size, rounding_mode='floor')
            x = windows.view(B, HW, WW, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x

        self.window_partition = window_partition
        self.window_reverse = window_reverse
        
        norm_layer = utils.get_class_constr(norm_layer)
        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention(
            self.dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        stochastic_depth_rate = kwargs['stochastic_depth_rate']
        self.drop_path = DropPath(stochastic_depth_rate) if stochastic_depth_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        def block_activation():
            act_name = activations.key() if isinstance(activations, utils.config_dict.ConfigDict) else activations
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activations,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activations, convert_to_kwargs = True)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=block_activation, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size L={L}, H={H}, W={W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DoubleSwinTransformerBlock(nn.Module):

    def __init__(self, in_channels, out_channels, img_size, patch_size, depth, num_heads_layers, activations = 'gelu', embed_dim=96, 
                window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout_rate=0., dropout_att_rate=0.,
                drop_path=0., norm_layer=nn.LayerNorm, *args, **kwargs):
        super(DoubleSwinTransformerBlock, self).__init__()
        
        assert in_channels == out_channels, f'in_channels and out_channels are not equal ({in_channels}, {out_channels})'
        
        # recalculate layer index
        i = int(torch.log2(torch.tensor(in_channels//embed_dim)))
        
        num_heads = num_heads_layers[i]
        
        self.dim = in_channels
        embed_img_size = img_size // patch_size
        self.input_resolution = (int(embed_img_size // 2**i), int(embed_img_size // 2**i))

        self.normalize = embed_dim*2**(len(num_heads_layers)-1) == out_channels
        if self.normalize:
            self.last_norm = norm_layer(out_channels)
            
        def block_activation():
            act_name = activations.key() if isinstance(activations, utils.config_dict.ConfigDict) else activations
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activations,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activations, convert_to_kwargs = True)
            
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=self.dim, input_resolution=self.input_resolution,
                                 num_heads=num_heads, act_layer=block_activation, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=dropout_rate, attn_drop=dropout_att_rate,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
            
        if self.normalize:
            x = self.last_norm(x)
            
        return x


class ConvBlock(nn.Module):
    '''
    Argumnets:
        size: nr of conv layers in the block
        downsample_in_block: whether last conv layers should have >1 stride
        block_activation: dictionary of activation function used in the block
        ...

    '''
    PARAMS = {
        'activations': 'relu',
        'size': 2,
        'padding': 1,
        'kernel_size': 3,
        'dropout': False,
        'stride': None
        }
    
    DROPOUT_RATE = 0.5
    
    @staticmethod
    def fill_kwargs(config_dict):
        act_func_name, act_func_dict = config_dict['activations'].item()
        if act_func_name in model.activations.activation_funcs_dict:
            act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
            act_func_dict.fill_with_defaults(act_default_dict)
        elif act_default_dict is not None:
            utils.fill_dict(config_dict['activations'])
        
        if config_dict['dropout']:
            config_dict.get_or_update('dropout/rate', ConvBlock.DROPOUT_RATE)


    def __init__(self, in_channels, out_channels, size = 2, kernel_size = 3, padding = 1,
                 activations = 'relu', dropout = False, stride = None, downsample_in_block = False, *args, **kwargs):
        super(ConvBlock, self).__init__()

        def conv_layer(in_channels, downsampling = False):
            stride_ = stride or (2 if downsampling else 1)
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride_, padding=padding, bias=True)
            return conv_layer

        def batch_norm():
            bn = nn.BatchNorm2d(out_channels)
            return bn

        def block_activation():
            act_name = activations.key() if isinstance(activations, utils.config_dict.ConfigDict) else activations
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activations,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activations, convert_to_kwargs = True)

        layers = []

        for i in range(size):
            downs = downsample_in_block if i == size-1 else False
            in_channels = in_channels if i == 0 else out_channels
            layers.append(conv_layer(in_channels=in_channels, downsampling=downs))
            layers.append(batch_norm())
            layers.append(block_activation())
        
        # optional dropout layer
        if dropout:
            layers.append(nn.Dropout2d(p = dropout['rate']))

        self.block = nn.Sequential(*layers)


    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):

    PARAMS = {
            'activation': 'relu',
            'kernel_size' : 2,
            'scale_factor': 2
    }

    @staticmethod
    def fill_kwargs(config_dict):
        act_func_name, act_func_dict = config_dict['activation'].item()
        if act_func_name in model.activations.activation_funcs_dict:
            act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
            act_func_dict.fill_with_defaults(act_default_dict)
        elif act_default_dict is not None:
            utils.fill_dict(config_dict['activation'])

    
    def __init__(self, in_channels, out_channels, activation = 'relu', kernel_size = 2, scale_factor = 2,
                 *args, **kwargs):
        super(UpConvBlock, self).__init__()

        def conv_layer():
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', bias=True)
            return conv

        def act_func():
            act_name = activation.key() if isinstance(activation, utils.config_dict.ConfigDict) else activation
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activation,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activation, convert_to_kwargs = True)

        self.convup=nn.Sequential(
            nn.Upsample(scale_factor = scale_factor),
            conv_layer(),
            act_func()
        )

    def forward(self, x):
        x = self.convup(x)
        return x



class MixingBlock(nn.Module):
    """ 
    A wrapper module for mixing_blocks that are used in the decoder to comine skip connection data and data from previous level.
    These blocks should have a get_out_ch method that (given the channel number of the plain and upsampled data from prev. level, 
    the channel number of the skip con. data and the output channel number of the current level) calculates the output channel number of itself.
    """
    def __init__(self, **kwargs):
        super().__init__()
    def get_out_ch(self, x_channels, x_up_channels, skip_channels, level_out_channels):
        return x_up_channels + skip_channels

class AttentionBlock(MixingBlock):
    """ 
    This is an implementation of the attention block for attention U-Net: https://arxiv.org/abs/1804.03999
    Gating signal part can be customize in the config dict.
    """

    PARAMS = {
        # 'activations': 'relu',
        'gating signal': {
            'argument name': 'gating_signal',
            'default': {'segmentation.models.blocks.ConvBlock': {'size': 1, 
                                                                 'kernel size': 1,
                                                                 'padding': 0}}
            },
        'halve_channels': False
        }
    
    @staticmethod
    def fill_kwargs(config_dict):
        # act_func_name, act_func_dict = config_dict['activations'].item()
        # act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
        # act_func_dict.fill_with_defaults(act_default_dict)
        
        utils.fill_dict(config_dict['gating signal'])
        # if config_dict['dropout']:
        #     config_dict.get_or_update('dropout/rate', ConvBlock.DROPOUT_RATE)


    def __init__(self, x_channels, x_up_channels, skip_channels, level_out_channels, gating_signal, *args, **kwargs):
        super(AttentionBlock, self).__init__()

        def make_gs_block(in_channels, out_channels):
            return utils.create_object_from_dict(gating_signal, in_channels = in_channels, out_channels = out_channels,
                                                 wrapper_class = model.Model)

        def conv_layer(in_channels, out_channels, kernel_size, stride):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, bias = True)
            return conv
        
        def bn_layer(channels):
            bn = nn.BatchNorm2d(channels)
            return bn
        
        self.gs_block = make_gs_block(x_channels, x_channels)

        self.W_g=nn.Sequential(
            conv_layer(x_channels, x_channels, kernel_size=1, stride=1),
            bn_layer(x_channels)
        )

        self.W_s=nn.Sequential(
            conv_layer(skip_channels, x_channels, kernel_size=2, stride=2),
            bn_layer(x_channels)
        )

        self.psi = nn.Sequential(
            conv_layer(x_channels, skip_channels, kernel_size=1, stride=1),
            bn_layer(skip_channels),
            nn.Sigmoid()
        )

        self.upsample=nn.Upsample(scale_factor=2)
        self.relu =nn.ReLU()

        # TODO: upsample and activation dict

    def forward(self, x, x_up, skip_val):
        g = self.gs_block(x)
        x1=self.W_s(skip_val)
        g1=self.W_g(g)
        p=self.relu(x1+g1)
        p=self.psi(p)
        p=self.upsample(p)
        weighted_skip_val = skip_val*p
        return torch.cat((x_up, weighted_skip_val), dim=1)


class ConcatBlock(MixingBlock):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, x, x_up, skip_val):
        return torch.cat((x_up, skip_val), dim=1)



class ConcatLinearBlock(MixingBlock):
    def __init__(self, x_channels, x_up_channels, skip_channels, level_out_channels, *args, **kwargs):
        super().__init__()
        
        self.linear_layer = nn.Linear(x_channels, level_out_channels)
    
    def get_out_ch(self, x_channels, x_up_channels, skip_channels, level_out_channels):
        return level_out_channels
        
    def forward(self, x, x_up, skip_val):
        x = torch.cat((x_up, skip_val), dim=2)
        x = self.linear_layer(x)
        return x


class ZeroFillResConnection(nn.Module):
    
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.zero_channels = out_channels - in_channels
    
    def forward(self, x):
        shape = list(x.shape)
        shape[1] = self.zero_channels
        zeros = torch.zeros(shape, device = x.device)
        return torch.cat((x, zeros), axis = 1)
    
class RepeatResConnection(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        if out_channels % in_channels != 0:
            raise ValueError(
                f'Identity residual connection with expand mode `repeat` needs the out channel size to be divisible by the in channel size, but got in channel size {in_channels} and out channel size {out_channels}.'
                )
        super().__init__()
        self.repeats = out_channels // in_channels
        
    def forward(self, x):
        return torch.cat((x,) * self.repeats, axis = 1)
    
class AddResConnection(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        if in_channels % out_channels != 0:
            raise ValueError(
                f'Identity residual connection with contract mode `add` needs the in channel size to be divisible by the out channel size, but got in channel size {in_channels} and out channel size {out_channels}.'
                )
        super().__init__()
        self.out_channels = out_channels
        
    def forward(self, x):
        xs = x.split(self.out_channels, dim = 1)
        return sum(xs)

class CutOffResConnection(nn.Module):
    
    def __init__(self, out_channels, *args, **kwargs):
        super().__init__()
        self.out_channels = out_channels
    
    def forward(self, x):
        return x[:, :self.out_channels]
    

class ResConnection(nn.Module):
    
    """
    Helper class for creating residual connections.
    
    Its `type_dict` argument specifies the type of residual connection, as well as additional parameters. Possible types:
        `identity`: copies the input, or part of the input identically. Parameters:
            `expand_method`: needs to be specified if the output has more channels than the input. Possible values:
                `fill_with_zeros`: fills the additional channels with zeros.
                `repeat`: repeats the input (output_size / input_size) times.
            `contract_method`: needs to be specified if the input has more channels than the output. Possible values:
                `add`: partitions the input into chunks equal to the output size and returns their sum.
                `cut_off`: only returns the first few channels of the input, equal to the output size.
        `convolution`: applies a convolution operator, followed by an optional batch norm and activation function.
    """
    
    CONV_PARAMS = {'kernel_size': 1, 'batch_norm': True, 'activation': False}
    
    # TODO: implement identity with downsampling and fix padding value for conv skip connections
    def __init__(self, in_channels, out_channels, downsampling = False,
                 type_dict = utils.ConfigDict({'identity': {'expand_method': 'fill_with_zeros',
                                                            'contract_method': 'add'}})):
        super().__init__()
        
        if not isinstance(type_dict, utils.ConfigDict):
            type_dict = utils.ConfigDict({type_dict: {}})
        type_dict.expand()
        
        shortcut_type = type_dict.key()
        
        if shortcut_type == 'identity':
            if downsampling:
                raise NotImplementedError(f'Identity skip connection with changing spatial size is not implented.')
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            elif in_channels < out_channels:
                expand_method = type_dict.value()['expand_method'].key()
                if expand_method == 'fill_with_zeros':
                    self.shortcut = ZeroFillResConnection(in_channels, out_channels)
                elif expand_method == 'repeat':
                    self.shortcut = RepeatResConnection(in_channels, out_channels)
                else:
                    raise ValueError(
                        f'Expand method of identity residual connection should be one of `fill_with_zeros` or `repeat`, not `{expand_method}`.'
                    )
            elif in_channels > out_channels:
                contract_method = type_dict.value()['contract_method'].key()
                if contract_method == 'add':
                    self.shortcut = AddResConnection(in_channels, out_channels)
                elif contract_method == 'cut_off':
                    self.shortcut = CutOffResConnection(out_channels)
                else:
                    raise ValueError(
                        f'Contract method of identity residual connection should be one of `add` or `cut_off`, not `{contract_method}`.'
                    )
        elif shortcut_type in ('conv', 'convolution'):
            params_dict = type_dict.value()
            layers = [torch.nn.Conv2d(in_channels, out_channels,
                                      kernel_size = params_dict['kernel_size'],
                                      stride = 1 + downsampling)]
            if params_dict['batch_norm']:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            if params_dict['activation']:
                layers.append(torch.nn.ReLU())
            self.shortcut = torch.nn.Sequential(*layers)
        else:
            raise ValueError(
                f'Shortcut type should be one of `identity` or `convolution`, not `{shortcut_type}`.'
            )
    
    def forward(self, x):
        return self.shortcut(x)
            

class ChannelwiseLayerNorm(nn.LayerNorm):

    """Normalises the input along each channel using trainable weights."""

    def __init__(self, normalized_shape, smoothing_term = 1e-6, *args, **kwargs):
        super().__init__(normalized_shape, eps = smoothing_term, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        x = super().forward(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2)

class ResNeXtBlock(nn.Module):

    # (conv 1x1 C -> C/4) -> (conv d3x3 C/4 -> C/4) -> (conv 1x1 C/4 -> C)
    # roughly equivalent to (conv 3x3 C/4 -> C/4) -> (conv 3x3 C/4 -> C/4)
    # so an equivalent to a basic block with 96 channels in and 96 channels out
    # would be a ResNeXt block with 384 channels in and 384 channels out
    
    @staticmethod
    def fill_kwargs(config_dict):
        act_func_name, act_func_dict = config_dict['activations'].item()
        if act_func_name in model.activations.activation_funcs_dict:
            act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
            act_func_dict.fill_with_defaults(act_default_dict)
        elif act_default_dict is not None:
            utils.fill_dict(config_dict['activations'])

    def __init__(self, in_channels, out_channels, kernel_size = 3, activations = 'relu', **kwargs):

        super().__init__()
        
        def conv(in_channels, out_channels, kernel_size, groups = 1):
            return nn.Conv2d(in_channels, out_channels, kernel_size, padding = 'same', groups = groups)
        
        def norm(num_features):
            return nn.BatchNorm2d(num_features)
        
        def act():
            act_name = activations.key() if isinstance(activations, utils.config_dict.ConfigDict) else activations
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activations,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activations, convert_to_kwargs = True)

        self.pointwise_conv_1 = conv(in_channels, out_channels // 4, kernel_size = 1)
        self.norm1 = norm(out_channels // 4)
        self.act1 = act()

        self.depthwise_conv = conv(out_channels // 4, out_channels // 4, kernel_size = kernel_size, groups = out_channels // 4)
        self.norm2 = norm(out_channels // 4)
        self.act2 = act()

        self.pointwise_conv_2 = conv(out_channels // 4, out_channels, kernel_size = 1)
        self.norm3 = norm(out_channels)
        self.act3 = act()
    
    def forward(self, x):
        x = self.pointwise_conv_1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.depthwise_conv(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.pointwise_conv_2(x)
        x = self.norm3(x)
        x = self.act3(x)

        return x

class InvertedBottleneckBlock(nn.Module):

    # (conv 1x1 C -> 4*C) -> (conv d3x3 4*C -> 4*C) -> (conv 1x1 4*C -> C)
    # roughly equivalent to (conv 3x3 C -> C) -> (conv 3x3 C -> C)
    
    @staticmethod
    def fill_kwargs(config_dict):
        act_func_name, act_func_dict = config_dict['activations'].item()
        if act_func_name in model.activations.activation_funcs_dict:
            act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
            act_func_dict.fill_with_defaults(act_default_dict)
        elif act_default_dict is not None:
            utils.fill_dict(config_dict['activations'])

    def __init__(self, in_channels, out_channels, kernel_size = 3, activations = 'relu', **kwargs):

        super().__init__()
        
        def conv(in_channels, out_channels, kernel_size, groups = 1):
            return nn.Conv2d(in_channels, out_channels, kernel_size, padding = 'same', groups = groups)
        
        def norm(num_features):
            return nn.BatchNorm2d(num_features)
        
        def act():
            act_name = activations.key() if isinstance(activations, utils.config_dict.ConfigDict) else activations
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activations,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activations, convert_to_kwargs = True)

        self.pointwise_conv_1 = conv(in_channels, out_channels * 4, kernel_size = 1)
        self.norm1 = norm(out_channels * 4)
        self.act1 = act()

        self.depthwise_conv = conv(out_channels * 4, out_channels * 4, kernel_size = kernel_size, groups = out_channels * 4)
        self.norm2 = norm(out_channels * 4)
        self.act2 = act()

        self.pointwise_conv_2 = conv(out_channels * 4, out_channels, kernel_size = 1)
        self.norm3 = norm(out_channels)
        self.act3 = act()
    
    def forward(self, x):
        x = self.pointwise_conv_1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.depthwise_conv(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.pointwise_conv_2(x)
        x = self.norm3(x)
        x = self.act3(x)

        return x

class ConvNeXtBlock(nn.Module):
    
    @staticmethod
    def fill_kwargs(config_dict):
        act_func_name, act_func_dict = config_dict['activations'].item()
        if act_func_name in model.activations.activation_funcs_dict:
            act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
            act_func_dict.fill_with_defaults(act_default_dict)
        elif act_default_dict is not None:
            utils.fill_dict(config_dict['activations'])
        
        utils.fill_dict(config_dict['normalisation'])

    def __init__(self, in_channels, out_channels, kernel_size = 3, activations = 'gelu',
                 normalisation = 'segmentation.models.blocks.ChannelwiseLayerNorm',
                 reduce_number_of_activations = True,
                 reduce_number_of_norm_layers = True,
                 channel_change_index = 1, **kwargs):
        """
        Arguments:
            `kernel_size`: kernel size of the depthwise convolution layer
            `normalisation`: config dict of the normalisation method to be used; should accept one positional argument that is the number of channels
            `reduce_number_of_activations`: if set to `True`, the block will have only one activation function, after the second layer; if set to `False`, all layers will be proceeded by an activation function
            `reduce_number_of_norm_layers`: if `True`, only the output of the first layer will be normalised; if `False`, normalisation will be added after every convolution
            `channel_change_index`: index of the convolutional layer (starting from 1) where the channel size should be changed if the input and output channel sizes differ
        """

        super().__init__()

        def conv(in_channels, out_channels, kernel_size, groups = 1):
            return nn.Conv2d(in_channels, out_channels, kernel_size, padding = 'same', groups = groups)
        
        def norm(num_features):
            return utils.create_object_from_dict(normalisation, None, None, model.Model, False, num_features)
        
        def act():
            act_name = activations.key() if isinstance(activations, utils.config_dict.ConfigDict) else activations
            if act_name in model.activations.activation_funcs_dict:
                return utils.config_dict.initialise_object_from_dict(
                                                        config_dict = activations,
                                                        classes_dict = model.activations.activation_funcs_dict
                                                    )
            else:
                return utils.create_object_from_dict(activations, convert_to_kwargs = True)

        if channel_change_index not in (1, 2, 3):
            raise ValueError(f'Argument `channel_change_index` should be between 1 and 3, got {channel_change_index}.')
        channels = (in_channels,) * (channel_change_index - 1) + (out_channels,) * (3 - channel_change_index)
        block1 = [('conv', conv(in_channels, channels[0], kernel_size, groups = min(in_channels, channels[0]))), ('norm', norm(channels[0]))]
        if not reduce_number_of_activations:
            block1.append(('activation', act()))
        self.block1 = nn.Sequential(OrderedDict(block1))

        block2 = [('conv', conv(channels[0], 4 * channels[1], 1))]
        if not reduce_number_of_norm_layers:
            block2.append(('norm', norm(4 * channels[1])))
        block2.append(('activation', act()))
        self.block2 = nn.Sequential(OrderedDict(block2))

        block3 = [('conv', conv(4 * channels[1], out_channels, 1))]
        if not reduce_number_of_norm_layers:
            block3.append(('norm', norm(out_channels)))
        if not reduce_number_of_activations:
            block3.append(('activation', act()))
        self.block3 = nn.Sequential(OrderedDict(block3))

    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class LayerScale(nn.Module):

    def __init__(self, n_channels, init_value = 1e-6, *args, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(n_channels, 1, 1))
    
    def forward(self, x, *args, **kwargs):
        return self.scale * x