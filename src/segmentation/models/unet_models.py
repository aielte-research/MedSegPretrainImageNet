from typing import Literal, Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn
import numpy as np

import utils
import model
from segmentation.models import blocks

from exception_handling import handle_exception


def drop_path(x, drop_prob: float = 0., training: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# Function that calculates the drop probabilities for stochastic depth
def stoch_depth_calc(depth, width, stoch_depth_type = 'swin_unet', min_prob = 0, max_prob = 0.5):
    if stoch_depth_type == 'swin_unet':
        down_unit_number = (depth + 1) * width
        probs = np.linspace(min_prob, max_prob, down_unit_number)

        down_probs_list = [probs[i*width : (i+1)*width] for i in range(depth)]
        bottom_probs = probs[(depth)*width : (depth+1)*width]
        up_probs_list = [probs[-((i+2)*width): -((i+1)*width)] for i in range(depth)]

    return down_probs_list, bottom_probs, up_probs_list



class UNet_encoder(nn.Module):
    '''
    A general U-Net encoder module, with configurable features.
    Arguments:
        basic_block, stem, downsampling: block dictionaries
        depth: number of downsampling in the model
        width: number of blocks in a level
        channels: list of channel numbers after the end of levels. Defaults to powers of 2 starting with 64
        residual_connections: can add different types of residual connections around basic blocks
        change_channel_in_block: if true, then number of channels will be changed inside the block, otherwise it will happen during downsampling
        trainable_downsampling : whether the downsampling block is trainable
        stochastic_depth_rate : max_prob for stoch_depth_calc
        layer_scaling: layer scaling parameter
        init_scheme: initialization scheme
    '''
    
    
    @staticmethod
    def fill_kwargs(config_dict):
        basic_block_dict = config_dict['basic_block']
        downsampling_dict = config_dict['downsampling']
        stem_dict = config_dict['stem']
        for submodel_dict in (basic_block_dict, downsampling_dict, stem_dict):
            utils.fill_dict(submodel_dict)

    def __init__(self,
                 in_channel_size : int = 3,
                 basic_block : utils.ConfigDict = 'segmentation.models.blocks.ConvBlock',
                 stem : utils.ConfigDict = {'torch.nn.Conv2d': {'kernel_size': 3, 'padding': 'same'}},
                 downsampling : utils.ConfigDict = {'torch.nn.MaxPool2d': {'kernel_size': 2}},
                 depth : int = 4,
                 width : int = 1,
                 channels : Union[Tuple[int], Literal['default']] = 'default',
                 residual_connections : Literal[False, 'identity', 'convolution'] = False,
                 change_channel_in_block : bool = True,
                 trainable_downsampling : bool = False,
                 stochastic_depth_rate : float = 0.0,
                 layer_scaling : Union[Literal[False], float] = False,
                 init_scheme : Optional[utils.ConfigDict] = None,
                 *args, **kwargs):
        
        super(UNet_encoder, self).__init__()
        
        self.depth = depth
        self.width = width
        
        # Creating list of channel numbers from predefined list or default. 
        if isinstance(channels, utils.config_dict.ConfigDict):
            channels = channels.key()
        if channels not in (None, 'default'):
            self.channels = channels
        else:
            self.channels = [64*(2**i) for i in range(depth+1)]
        # This is needed because by default the first level starts and ends with the same channel number (stem makes it possible)
        if len(self.channels) < depth + 2:
            self.channels = [self.channels[0], *self.channels]


        # Defining needed attributes and functions:
        if stem is not None:
            self.first_block = utils.create_object_from_dict(stem,
                                                             wrapper_class = model.Model,
                                                             in_channels = in_channel_size,
                                                             out_channels= self.channels[0])
        else:
            self.first_block = nn.Identity()

        block_constr = utils.get_class_constr(basic_block.key())
        self.integrated_downsample = downsampling == None
        def make_basic_block(in_channels, out_channels, downsample_in_block = self.integrated_downsample, position = 1, stochastic_depth_rate = 0):
            kwargs = dict(in_channels = in_channels, out_channels = out_channels)
            if utils.accepts_kwarg(block_constr, 'downsample_in_block'):
                kwargs['downsample_in_block'] = downsample_in_block
            if utils.accepts_kwarg(block_constr, 'position'):
                kwargs['position'] = position
            if utils.accepts_kwarg(block_constr, 'stochastic_depth_rate'):
                kwargs['stochastic_depth_rate'] = stochastic_depth_rate
            return utils.create_object_from_dict(basic_block, wrapper_class = model.Model, **kwargs)

        if not self.integrated_downsample:
            def make_downsampling_block(in_channels, out_channels):
                if trainable_downsampling:
                    return utils.create_object_from_dict(downsampling, wrapper_class = model.Model,
                                                         in_channels = in_channels, out_channels = out_channels)
                else:
                    return utils.create_object_from_dict(downsampling, wrapper_class = model.Model)

        self.res_con = bool(residual_connections)
        if self.res_con:
            def make_shortcut(in_channels, out_channels, downsampling = False):
                return blocks.ResConnection(type_dict = residual_connections,
                                            in_channels = in_channels,
                                            out_channels = out_channels,
                                            downsampling = downsampling)

        if stochastic_depth_rate:
            self.stochastic_depth = True
            self.down_sd_probs_list, self.bottom_sd_probs, self.up_sd_probs_list = stoch_depth_calc(
                                                                            max_prob = stochastic_depth_rate,
                                                                            depth = depth,
                                                                            width = width
                                                                            )
        else:
            self.stochastic_depth = False
            self.down_sd_probs_list, self.bottom_sd_probs, self.up_sd_probs_list = stoch_depth_calc(
                                                                            max_prob = 0,
                                                                            depth = depth,
                                                                            width = width
                                                                            )

        self.layer_scale = (layer_scaling != False)

        # Creating the blocks and layers:
        down_layers = []

        for i in range(depth):
            # Creation of the first basic block and possibly other blocks for each level
            unit = {'conv0': make_basic_block(in_channels = self.channels[i + (not change_channel_in_block)], out_channels = self.channels[i+1], 
                                            downsample_in_block = self.integrated_downsample if width == 1 else False,
                                            position = 0, stochastic_depth_rate = self.down_sd_probs_list[i][0])}
            if self.res_con:
                unit['shortcut0'] = make_shortcut(in_channels= self.channels[i + (not change_channel_in_block)], out_channels= self.channels[i+1], 
                                                downsampling = self.integrated_downsample if width == 1 else False)
            if self.layer_scale:
                unit['layer_scale0'] = blocks.LayerScale(self.channels[i + 1], layer_scaling)   
            for j in range(1, width):
                # If width > 1 add additional basic blocks and other blocks
                unit[f'conv{j}'] =  make_basic_block(in_channels = self.channels[i+1], out_channels = self.channels[i+1], 
                                                    downsample_in_block = self.integrated_downsample if j == width-1 else False,
                                                    position = j, stochastic_depth_rate = self.down_sd_probs_list[i][j])
                if self.res_con:
                    unit[f'shortcut{j}'] = make_shortcut(in_channels= self.channels[i+1], out_channels= self.channels[i+1], 
                                                        downsampling = self.integrated_downsample if j == width-1 else False)
                if self.layer_scale:
                    unit[f'layer_scale{j}'] = blocks.LayerScale(self.channels[i + 1], layer_scaling)  
            # Create downsampling layer                      
            if not self.integrated_downsample:
                unit['downsampl'] = make_downsampling_block(in_channels = self.channels[i + 1], out_channels = self.channels[i + 1 + (not change_channel_in_block)])
            down_layers.append(nn.ModuleDict(unit))
        self.down_layers = nn.ModuleList(down_layers)
        
        # Same things for the last (bottom) level
        bottom_block = {'conv0': make_basic_block(in_channels = self.channels[-2 + (not change_channel_in_block)], out_channels = self.channels[-1], 
                                            downsample_in_block = False,
                                            position = 0, stochastic_depth_rate = self.bottom_sd_probs[0])}
        if self.res_con:
            bottom_block['shortcut0'] = make_shortcut(in_channels= self.channels[-2 + (not change_channel_in_block)], out_channels= self.channels[-1], downsampling = False)
        if self.layer_scale:
            bottom_block['layer_scale0'] = blocks.LayerScale(self.channels[-1], layer_scaling)
        for j in range(1, width):
            bottom_block[f'conv{j}'] =  make_basic_block(in_channels = self.channels[-1], out_channels = self.channels[-1], 
                                                downsample_in_block = False,
                                                position = j, stochastic_depth_rate = self.bottom_sd_probs[j])
            if self.res_con:
                bottom_block[f'shortcut{j}'] = make_shortcut(in_channels= self.channels[-1], out_channels= self.channels[-1], 
                                                    downsampling = False)
            if self.layer_scale:
                bottom_block[f'layer_scale{j}'] = blocks.LayerScale(self.channels[-1], layer_scaling)
        self.bottom_block = nn.ModuleDict(bottom_block)
        
        
    def forward(self, x, return_skip_vals=False):
        skip_values = []

        x = self.first_block(x)

        for i, unit in enumerate(self.down_layers):
            for j in range(self.width):
                x1 = unit[f'conv{j}'](x)
                if self.layer_scale:
                    x1 = unit[f'layer_scale{j}'](x1)
                if self.res_con:
                    if self.stochastic_depth:
                        x1 = drop_path(x1, drop_prob = self.down_sd_probs_list[i][j], training=self.training)
                    x2 = unit[f'shortcut{j}'](x)
                    x = x1 + x2
                else:
                    x = x1
            skip_values.append(x)
            if not self.integrated_downsample:
                x = unit['downsampl'](x)
        
        for j in range(self.width):
            x1 = self.bottom_block[f'conv{j}'](x)
            if self.layer_scale:
                x1 = self.bottom_block[f'layer_scale{j}'](x1)
            if self.res_con:            
                if self.stochastic_depth:
                    x1 = drop_path(x1, drop_prob=self.bottom_sd_probs[j], training = self.training)
                x2 = self.bottom_block[f'shortcut{j}'](x)
                x = x1 + x2
            else:
                x = x1

        if return_skip_vals:
            return x, skip_values
        else:
            return x



class UNet_decoder(nn.Module):
    '''
    A general U-Net decoder module, with configurable features.
    Arguments:
        basic_block, updownsampling_block, mixing_block, final_block: block dictionaries
        depth: number of upsampling in decoder
        channels: List of channel numbers
                Starts with the output channel number of the encoder, after that the channels at the end of the decoder levels
        residual_connections: can add different types of residual connections around basic blocks
        skip_con_channels_list : list of number of channels of the data tensors comming through the skip connections
                                the length of this list determines the number of skip connections
        stochastic_depth_rate : max_prob for stoch_depth_calc
        layer_scaling : layer scaling parameter
    '''

    def __init__(self, basic_block = None, upsampling_block = None,  mixing_block = None, init_scheme = None, residual_connections = False,
                 stochastic_depth_rate = 0.0, output_ch=1, depth = 4, width = 1, channels = None,
                 final_block = None, layer_scaling = False, skip_con_channels_list = None, *args, **kwargs):
        super(UNet_decoder, self).__init__()
        
        self.depth = depth
        self.width = width
        # Channels have to be given correctly either by default or in the config_dict.
        self.channels = channels     


        # Defining needed attributes and functions:
        block_constr = utils.get_class_constr(basic_block.key())
        def make_basic_block(in_channels, out_channels, position = 1, stochastic_depth_rate = 0):
            kwargs = dict(in_channels = in_channels, out_channels = out_channels)
            if utils.accepts_kwarg(block_constr, 'position'):
                kwargs['position'] = position
            if utils.accepts_kwarg(block_constr, 'stochastic_depth_rate'):
                kwargs['stochastic_depth_rate'] = stochastic_depth_rate
            return utils.create_object_from_dict(basic_block, wrapper_class = model.Model, **kwargs)
        

        self.res_con = bool(residual_connections)
        if self.res_con:
            def make_shortcut(in_channels, out_channels, downsampling = False):
                return blocks.ResConnection(type_dict = residual_connections,
                                            in_channels = in_channels,
                                            out_channels = out_channels,
                                            downsampling = downsampling)

        if stochastic_depth_rate:
            self.stochastic_depth = True
            self.down_sd_probs_list, self.bottom_sd_probs, self.up_sd_probs_list = stoch_depth_calc(
                                                                            max_prob = stochastic_depth_rate,
                                                                            depth = depth,
                                                                            width = width
                                                                            )
        else:
            self.stochastic_depth = False
            self.down_sd_probs_list, self.bottom_sd_probs, self.up_sd_probs_list = stoch_depth_calc(
                                                                            max_prob = 0,
                                                                            depth = depth,
                                                                            width = width
                                                                            )

        self.layer_scale = (layer_scaling != False)

        self.skip_con_nr = len(skip_con_channels_list)

        # A ratio in [0,1] that determines the rate of decrease in channels for the upsampling blocks.
        self.upsample_channel_decrease_ratio = upsampling_block[upsampling_block.key()].get('channel_decrease_ratio', 0.5)
        
        def make_upsampling_block(in_channels, out_channels):
            return utils.create_object_from_dict(upsampling_block, in_channels = in_channels, out_channels = out_channels,
                                                 wrapper_class = model.Model)

        # The mixing block defines the way how the data from the previous level and 
        # the data coming from the skip connection are aggregated. (Like concatenation, attention block, etc.)
        if mixing_block == 'concatenate':
            def make_mixing_block(**kwargs):
                return blocks.ConcatBlock(**kwargs) 
        else:
            def make_mixing_block(**kwargs):
                return utils.create_object_from_dict(mixing_block, convert_to_kwargs= True, **kwargs)

        # Creating the blocks and layers:
        up_layers = []

        for i in range(depth):
            # Creation of the first upsampling and basic block and possibly other blocks for each level
            ups_out_channels = int(self.channels[i] * self.upsample_channel_decrease_ratio)
            unit = {'upsampl': make_upsampling_block(in_channels = self.channels[i], out_channels = ups_out_channels)}
            # While we have data from the skip connections we use it in the mixing block
            # TODO: option to define which levels we should use the skip connection data
            if i < self.skip_con_nr:
                unit['mixing'] = make_mixing_block(x_channels = self.channels[i], 
                                                    x_up_channels = ups_out_channels, 
                                                    skip_channels = skip_con_channels_list[i],
                                                    level_out_channels = self.channels[i+1])
                mixing_out_ch_calc = unit['mixing'].get_out_ch
            else:
                mixing_out_ch_calc = lambda **kwargs: kwargs['x_up_channels']
            # Calculating the output channel number of the mixing block
            mixing_out_ch = mixing_out_ch_calc(x_channels=self.channels[i], 
                                               x_up_channels = ups_out_channels, 
                                               skip_channels = 0 if i >= self.skip_con_nr else skip_con_channels_list[i],
                                               level_out_channels = self.channels[i+1])
            unit['conv0'] = make_basic_block(in_channels = mixing_out_ch, out_channels = self.channels[i+1],
                                             position = 0, stochastic_depth_rate = self.up_sd_probs_list[i][0])
            if self.res_con:
                unit['shortcut0'] = make_shortcut(in_channels = mixing_out_ch, out_channels = self.channels[i+1])
            if self.layer_scale:
                unit['layer_scale0'] = blocks.LayerScale(self.channels[i + 1], layer_scaling)
            for j in range(1, width):
                # If width > 1 add additional basic blocks and other blocks
                unit[f'conv{j}'] =  make_basic_block(in_channels = self.channels[i+1], out_channels = self.channels[i+1], 
                                                     position = j, stochastic_depth_rate = self.up_sd_probs_list[i][j])
                if self.res_con:
                    unit[f'shortcut{j}'] = make_shortcut(in_channels= self.channels[i+1], out_channels= self.channels[i+1], 
                                                        downsampling = False)
                if self.layer_scale:
                    unit[f'layer_scale{j}'] = blocks.LayerScale(self.channels[i+1], layer_scaling)
            up_layers.append(nn.ModuleDict(unit))
        self.up_layers = nn.ModuleList(up_layers)

        # Creating the final block
        if final_block is not None:
            self.final_block = utils.create_object_from_dict(final_block, wrapper_class = model.Model,
                                                            in_channels = self.channels[-1], out_channels = output_ch)
        else:
            self.final_block = nn.Identity()
        
    def forward(self, x, skip_values):

        for i, unit in enumerate(self.up_layers):
            x_up = unit['upsampl'](x)
            if i < self.skip_con_nr:
                skip_val = skip_values.pop()
                x = unit['mixing'](x = x, x_up = x_up, skip_val = skip_val)
            else:
                x = x_up
            for j in range(self.width):
                x1 = unit[f'conv{j}'](x)
                if self.layer_scale:
                    x1 = unit[f'layer_scale{j}'](x1)
                if self.res_con:    
                    if self.stochastic_depth:
                        x1 = drop_path(x1, drop_prob = self.up_sd_probs_list[i][j], training = self.training)
                    x2 = unit[f'shortcut{j}'](x)
                    x = x1 + x2
                else:
                    x = x1
        
        output = self.final_block(x)

        return output


class UNet(nn.Module):
    '''
    Arguments:
        downsampling_block, up_block, upsampling_block, basic_block, mixing_block,
        preproc_block, final_block: block dictionaries
        depth: number of downsampling in the model (channels list can overwrite this)
        width: number of basic blocks on a level
        channels, encoder_channels_decoder_channels: list of channel numbers after the end of levels. Defaults to powers of 2 starting with 64.
        skip_con_channels_list : list of number of channels of the data tensors comming through the skip connections
                                the length of this list determines the number of skip connections
        residual_connections: can add different types of residual connections around basic blocks
        stochastic_depth_rate : max_prob for stoch_depth_calc
        final_activation: final activation dictionary
        layer_scaling: layer scaling parameter
        change_channel_in_block: if true, then number of channels will be changed inside the block, otherwise it will happen during downsampling
        trainable_downsampling : whether the downsampling block is trainable
        img_ch, output_ch: number of channels of the input and output
        encoder: model dictionary for an encoder different to the standard UNet encoder
    '''

    PARAMS = {
        'architecture/in channel size': {
            'argument name': 'img_ch',
            'default': 3
            },
        'architecture/out channel size': {
            'argument name': 'output_ch',
            'default': 1
            },
        'architecture/depth': {     # number of downsamplings
            'argument name': 'depth',
            'default': 4
            },
        'architecture/width': {     # number of basic blocks on a level
            'argument name': 'width',
            'default': 1
            },
        'architecture/basic block': {
            'argument name': 'basic_block',
            'default': 'segmentation.models.blocks.ConvBlock'
            },
        'architecture/mixing block': {
            'argument name': 'mixing_block',
            'default': 'concatenate'
            },
        'architecture/stem': {
            'argument name': 'preproc_block',
            'default': {'torch.nn.Conv2d': {'kernel_size': 3, 'padding': 'same'}}
            },
        'architecture/final_block': {
            'argument name': 'final_block',
            'default': {'torch.nn.Conv2d': {'kernel_size': 1}}
            },
        'architecture/upsampling': {
            'argument name': 'upsampling_block',
            'default': 'segmentation.models.blocks.UpConvBlock'
            },
        'architecture/downsampling': {
            'argument name': 'downsampling_block',
            'default': {'torch.nn.MaxPool2d': {'kernel_size': 2}}
            },
        'architecture/channels': {
            'argument name': 'channels',
            'default': 'default'
            },    
        'architecture/encoder_channels': {
            'argument name': 'encoder_channels',
            'default': None
            },
        'architecture/decoder_channels': {
            'argument name': 'decoder_channels',
            'default': None
            },
        'architecture/skip_con_channels': {
            'argument name': 'skip_con_channels',
            'default': None
            },
        'architecture/residual_connections': {
            'default': False,       # one of False, convolution, identity
            'argument name': 'residual_connections'
            },
        'architecture/stochastic depth rate': {
            'default': 0,
            'argument name': 'stochastic_depth_rate'
            },
        'architecture/activation function/final': {
            'default': 'sigmoid',
            'argument name': 'final_activation'
            },
        'architecture/layer_scaling': {
            'default': False,
            'argument name': 'layer_scaling'
            },
        'architecture/change_channels_in_block': {
            'argument name': 'change_channels_in_block',
            'default': True
            },
        'architecture/trainable_downsampling': {
            'argument name': 'trainable_downsampling',
            'default': False
            },
        'architecture/encoder': {'argument name': 'encoder', 'default': None},

        'weight_init': {'torch.nn.init.kaiming_normal_': {'nonlinearity': 'relu'}}
    }

    @staticmethod
    def fill_kwargs(config_dict : utils.ConfigDict):
        # TODO: comment what happens here..
        basic_block_dict = config_dict['architecture/basic block']
        upsampling_dict = config_dict['architecture/upsampling']
        downsampling_dict = config_dict['architecture/downsampling']
        stem_dict = config_dict['architecture/stem']
        final_block_dict = config_dict['architecture/final_block']
        for submodel_dict in (basic_block_dict, upsampling_dict, downsampling_dict, stem_dict, final_block_dict):
            utils.fill_dict(submodel_dict)

        mixing_dict = config_dict['architecture/mixing block']
        if mixing_dict.key() != 'concatenate':
            utils.fill_dict(mixing_dict)
        
        act_func_name, act_func_dict = config_dict['architecture/activation function/final'].item()
        if act_func_name in model.activations.activation_funcs_dict:
            act_default_dict = model.activations.activation_funcs_dict[act_func_name]['arguments']
            act_func_dict.fill_with_defaults(act_default_dict)
        elif act_default_dict is not None:
            utils.fill_dict(config_dict['architecture/activation function/final'])
        
        res_con_dict = config_dict['architecture/residual_connections']
        if res_con_dict:
            res_con_keys = list(res_con_dict.keys())
            if len(res_con_keys) == 1 and res_con_dict.key() not in ('up', 'down'):
                res_cons = utils.config_dict.ConfigDict({'down': res_con_dict.copy(), 'up': res_con_dict.copy()})
            else:
                res_cons = res_con_dict
            
            down_res_con = res_cons.get_or_update('down', 'identity')
            up_res_con = res_cons.get_or_update('up', 'identity')
            res_cons.expand()
            
            if down_res_con and (down_res_con == 'identity' or down_res_con.key() == 'identity') and config_dict['architecture/change_channels_in_block']:
                res_cons.get_or_update('down/identity/expand_method', 'fill_with_zeros')
            
            if up_res_con and (up_res_con == 'identity' or up_res_con.key() == 'identity'):
                res_cons.get_or_update('up/identity/contract_method', 'add')
            
            for kw in ('down', 'up'):
                res_con = res_cons.get(kw)
                if res_con and res_con.key() in ('conv', 'convolution'):
                    res_con.value().fill_with_defaults(blocks.ResConnection.CONV_PARAMS)         
                    
            config_dict['architecture/residual_connections'] = res_cons
        
        if config_dict.get('architecture/encoder', None) is None:
            config_dict.pop('architecture/encoder', None)
        else:
            for key in ('in_channel_size', 'stem', 'downsampling', 'trainable_downsampling', 'change_channels_in_block'):
                config_dict.pop('architecture/' + key, None)
            if res_con_dict:
                config_dict.pop('architecture/residual_connections/down', None)
            utils.fill_dict(config_dict['architecture/encoder'])
    
    def init_weights(self, config_dict, *args, **kwargs):
        # TODO: this only works if the encoder was trained as the first layer of a FeedForwardNetwork
        key = 'weight_initialisation' if 'weight_initialisation' in config_dict else 'weight_init'
        key = key + '/encoder'
        
        if key not in config_dict:
            return
        
        encoder_weights = config_dict[key].trim().get('weights', False)
        strict = config_dict[key].get('strict', True)
        
        if not encoder_weights:
            return
        
        try:
            state_dict = torch.load(encoder_weights)
            encoder_state_dict = {k[9:]: v for k, v in state_dict.items() if k[:8] == 'layers.0'}
            missing, _ = self.encoder.load_state_dict(encoder_state_dict, strict = False)
            for key in missing:
                shortened_key = key.replace('.model.', '.')
                if shortened_key in encoder_state_dict:
                    encoder_state_dict[key] = encoder_state_dict.pop(shortened_key)
            missing, unexpected  = self.encoder.load_state_dict(encoder_state_dict, strict = strict)
            if not strict:
                if len(missing) > 0:
                    keys = ', '.join(missing)
                    warnings.warn(f'Missing keys in state dict {encoder_weights}: {keys}. These submodules will remain randomly initialised.')
                if len(unexpected) > 0:
                    keys = ', '.join(unexpected)
                    warnings.warn(f'Unexpected keys in state dict {encoder_weights}: {keys}. These keys were ignored when initialising the encoder weights.')
            print('Successfully initialised encoder with pretrained weights.')
        except Exception as e:
            msg = f'An exception occured while trying to load the weights of {encoder_weights}. Leaving the encoder weights as randomly initialised.'
            handle_exception(e, msg)


    def __init__(self, basic_block = None, mixing_block = None, upsampling_block = None, init_scheme = None,
                 downsampling_block = None, residual_connections = False, stochastic_depth_rate = 0.0,
                 img_ch=3, output_ch=1, final_activation=None,  depth = 4, width = 1, channels = None, encoder_channels = None, decoder_channels = None,
                 preproc_block = None, final_block = None, layer_scaling = False,
                 change_channels_in_block = True, trainable_downsampling = False, encoder = None, skip_con_channels = None, *args, **kwargs):
        super(UNet, self).__init__()
        
        # Defining final activation layer:
        final_act_name = final_activation if not isinstance(final_activation, utils.config_dict.ConfigDict) else final_activation.key()
        if final_activation is None:
            self.final_act = None
        elif final_act_name in model.activations.activation_funcs_dict:
            self.final_act = utils.config_dict.initialise_object_from_dict(
                                                    config_dict = final_activation,
                                                    classes_dict = model.activations.activation_funcs_dict
                                                    )
        else:
            self.final_act = utils.create_object_from_dict(final_activation, convert_to_kwargs = True)

        # Getting default depth and width
        self.depth = depth
        self.width = width

        # If encoder and decoder channels are not defined, channels will be used:
        if isinstance(channels, utils.config_dict.ConfigDict):
            channels = channels.key()
        if channels not in (None, 'default'):
            self.channels = channels
        else:
            self.channels = [64*(2**i) for i in range(depth+1)]

        # Getting encoder channels (encoder['channel_sizes'] > encoder_channels > channels)
        if encoder is not None:
            self.encoder_channels = encoder[encoder.key()].get('channel_sizes')
        else:
            self.encoder_channels = self.channels if encoder_channels is None else encoder_channels
        # Getting decoder channels (decoder_channels > reversed encoder_channels)
        self.decoder_channels = self.channels[::-1] if decoder_channels is None else [self.encoder_channels[-1], *decoder_channels]

        # The depth of the encoder and decoder parts (might be different)
        self.encoder_depth = len(self.encoder_channels)-1
        self.decoder_depth = len(self.decoder_channels)-1

        # Get skip connection channels from encoder channels if not specified otherwise
        skip_con_channels_list =  skip_con_channels or self.encoder_channels[:-1][::-1]


        # Setting up other attributes
        self.integrated_downsample = downsampling_block == None
        
        if residual_connections:
            down_res_cons = residual_connections['down']
            up_res_cons = residual_connections['up']
        else:
            down_res_cons, up_res_cons = False, False

        # Create encoder from config dict if specified, else create encoder from UNet_encoder
        if encoder is not None:
            encoder_model = utils.create_object_from_dict(encoder, wrapper_class = model.Model)
            self.encoder = getattr(encoder_model, 'model', encoder_model)
        else:
            self.encoder = UNet_encoder(basic_block = basic_block,
                                        init_scheme = init_scheme,
                                        downsampling = downsampling_block,
                                        trainable_downsampling = trainable_downsampling,
                                        residual_connections = down_res_cons,
                                        stochastic_depth_rate = stochastic_depth_rate,
                                        in_channel_size = img_ch,
                                        depth = self.encoder_depth,
                                        width = width,
                                        channels = self.encoder_channels,
                                        change_channel_in_block = change_channels_in_block,
                                        stem = preproc_block,
                                        layer_scaling = layer_scaling)
        # Create UNet_decoder
        self.decoder = UNet_decoder(basic_block = basic_block,
                                    mixing_block = mixing_block,
                                    init_scheme = init_scheme,
                                    upsampling_block = upsampling_block,
                                    residual_connections = up_res_cons,
                                    stochastic_depth_rate = stochastic_depth_rate,
                                    output_ch = output_ch,
                                    depth = self.decoder_depth,
                                    width = width,
                                    skip_con_channels_list = skip_con_channels_list,
                                    channels = self.decoder_channels,
                                    final_block = final_block,
                                    layer_scaling = layer_scaling)


    def forward(self, x):
        x, skip_vals = self.encoder(x, return_skip_vals=True)
        output = self.decoder(x, skip_vals)

        if self.final_act is not None:
            output = self.final_act(output)

        return output


