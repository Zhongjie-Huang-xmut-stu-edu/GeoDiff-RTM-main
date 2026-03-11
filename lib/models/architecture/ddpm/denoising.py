from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule

from mmgen.models.architectures.ddpm.modules import TimeEmbedding, EmbedSequential
from mmgen.models.architectures.ddpm.denoising import DenoisingUnet
from mmgen.models.builder import MODULES, build_module



class SpatialOperation(nn.Module):
    """空间注意力操作模块"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class AdditiveTokenMixer(nn.Module):
    
    def __init__(self, dim, attn_bias=False, proj_drop=0.):
        super().__init__()
        
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        
        
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        
        
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        q, k, v = self.qkv(x).chunk(3, dim=1)
        
        
        q = self.oper_q(q)
        k = self.oper_k(k)
        
        
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        
        
        return x + out 



@MODULES.register_module()
class DenoisingUnetMod(DenoisingUnet):

    def __init__(self,
                 image_size,
                 in_channels=3,
                 concat_cond_channels=0,
                 base_channels=128,
                 resblocks_per_downsample=3,
                 num_timesteps=1000,
                 use_rescale_timesteps=True,
                 dropout=0,
                 embedding_channels=-1,
                 num_classes=0,
                 channels_cfg=None,
                 groups=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1,
                 use_scale_shift_norm=False,
                 num_heads=4,
                 time_embedding_mode='sin',
                 time_embedding_cfg=None,
                 resblock_cfg=dict(type='DenoisingResBlockMod'),
                 attention_cfg=dict(type='MultiHeadAttentionMod'),
                 downsample_conv=True,
                 upsample_conv=True,
                 downsample_cfg=dict(type='DenoisingDownsampleMod'),
                 upsample_cfg=dict(type='DenoisingUpsampleMod'),
                 attention_res=[16, 8],
                 pretrained=None,
                 
                 use_catm=False,
                 catm_positions=['down', 'mid', 'up'],
                 catm_start_level=1):
        
        
        self.use_catm = use_catm
        self.catm_positions = catm_positions
        self.catm_start_level = catm_start_level
        
        super(DenoisingUnet, self).__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.use_rescale_timesteps = use_rescale_timesteps

        out_channels = in_channels
        self.out_channels = out_channels
        self.concat_cond_channels = concat_cond_channels

        # check type of image_size
        if isinstance(image_size, list) or isinstance(image_size, tuple):
            assert len(image_size) == 2, 'The length of `image_size` should be 2.'
        elif isinstance(image_size, int):
            image_size = [image_size, image_size]
        else:
            raise TypeError('Only support `int` and `list[int]` for `image_size`.')
        self.image_size = image_size

        if isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channels_cfg`, '
                             f'receive {type(channels_cfg)}')

        embedding_channels = base_channels * 4 \
            if embedding_channels == -1 else embedding_channels
        self.time_embedding = TimeEmbedding(
            base_channels,
            embedding_channels=embedding_channels,
            embedding_mode=time_embedding_mode,
            embedding_cfg=time_embedding_cfg,
            act_cfg=act_cfg)

        if self.num_classes != 0:
            self.label_embedding = nn.Embedding(self.num_classes,
                                                embedding_channels)

        self.resblock_cfg = deepcopy(resblock_cfg)
        self.resblock_cfg.setdefault('dropout', dropout)
        self.resblock_cfg.setdefault('groups', groups)
        self.resblock_cfg.setdefault('norm_cfg', norm_cfg)
        self.resblock_cfg.setdefault('act_cfg', act_cfg)
        self.resblock_cfg.setdefault('embedding_channels', embedding_channels)
        self.resblock_cfg.setdefault('use_scale_shift_norm',
                                     use_scale_shift_norm)
        self.resblock_cfg.setdefault('shortcut_kernel_size',
                                     shortcut_kernel_size)

        # get scales of ResBlock to apply attention
        attention_scale = [min(image_size) // int(res) for res in attention_res]
        self.attention_cfg = deepcopy(attention_cfg)
        self.attention_cfg.setdefault('num_heads', num_heads)
        self.attention_cfg.setdefault('groups', groups)
        self.attention_cfg.setdefault('norm_cfg', norm_cfg)

        self.downsample_cfg = deepcopy(downsample_cfg)
        self.downsample_cfg.setdefault('groups', groups)
        self.downsample_cfg.setdefault('with_conv', downsample_conv)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.upsample_cfg.setdefault('groups', groups)
        self.upsample_cfg.setdefault('with_conv', upsample_conv)

        # init the channel scale factor
        scale = 1
        self.in_blocks = nn.ModuleList([
            EmbedSequential(
                nn.Conv2d(in_channels + concat_cond_channels, base_channels, 3, 1, padding=1, groups=groups))
        ])
        self.in_channels_list = [base_channels]

        # construct the encoder part of Unet
        for level, factor in enumerate(self.channel_factor_list):
            in_channels_ = base_channels if level == 0 \
                else base_channels * self.channel_factor_list[level - 1]
            out_channels_ = base_channels * factor

            for _ in range(resblocks_per_downsample):
                layers = [
                    build_module(self.resblock_cfg, {
                        'in_channels': in_channels_,
                        'out_channels': out_channels_
                    })
                ]
                in_channels_ = out_channels_

                if scale in attention_scale:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))

                self.in_channels_list.append(in_channels_)
                self.in_blocks.append(EmbedSequential(*layers))

            if level != len(self.channel_factor_list) - 1:
                self.in_blocks.append(
                    EmbedSequential(
                        build_module(self.downsample_cfg,
                                     {'in_channels': in_channels_})))
                self.in_channels_list.append(in_channels_)
                scale *= 2

        # construct the bottom part of Unet
        self.mid_blocks = EmbedSequential(
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
            build_module(self.attention_cfg, {'in_channels': in_channels_}),
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
        )

        # construct the decoder part of Unet
        in_channels_list = deepcopy(self.in_channels_list)
        self.out_blocks = nn.ModuleList()
        for level, factor in enumerate(self.channel_factor_list[::-1]):
            for idx in range(resblocks_per_downsample + 1):
                layers = [
                    build_module(
                        self.resblock_cfg, {
                            'in_channels':
                            in_channels_ + in_channels_list.pop(),
                            'out_channels': base_channels * factor
                        })
                ]
                in_channels_ = base_channels * factor
                if scale in attention_scale:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))
                if (level != len(self.channel_factor_list) - 1
                        and idx == resblocks_per_downsample):
                    layers.append(
                        build_module(self.upsample_cfg,
                                     {'in_channels': in_channels_}))
                    scale //= 2
                self.out_blocks.append(EmbedSequential(*layers))

        self.out = ConvModule(
            in_channels=in_channels_,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

        
        if self.use_catm:
            self.setup_catm_modules()

        self.init_weights(pretrained)

    
    def setup_catm_modules(self):
        """设置CATM模块"""
        self.catm_modules = nn.ModuleDict()
        base_channels = self.in_channels_list[0]  
        
        
        if 'down' in self.catm_positions:
            for i in range(len(self.in_blocks)):
                if i >= self.catm_start_level and i < len(self.in_channels_list):
                    channels = self.in_channels_list[i]
                    catm_module = AdditiveTokenMixer(channels)
                    self.catm_modules[f'down_catm_{i}'] = catm_module
        
        
        if 'mid' in self.catm_positions:
            
            mid_channels = self.in_channels_list[-1]
            catm_module = AdditiveTokenMixer(mid_channels)
            self.catm_modules['mid_catm'] = catm_module
        
        
        if 'up' in self.catm_positions:
            
            for i in range(len(self.out_blocks)):
                if i >= self.catm_start_level:
                    
                    level = i // (len(self.out_blocks) // len(self.channel_factor_list))
                    if level < len(self.channel_factor_list):
                        factor = self.channel_factor_list[::-1][level]
                        channels = base_channels * factor
                        catm_module = AdditiveTokenMixer(channels)
                        self.catm_modules[f'up_catm_{i}'] = catm_module

    def forward(self, x_t, t, label=None, concat_cond=None, return_noise=False):
        if self.use_rescale_timesteps:
            t = t.float() * (1000.0 / self.num_timesteps)
        embedding = self.time_embedding(t)

        if label is not None:
            assert hasattr(self, 'label_embedding')
            embedding = self.label_embedding(label) + embedding

        h, hs = x_t, []
        if self.concat_cond_channels > 0:
            h = torch.cat([h, concat_cond], dim=1)
        
        
        for i, block in enumerate(self.in_blocks):
            h = block(h, embedding)
            
            
            catm_key = f'down_catm_{i}'
            if self.use_catm and catm_key in self.catm_modules:
                h = self.catm_modules[catm_key](h)
                
            hs.append(h)

        
        h = self.mid_blocks(h, embedding)
        
        
        if self.use_catm and 'mid_catm' in self.catm_modules:
            h = self.catm_modules['mid_catm'](h)

        
        for i, block in enumerate(self.out_blocks):
            h = block(torch.cat([h, hs.pop()], dim=1), embedding)
            
            
            catm_key = f'up_catm_{i}'
            if self.use_catm and catm_key in self.catm_modules:
                h = self.catm_modules[catm_key](h)
                
        outputs = self.out(h)

        return outputs