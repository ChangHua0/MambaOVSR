from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import torchvision.ops
from einops import rearrange, repeat
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from .util import cat_conv


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        self.init_weights(self.conv1)
        self.init_weights(self.conv2)

    @staticmethod
    def init_weights(conv):
        init.kaiming_normal_(conv.weight, a=0, mode='fan_in')
        conv.weight.data *= 0.1  # for residual block
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class DCN_sep(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_channels_features: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 bias: bool = True,
                 mask: bool = True):
        super(DCN_sep, self).__init__()

        self.dcn = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias)

        kernel_size_ = _pair(kernel_size)
        offset_channels = deformable_groups * kernel_size_[0] * kernel_size_[1]

        self.conv_offset = nn.Conv2d(in_channels_features, offset_channels * 2, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, bias=True)
        self.conv_mask = nn.Conv2d(in_channels_features, offset_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=True) if mask else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor, feature: torch.Tensor):
        offset = self.conv_offset(feature)
        mask = torch.sigmoid(self.conv_mask(feature)) if self.conv_mask else None

        return self.dcn(input, offset, mask)


class PCDLayer(nn.Module):
    """ Alignment module using Pyramid, Cascading and Deformable convolution"""
    def __init__(self, args, first_layer: bool):
        super(PCDLayer, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups

        self.offset_conv1 = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.dcnpack = DCN_sep(self.nf, self.nf, self.nf, 3, stride=1, padding=1, dilation=1,
                               deformable_groups=self.groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not first_layer:
            self.offset_conv2 = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)
            self.fea_conv = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)

    def forward(self, current_sources: Tuple[torch.Tensor, torch.Tensor],
                last_offset: torch.Tensor, last_feature: torch.Tensor):
        offset = self.lrelu(cat_conv(self.offset_conv1, current_sources))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = offset.shape
            last_offset = last_offset[..., :h, :w] 
            offset = self.lrelu(cat_conv(self.offset_conv2, (offset, last_offset * 2))) 
        offset = self.lrelu(self.offset_conv3(offset))
        feature = self.dcnpack(current_sources[0], offset)
        if last_feature is not None: 
            last_feature = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = feature.shape
            last_feature = last_feature[..., :h, :w]
            feature = cat_conv(self.fea_conv, (feature, last_feature))
        feature = self.lrelu(feature)
        return offset, feature


class PCMLayer(nn.Module):
    """ Alignment module using Pyramid, Cascading and Mamba"""
    def __init__(self, args, depth, first_layer: bool):
        super(PCMLayer, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.depth = depth

        self.offset_conv1 = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        # Mamba块
        self.mamba = BiMambaBlock(self.nf, depth=self.depth)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not first_layer:
            self.offset_conv2 = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)
            self.fea_conv = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)

    def forward(self, current_sources: Tuple[torch.Tensor, torch.Tensor],last_feature: torch.Tensor):
        '''current_sources: concate的两帧图像
            last_feature: 上一层的特征
        '''
        feature = self.mamba(current_sources)
        if last_feature is not None: 
            last_feature = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = feature.shape
            last_feature = last_feature[..., :h, :w]
            feature = cat_conv(self.fea_conv, (feature, last_feature))
        feature = self.lrelu(feature)
        return feature

class PCMLayer_offset(nn.Module):
    """ Alignment module using Pyramid, Cascading and Mamba"""
    '''结合之前尺度的特征与当前尺度的特征'''
    def __init__(self, args, depth,  first_layer: bool):
        super(PCMLayer_offset, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.depth = depth
        
        self.offset_conv1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.mamba = BiMambaBlock(self.nf, depth=self.depth)                                                      
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1x1 = nn.Conv2d(self.nf * 2, self.nf, 1, 1, 0)

        if not first_layer:
            self.offset_conv2 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
            self.fea_conv = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
        self.fuse = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)

    def forward(self, current_sources: Tuple[torch.Tensor, torch.Tensor],last_feature: torch.Tensor,last_offset: torch.Tensor):
        '''current_sources: concate的两帧图像
            last_feature: 上一层的特征
            last_offset:上层的偏移
        '''
        offset = self.mamba(current_sources)

        B, C, H, W = offset.shape
        offset = self.lrelu(self.conv1x1(offset.view(B//2, -1, H, W)))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = offset.shape
            last_offset = last_offset[..., :h, :w]
            offset = cat_conv(self.offset_conv2, (offset, last_offset*2))
        offset = self.lrelu(self.offset_conv1(offset))
        feature = self.fuse(torch.cat((current_sources[:B//2, :, :, :],offset),dim=1))

        if last_feature is not None:
            last_feature = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = feature.shape
            last_feature = last_feature[..., :h, :w]
            feature = cat_conv(self.fea_conv, (feature, last_feature))
        feature = self.lrelu(feature)
        return offset, feature

class PCMLayer_offset_layer(nn.Module):
    """ Alignment module using Pyramid, Cascading and Mamba"""
    '''结合之前尺度的特征与当前尺度的特征'''
    def __init__(self, args, depth,  first_layer: bool):
        super(PCMLayer_offset_layer, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.depth = depth

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # 0→t
        self.l_offset_conv1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)

        self.l_mamba = BiMambaBlock(self.nf, depth=self.depth)                                                      
        self.l_conv1x1 = nn.Conv2d(self.nf * 2, self.nf, 1, 1, 0)

        if not first_layer:
            self.l_offset_conv2 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
            self.r_offset_conv2 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
            self.fea_conv = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)

        self.l_fuse = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)

        # t→0
        self.r_offset_conv1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)

        self.r_mamba = BiMambaBlock(self.nf, depth=self.depth)                                                      
        self.r_conv1x1 = nn.Conv2d(self.nf * 2, self.nf, 1, 1, 0)

        self.r_fuse = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)

        self.fusion = nn.Conv2d(2 * self.nf, self.nf, 1, 1)

        # motion offset
        layersLtROffset = []
        layersLtROffset.append(nn.Conv2d(128, 64, 3, 1, 1, bias=True))
        layersLtROffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersLtROffset.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.layersLtROffset = nn.Sequential(*layersLtROffset)

        layersRtLOffset = []
        layersRtLOffset.append(nn.Conv2d(128, 64, 3, 1, 1, bias=True))
        layersRtLOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersRtLOffset.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.layersRtLOffset = nn.Sequential(*layersRtLOffset)
        
        # fusion
        layersFusion = []
        layersFusion.append(nn.Conv2d(320, 320, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(320, 320, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(320, 320, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(320, 64, 1, 1, 0, bias=True))
        self.layersFusion = nn.Sequential(*layersFusion)


    def forward(self, f1:torch.Tensor, f2:torch.Tensor,last_feature: torch.Tensor,last_offset_l: torch.Tensor,last_offset_r: torch.Tensor):
        ''' f1: 当前层左边帧特征
            f2：当前层右边帧特征
            last_feature: 上一层混合的特征
            last_offset_l:上层0→t的偏移
            last_offset_r:上层t→0的偏移
        '''

        l_current_sources = torch.cat((f1,f2), dim=0)
        l_offset = self.l_mamba(l_current_sources)

        B, C, H, W = l_offset.shape
        l_offset = self.lrelu(self.l_conv1x1(l_offset.view(B//2, -1, H, W)))
        if last_offset_l is not None:
            last_offset_l = F.interpolate(last_offset_l, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = l_offset.shape
            last_offset_l = last_offset_l[..., :h, :w]
            l_offset = cat_conv(self.l_offset_conv2, (l_offset, last_offset_l*2))
        l_offset = self.lrelu(self.l_offset_conv1(l_offset))
        l_feature = self.l_fuse(torch.cat((l_current_sources[:B//2, :, :, :],l_offset),dim=1))


        r_current_sources = torch.cat((f2,f1), dim=0)
        r_offset = self.r_mamba(r_current_sources)

        B, C, H, W = r_offset.shape
        r_offset = self.lrelu(self.r_conv1x1(r_offset.view(B//2, -1, H, W)))
        if last_offset_r is not None:
            last_offset_r = F.interpolate(last_offset_r, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = r_offset.shape
            last_offset_r = last_offset_r[..., :h, :w]
            r_offset = cat_conv(self.r_offset_conv2, (r_offset, last_offset_r*2))
        r_offset = self.lrelu(self.r_offset_conv1(r_offset))
        r_feature = self.r_fuse(torch.cat((r_current_sources[:B//2, :, :, :],r_offset),dim=1))

        feature = cat_conv(self.fusion, (l_feature, r_feature))


        if last_feature is not None: 
            last_feature = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = feature.shape
            last_feature = last_feature[..., :h, :w]
            feature = cat_conv(self.fea_conv, (feature, last_feature))
        feature = self.lrelu(feature)

        LtROffset = self.layersLtROffset(torch.cat([f1,feature],dim=1))
        RtLOffset = self.layersLtROffset(torch.cat([feature,f2],dim=1))
        en_feature = self.layersFusion(torch.cat([f1, LtROffset, feature, RtLOffset, f2],dim=1))
        feature = en_feature + feature

        return feature,l_offset,r_offset



class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        L = 2 * H * W
        
        x = x.view(B, -1, L//2).transpose(1, 2)
        
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, L, C)
        return x.transpose(1, 2).contiguous()

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 4
        B = B // 2
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(torch.transpose(x, dim0=2, dim1=3).contiguous())], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        # print(y.shape)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B//2, H*W, 2, int(self.expand*C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B, H, W, int(self.expand*C))#.view(B//2, 2*H, 2*W, -1)
        
        y = y.half()
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)

        x = input * self.skip_scale + self.self_attention(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()


class BiMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                norm_layer=nn.LayerNorm,
                d_state=16,
            ))


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() # (B, C, H, W) ->(B, H, W, C)
        x = self.norm(x).permute(0, 3, 1, 2).contiguous() # (B, H, W, C) ->(B, C, H, W)
        x = self.proj(x)
        return x

