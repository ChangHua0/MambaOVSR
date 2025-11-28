# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi

import torch
from torch import nn

from einops import rearrange, repeat



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors))) # map函数将lambda函数应用于tensors中的每个元素,遍历tensors中的每个元素，计算每个元素的维度数，然后创建一个包含所有不同维度数的集合
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')



class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)



class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_frames = 7
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang': # 语言任务
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel': # 视觉任务
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant': # 常量
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
 
        freqs = torch.einsum('..., f -> ... f', t, freqs) # 对 t 和 freqs 执行爱因斯坦求和约定，计算所有位置和频率的组合
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2) # 通过重复频率维度扩展频率矩阵，使其适应输入的维度
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1) # 通过在最后一维连接张量来扩展频率矩阵，使其可以在所有空间维度上应用

        # 计算余弦和正弦频率矩阵，将它们展开为适当的形状
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).repeat(num_frames,1)
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).repeat(num_frames,1)
        # 使用 register_buffer 将频率矩阵注册为模型的常量，不会在训练过程中被更新
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin) 

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t): 
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        


class VisionRotaryEmbeddingHW(nn.Module):
    def __init__(
        self,
        dim,
        h_pt_seq_len,
        w_pt_seq_len,
        h_ft_seq_len=None,
        w_ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_frames = 7,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if h_ft_seq_len is None: h_ft_seq_len = h_pt_seq_len
        h_t = torch.arange(h_ft_seq_len) / h_ft_seq_len * h_pt_seq_len
        if w_ft_seq_len is None: w_ft_seq_len = w_pt_seq_len
        w_t = torch.arange(w_ft_seq_len) / w_ft_seq_len * w_pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', h_t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', w_t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1) # [16,16,64]

        # 计算余弦和正弦频率矩阵，将它们展开为适当的形状
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).repeat(num_frames,1) # [16*16, 64]
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).repeat(num_frames,1)
        # 使用 register_buffer 将频率矩阵注册为模型的常量，不会在训练过程中被更新
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t): 
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        
class VisionRotaryEmbeddingHW_R(nn.Module):
    def __init__(
        self,
        dim,
        h_pt_seq_len,
        w_pt_seq_len,
        h_ft_seq_len=None,
        w_ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_frames = 7,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if h_ft_seq_len is None: h_ft_seq_len = h_pt_seq_len
        h_t = torch.arange(h_ft_seq_len) / h_ft_seq_len * h_pt_seq_len
        if w_ft_seq_len is None: w_ft_seq_len = w_pt_seq_len
        w_t = torch.arange(w_ft_seq_len) / w_ft_seq_len * w_pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', h_t, freqs) # 8 * 16
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', w_t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1) # [16,16,64]

        # 计算余弦和正弦频率矩阵，将它们展开为适当的形状
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).repeat(num_frames,1) # [16*16, 64]
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).repeat(num_frames,1)
        # 使用 register_buffer 将频率矩阵注册为模型的常量，不会在训练过程中被更新
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, cls_positions, token_idx): 
        # cls_token 不加入位置信息
        # 分开图像token和类别tokens
        mask = torch.ones(t.size(1), dtype=torch.bool)
        mask[cls_positions] = False
        cls_token_mask = ~mask
        # 应用掩码，移除类别token
        image_token = t[:, mask, :]
        cls_token = t[:, cls_token_mask, :]
        # 只对图像token添加位置信息
        image_token = image_token * self.freqs_cos + rotate_half(image_token) * self.freqs_sin
        # 拼接图像token与cls_tokens
        tokens = torch.cat([image_token, cls_token], dim=1)[:, token_idx]
        return tokens
    
        # if t.shape[1] % 2 != 0:
        #     t_spatial = t[:, 1:, :]
        #     t_spatial = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
        #     return torch.cat((t[:, :1, :], t_spatial), dim=1)
        # else:
        #     return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin