import torch.nn as nn
import torch.nn.functional as F
import torch

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY

from ..util import cat_conv
from .bdm import Mamba
from .implicit_alignment import ImplicitWarpModule

"""CycMuNet Private Network Build Block"""


class Pro_align(nn.Module):
    def __init__(self, args):
        super(Pro_align, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.conv1x1 = nn.Conv2d(self.nf * 3, self.nf, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv1_3x3 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, l1, l2, l3):
        r1 = self.lrelu(self.conv3x3(l1))
        r2 = self.lrelu(self.conv3x3(l2))
        r3 = self.lrelu(self.conv3x3(l3))
        fuse = self.lrelu(cat_conv(self.conv1x1, [r1, r2, r3]))
        r1 = self.lrelu(cat_conv(self.conv1_3x3, [r1, fuse]))
        r2 = self.lrelu(cat_conv(self.conv1_3x3, [r2, fuse]))
        r3 = self.lrelu(cat_conv(self.conv1_3x3, [r3, fuse]))
        return l1 + r1, l2 + r2, l3 + r3


class SR(nn.Module):
    def __init__(self, args):
        super(SR, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)
        self.Pro_align = Pro_align(args)
        self.conv1x1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def upsample(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return self.lrelu(self.conv1x1(x))

    def forward(self, l1, l2, l3):
        l1, l2, l3 = self.Pro_align(l1, l2, l3)
        return tuple(self.upsample(i) for i in (l1, l2, l3))


class DR(nn.Module):
    def __init__(self, args):
        super(DR, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (1 / self.args.upscale_factor, 1 / self.args.upscale_factor)
        self.Pro_align = Pro_align(args)
        self.conv = nn.Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def downsample(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return self.lrelu(self.conv(x))

    def forward(self, l1, l2, l3):
        l1 = self.downsample(l1)
        l2 = self.downsample(l2)
        l3 = self.downsample(l3)
        return self.Pro_align(l1, l2, l3)


class Up_projection(nn.Module):
    def __init__(self, args):
        super(Up_projection, self).__init__()
        self.args = args
        self.SR = SR(args)
        self.DR = DR(args)
        self.SR1 = SR(args)

    def forward(self, l1, l2, l3):
        h1, h2, h3 = self.SR(l1, l2, l3)
        d1, d2, d3 = self.DR(h1, h2, h3)
        r1, r2, r3 = d1 - l1, d2 - l2, d3 - l3
        s1, s2, s3 = self.SR1(r1, r2, r3)
        return h1 + s1, h2 + s2, h3 + s3  # h1 + s1, h2 + s3, h3 + s3


class Down_projection(nn.Module):
    def __init__(self, args):
        super(Down_projection, self).__init__()
        self.args = args
        self.SR = SR(args)
        self.DR = DR(args)
        self.DR1 = DR(args)

    def forward(self, h1, h2, h3):
        l1, l2, l3 = self.DR(h1, h2, h3)
        s1, s2, s3 = self.SR(l1, l2, l3)
        r1, r2, r3 = s1 - h1, s2 - h2, s3 - h3
        d1, d2, d3 = self.DR1(r1, r2, r3)
        return l1 + d1, l2 + d2, l3 + d3



class Mamba_align(nn.Module):
    def __init__(self, # args,
        n_embd,                # dimension of the input features
        kernel_size=3,         # conv kernel size
        ):
        super(Mamba_align, self).__init__()
        #self.args = args
        # self.nf = self.args.nf
        self.nf = n_embd
        self.conv1x1 = nn.Conv2d(self.nf * 7, self.nf, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv1_3x3 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.n_embd = n_embd
        self.mamba = Mamba(n_embd, d_conv=kernel_size,use_fast_path=True, expand=1)

    def forward(self, l1, l2, l3, l4, l5, l6, l7):
        '''
        传入7帧图像
        将concate起来的图像送入Mamba模块提取时间信息
        '''
        r1 = self.lrelu(self.conv3x3(l1))
        r2 = self.lrelu(self.conv3x3(l2))
        r3 = self.lrelu(self.conv3x3(l3))
        r4 = self.lrelu(self.conv3x3(l4))
        r5 = self.lrelu(self.conv3x3(l5))
        r6 = self.lrelu(self.conv3x3(l6))
        r7 = self.lrelu(self.conv3x3(l7))

        # 增加一个时间维度
        dim = 2
        r1_expand = torch.unsqueeze(r1, dim= dim)
        r2_expand = torch.unsqueeze(r2, dim= dim)
        r3_expand = torch.unsqueeze(r3, dim= dim)
        r4_expand = torch.unsqueeze(r4, dim= dim)
        r5_expand = torch.unsqueeze(r5, dim= dim)
        r6_expand = torch.unsqueeze(r6, dim= dim)
        r7_expand = torch.unsqueeze(r7, dim= dim)
        
        
        # fuse = self.lrelu(cat_conv(self.conv1x1, [r1, r2, r3]))
        fuse = torch.cat((r1_expand, r2_expand, r3_expand, r4_expand, r5_expand, r6_expand, r7_expand), dim=dim)
        
        # 这里需要写一个Mamba模块处理3D视频数据
        # patch embed
        B, C, nf, H, W = fuse.shape
        assert C == self.n_embd
        n_tokens = fuse.shape[2:].numel()
        img_dims = fuse.shape[2:]
        x_flat = fuse.reshape(B, C, n_tokens).transpose(-1, -2)
        # Mamba
        out = self.mamba(x_flat) # out [B, L D]
        # un patch embed
        out = out.transpose(1,2).view(x_flat.shape[0], self.n_embd, nf, H, W)
        # video->image
        out = out.reshape(B, -1, H, W)
        
        out = self.lrelu(self.conv1x1(out))
        
        r1 = self.lrelu(cat_conv(self.conv1_3x3, [r1, out]))
        r2 = self.lrelu(cat_conv(self.conv1_3x3, [r2, out]))
        r3 = self.lrelu(cat_conv(self.conv1_3x3, [r3, out]))
        r4 = self.lrelu(cat_conv(self.conv1_3x3, [r4, out]))
        r5 = self.lrelu(cat_conv(self.conv1_3x3, [r5, out]))
        r6 = self.lrelu(cat_conv(self.conv1_3x3, [r6, out]))
        r7 = self.lrelu(cat_conv(self.conv1_3x3, [r7, out]))
        return l1 + r1, l2 + r2, l3 + r3, l4 + r4, l5 + r5, l6 + r6, l7 + r7


class Mamba_Flow_align(nn.Module):
    def __init__(self, # args,
        n_embd,                # dimension of the input features
        kernel_size=3,         # conv kernel size
        num_head = 8
        ):
        super(Mamba_Flow_align, self).__init__()
        #self.args = args
        # self.nf = self.args.nf
        self.nf = n_embd
        self.conv1x1 = nn.Conv2d(self.nf * 7, self.nf, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv1_3x3 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
        self.conv_rec = nn.Conv2d(self.nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.n_embd = n_embd
        self.mamba = Mamba(n_embd, d_conv=kernel_size,use_fast_path=True, expand=1)
        self.spynet_path = '/home/xuxin/xxhd/home/xuxin/ch/opera/cycmunet-main/torch/model/cycmunet/flownet/spynet_sintel_final-3d2a1287.pth'
        self.implicit_warp = ImplicitWarpModule(
            dim=n_embd,
            pe_dim=n_embd,
            num_heads=num_head,
            pe_temp=0.01)
        self.spynet = SpyNet(self.spynet_path)

        # 不训练光流提取网络
        for param in self.spynet.parameters():
            param.requires_grad = False

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w) # 全部 nx(t-1)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)# 不要第一帧

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        # if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
        #     flows_forward = None
        # else:
        #     flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        # if self.cpu_cache:
        #     flows_backward = flows_backward.cpu()
        #     flows_forward = flows_forward.cpu()

        return flows_backward

    def frame_alignment(self, feat_prop, feat_current, flow_n1):
        # pe_align: frame alignment using implicit alignment
        # 后一帧对齐到前一帧
        return self.implicit_warp(feat_prop, feat_current, flow_n1.permute(0, 2, 3, 1))

    def propagate(self, feats, flows):
        """得到对齐后的结果
        Args:
            feats: tensor with shape(B, nf, C, H, W).
            flows: Optical flows with shape (n, t - 1, 2, h, w).
        Return:
            对齐后的结果 shape(B, nf, C, H, W).
        """

        n, t, _, h, w = flows.size()
        # 初始化变量 frame_idx 和 flow_idx，用于确定帧和光流的索引
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        # mapping_idx 是一个索引列表用于处理镜像扩展的序列
        mapping_idx = list(range(0, len(feats)))
        mapping_idx += mapping_idx[::-1]

        # 向后传播: 则帧和光流的索引需要反转
        frame_idx = frame_idx[::-1]
        flow_idx = frame_idx
        result = []

        # 初始化为零张量，用于存储传播后的特征
        feat_prop = flows.new_zeros(n, self.n_embd, h, w) # B,C,H,W
        # 遍历每个帧的索引，根据 module_name 获取当前特征
        for i, idx in enumerate(frame_idx):
            feat_current = feats[mapping_idx[idx]]

            # 如果i>0, 实现帧对齐
            if i > 0:
                # 获取当前光流
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                # 对齐
                feat_prop = self.frame_alignment(feat_prop, feat_current, flow_n1)

            # 如果 i == 0，初始化特征传播为当前帧
            if i == 0: # 如果是第一帧，没有后一帧
                feat_prop = feat_current
            # 将传播后的特征 feat_prop 添加到列表中
            result.append(feat_prop)
        
        result = torch.stack(result, dim=2)
        return result

    def forward(self, l1, l2, l3, l4, l5, l6, l7):
        '''
        传入7帧图像
        将concate起来的图像送入Mamba模块提取时间信息
        '''
        # 通道数变为3以用于计算光流
        lqs = []
        lq1 = self.conv_rec(l1)
        lqs.append(lq1)
        lq2 = self.conv_rec(l2)
        lqs.append(lq2)
        lq3 = self.conv_rec(l3)
        lqs.append(lq3)
        lq4 = self.conv_rec(l4)
        lqs.append(lq4)
        lq5 = self.conv_rec(l5)
        lqs.append(lq5)
        lq6 = self.conv_rec(l6)
        lqs.append(lq6)
        lq7 = self.conv_rec(l7)
        lqs.append(lq7)
        lqs = torch.stack(lqs, dim=1)

        # 特征
        fuse = []
        r1 = self.lrelu(self.conv3x3(l1))
        fuse.append(r1)
        r2 = self.lrelu(self.conv3x3(l2))
        fuse.append(r2)
        r3 = self.lrelu(self.conv3x3(l3))
        fuse.append(r3)
        r4 = self.lrelu(self.conv3x3(l4))
        fuse.append(r4)
        r5 = self.lrelu(self.conv3x3(l5))
        fuse.append(r5)
        r6 = self.lrelu(self.conv3x3(l6))
        fuse.append(r6)
        r7 = self.lrelu(self.conv3x3(l7))
        fuse.append(r7)
        
        B, C, H, W = r1.shape
        nf = len(fuse)
        # 计算后向光流，用于描述时间帧之间的运动信息
        # 只使用后向光流
        flows_backward = self.compute_flow(lqs)

        # feature propgation
        feats = self.propagate(fuse, flows_backward) # B, C, nf, H, W

        # 这里需要写一个Mamba模块处理3D视频数据
        # patch embed
        assert C == self.n_embd
        n_tokens = feats.shape[2:].numel()
        img_dims = feats.shape[2:]
        x_flat = feats.reshape(B, C, n_tokens).transpose(-1, -2)
        # Mamba
        out = self.mamba(x_flat) # out [B, L D]
        # un patch embed
        out = out.transpose(1,2).view(x_flat.shape[0], self.nf, nf, H, W)
        # video->image
        out = out.reshape(B, -1, H, W)
        
        out = self.lrelu(self.conv1x1(out))
        
        r1 = self.lrelu(cat_conv(self.conv1_3x3, [r1, out]))
        r2 = self.lrelu(cat_conv(self.conv1_3x3, [r2, out]))
        r3 = self.lrelu(cat_conv(self.conv1_3x3, [r3, out]))
        r4 = self.lrelu(cat_conv(self.conv1_3x3, [r4, out]))
        r5 = self.lrelu(cat_conv(self.conv1_3x3, [r5, out]))
        r6 = self.lrelu(cat_conv(self.conv1_3x3, [r6, out]))
        r7 = self.lrelu(cat_conv(self.conv1_3x3, [r7, out]))
        return l1 + r1, l2 + r2, l3 + r3, l4 + r4, l5 + r5, l6 + r6, l7 + r7

class SR_upgrade(nn.Module):
    def __init__(self, args):
        super(SR_upgrade, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)
        self.Pro_align = Mamba_align(n_embd=self.nf)
        self.conv1x1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def upsample(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return self.lrelu(self.conv1x1(x))

    def forward(self, l1, l2, l3, l4, l5, l6, l7):
        l1, l2, l3, l4, l5, l6, l7 = self.Pro_align(l1, l2, l3, l4, l5, l6, l7)
        return tuple(self.upsample(i) for i in (l1, l2, l3, l4, l5, l6, l7))


class DR_upgrade(nn.Module):
    def __init__(self, args):
        super(DR_upgrade, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (1 / self.args.upscale_factor, 1 / self.args.upscale_factor)
        self.Pro_align = Mamba_align(n_embd=self.nf)
        self.conv = nn.Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def downsample(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return self.lrelu(self.conv(x))

    def forward(self, l1, l2, l3, l4, l5, l6, l7):
        l1 = self.downsample(l1)
        l2 = self.downsample(l2)
        l3 = self.downsample(l3)
        l4 = self.downsample(l4)
        l5 = self.downsample(l5)
        l6 = self.downsample(l6)
        l7 = self.downsample(l7)
        return self.Pro_align(l1, l2, l3, l4, l5, l6, l7)


class Up_projection_upgrade(nn.Module):
    def __init__(self, args):
        super(Up_projection_upgrade, self).__init__()
        self.args = args
        self.SR = SR_upgrade(args)
        self.DR = DR_upgrade(args)
        self.SR1 = SR_upgrade(args)

    def forward(self, l1, l2, l3, l4, l5, l6, l7):
        h1, h2, h3, h4, h5, h6, h7 = self.SR(l1, l2, l3, l4, l5, l6, l7)
        d1, d2, d3, d4, d5, d6, d7 = self.DR(h1, h2, h3, h4, h5, h6, h7)
        r1, r2, r3, r4, r5, r6, r7 = d1 - l1, d2 - l2, d3 - l3, d4 - l4, d5 - l5, d6 - l6, d7 - l7
        s1, s2, s3, s4, s5, s6, s7 = self.SR1(r1, r2, r3, r4, r5, r6, r7)
        return h1 + s1, h2 + s2, h3 + s3, h4 + s4, h5 + s5, h6 + s6, h7 + s7


class Down_projection_upgrade(nn.Module):
    def __init__(self, args):
        super(Down_projection_upgrade, self).__init__()
        self.args = args
        self.SR = SR_upgrade(args)
        self.DR = DR_upgrade(args)
        self.DR1 = DR_upgrade(args)

    def forward(self, h1, h2, h3, h4, h5, h6, h7):
        l1, l2, l3, l4, l5, l6, l7 = self.DR(h1, h2, h3, h4, h5, h6, h7)
        s1, s2, s3, s4, s5, s6, s7 = self.SR(l1, l2, l3, l4, l5, l6, l7)
        r1, r2, r3, r4, r5, r6, r7 = s1 - h1, s2 - h2, s3 - h3, s4 - h4, s5 - h5, s6 - h6, s7 - h7
        d1, d2, d3, d4, d5, d6, d7 = self.DR1(r1, r2, r3, r4, r5, r6, r7)
        return l1 + d1, l2 + d2, l3 + d3, l4 + d4, l5 + d5, l6 + d6, l7 + d7 

class Enhance_midframe(nn.Module):
    def __init__(self, args):
        super(Enhance_midframe, self).__init__()
        self.nf = args.nf
        self.conv_m = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
        self.conv1x1_1 = nn.Conv2d(self.nf * 5, self.nf * 4, 1, 1, 0)
        self.conv1x1_2 = nn.Conv2d(self.nf * 4, self.nf * 3, 1, 1, 0)
        self.conv1x1_3 = nn.Conv2d(self.nf * 3, self.nf * 2, 1, 1, 0)
        self.conv1x1_4 = nn.Conv2d(self.nf * 2, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self,l, m, r):
        lm  = self.conv_m(torch.cat((l,m), dim=1))
        mr  = self.conv_m(torch.cat((m,r), dim=1))
        concat = torch.cat((l, lm, m, mr, r), dim=1)
        res = self.lrelu(self.conv1x1_1(concat))
        res = self.lrelu(self.conv1x1_2(res))
        res = self.lrelu(self.conv1x1_3(res))
        res = self.conv1x1_4(res)

        return res + m




if __name__ == '__main__':
    
    model = Mamba_align(n_embd=64)
    l1 = torch.rand((1,64,64,64))
    l2 = torch.rand((1,64,64,64))
    l3 = torch.rand((1,64,64,64))
    l4 = torch.rand((1,64,64,64))
    l5 = torch.rand((1,64,64,64))
    l6 = torch.rand((1,64,64,64))
    l7 = torch.rand((1,64,64,64))
    r1, r2, r3, r4, r5, r6, r7 = model(l1, l2, l3, l4,l5, l6,l7)
    print(r1.shape)