from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.archs.arch_util import Upsample

from ..util import cat_conv
from ..part import ResidualBlock_noBN,PCMLayer_offset
from mamba_ssm.modules.mamba_simple import Mamba
from .videoMamba import VisionMamba_patch_RoPE_R
from .mambaIR import CAB
"""CycMuNet model partitions"""





class head(nn.Module):
    def __init__(self, args):
        super(head, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        
        if self.args.format == 'rgb':
                self.conv_first = nn.Conv2d(3, self.nf, 3, 1, 1)
                self.forward = self.forward_rgb
        if self.args.format == 'yuv444':
                self.conv_first = nn.Conv2d(3, self.nf, 3, 1, 1)
                self.forward = self.forward_yuv444
        if self.args.format == 'yuv422':
                self.conv_first_y = nn.Conv2d(1, self.nf, 3, 1, 1)
                self.conv_up = nn.ConvTranspose2d(2, self.nf, (1, 3), (1, 2), (0, 1), (0, 1))
                self.forward = self.forward_yuv42x
        if self.args.format == 'yuv420':
                self.conv_first_y = nn.Conv2d(1, self.nf, 3, 1, 1)
                self.conv_up = nn.ConvTranspose2d(2, self.nf, 3, 2, 1, 1)
                self.forward = self.forward_yuv42x
        if self.args.format == 'unk':
                raise ValueError(f'unknown input pixel format: {self.args.format}')

    def forward_rgb(self, x: torch.Tensor):
        x = self.lrelu(self.conv_first(x))
        return x

    def forward_yuv444(self, yuv: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(yuv, dim=1)
        x = self.lrelu(self.conv_first(x))
        return x

    def forward_yuv42x(self, yuv: Tuple[torch.Tensor, torch.Tensor]):
        y, uv = yuv
        y = self.conv_first_y(y)
        uv = self.conv_up(uv)
        x = self.lrelu(y + uv)
        return x

class feature_extract(nn.Module):
    def __init__(self, args):
        super(feature_extract, self).__init__()
        self.args = args
        self.nf = self.args.nf

        self.layers = self.args.layers
        self.front_RBs = 5

        self.feature_extraction = nn.Sequential(*(ResidualBlock_noBN(nf=self.nf) for _ in range(self.front_RBs)))
        self.fea_conv1s = nn.ModuleList(nn.Conv2d(self.nf, self.nf, 3, 2, 1, bias=True) for _ in range(self.layers - 1))
        self.fea_conv2s = nn.ModuleList(nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True) for _ in range(self.layers - 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor):
        features: List[torch.Tensor] = [self.feature_extraction(x)]
        for i in range(self.layers - 1):
            feature = features[-1]
            _, _, h, w = feature.shape
            h = torch.div(h + 1, 2, rounding_mode="trunc") * 2 - h
            w = torch.div(w + 1, 2, rounding_mode="trunc") * 2 - w
            feature = F.pad(feature, (0, w, 0, h), mode="replicate")
            feature = self.lrelu(self.fea_conv1s[i](feature))
            feature = self.lrelu(self.fea_conv2s[i](feature))
            features.append(feature)
        return tuple(features[::-1])  # lowest dimension layer at first
    
class feature_recon(nn.Module): # 用于特征重建，40个残差块
    def __init__(self, args):
        super(feature_recon, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.back_RBs = 40
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)

        self.recon_trunk = nn.Sequential(*(ResidualBlock_noBN(nf=self.nf) for _ in range(self.back_RBs)))

    def forward(self, x):
        out = self.recon_trunk(x)
        return out
    
class tail(nn.Module):
    def __init__(self, args):
        super(tail, self).__init__()
        self.args = args
        self.nf = self.args.nf
        if self.args.format == 'rgb':
                self.conv_last2 = nn.Conv2d(self.nf, 3, 3, 1, 1)
                self.forward = self.forward_rgb
        if self.args.format == 'yuv444':
                self.conv_last2 = nn.Conv2d(self.nf, 3, 3, 1, 1)
                self.forward = self.forward_yuv444
        if self.args.format == 'yuv422':
                self.conv_last_y = nn.Conv2d(self.nf, 1, 3, 1, 1)
                self.conv_last_uv = nn.Conv2d(self.nf, 2, (1, 3), (1, 2), (0, 1))
                self.forward = self.forward_yuv42x
        if self.args.format == 'yuv420':
                self.conv_last_y = nn.Conv2d(self.nf, 1, 3, 1, 1)
                self.conv_last_uv = nn.Conv2d(self.nf, 2, 3, 2, 1)
                self.forward = self.forward_yuv42x
        if self.args.format == 'unk':
                raise ValueError(f'unknown input pixel format: {self.args.format}')

    def forward_rgb(self, x):
        out = self.conv_last2(x)
        return out,

    def forward_yuv444(self, x):
        out = self.conv_last2(x)
        return out[:, :1, ...], out[:, 1:, ...]

    def forward_yuv42x(self, x):
        y = self.conv_last_y(x)
        uv = self.conv_last_uv(x)
        return y, uv

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):

                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:

            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class MSMM(nn.Module):
    def __init__(self, args):
        super(MSMM, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.img_size = self.args.img_size
        self.conv1x1 = nn.Conv2d(self.nf * 7, self.nf, 1, 1, 0)
        self.conv1x1_s = nn.Conv2d(self.nf * 3, self.nf, 1, 1, 0)
        self.conv1x1_ = nn.Conv2d(self.nf * 3, self.nf, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.num_block = 20
        self.num_block_short = 15 
        
        self.attn_drop_rate = 0.
        self.stmamba_f = VisionMamba_patch_RoPE_R(img_size=self.img_size, patch_size=4, depth=self.num_block, embed_dim=64, channels=64,num_frames=7,num_cls_tokens=35)

        self.stmamba_f1 = VisionMamba_patch_RoPE_R(img_size=self.img_size, patch_size=4, depth=self.num_block_short, embed_dim=64, channels=64,num_frames=3,num_cls_tokens=15)
        self.stmamba_f2 = VisionMamba_patch_RoPE_R(img_size=self.img_size, patch_size=4, depth=self.num_block_short, embed_dim=64, channels=64,num_frames=3,num_cls_tokens=15)
        self.stmamba_f3 = VisionMamba_patch_RoPE_R(img_size=self.img_size, patch_size=4, depth=self.num_block_short, embed_dim=64, channels=64,num_frames=3,num_cls_tokens=15)

        self.mamba_f = Mamba(d_model=self.nf)

        self.conv_blk1 = CAB(num_feat=self.nf*3)
        self.conv_last1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv_blk2 = CAB(num_feat=self.nf*3)
        self.conv_last2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv_blk3 = CAB(num_feat=self.nf*3)
        self.conv_last3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv_blk4 = CAB(num_feat=self.nf*3)
        self.conv_last4 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv_blk5 = CAB(num_feat=self.nf*3)
        self.conv_last5 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv_blk6 = CAB(num_feat=self.nf*3)
        self.conv_last6 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv_blk7 = CAB(num_feat=self.nf*3)
        self.conv_last7 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)

    def forward(self, l1, l2, l3,l4, l5, l6, l7, all_frames=False):
        lqlist = []
        lqlist1 = []
        lqlist2 = []
        lqlist3 = []
        l1 = self.lrelu(self.conv3x3(l1))
        lqlist.append(l1)
        lqlist1.append(l1)
        l2 = self.lrelu(self.conv3x3(l2))
        lqlist.append(l2)
        lqlist1.append(l2)
        l3 = self.lrelu(self.conv3x3(l3))
        lqlist.append(l3)
        lqlist1.append(l3)
        lqlist2.append(l3)
        l4 = self.lrelu(self.conv3x3(l4))
        lqlist.append(l4)
        lqlist2.append(l4)
        l5 = self.lrelu(self.conv3x3(l5))
        lqlist.append(l5)
        lqlist2.append(l5)
        lqlist3.append(l5)
        l6 = self.lrelu(self.conv3x3(l6))
        lqlist.append(l6)
        lqlist3.append(l6)
        l7 = self.lrelu(self.conv3x3(l7))
        lqlist.append(l7)
        lqlist3.append(l7)

        lqs = torch.stack(lqlist,dim=2)
        lqs1 = torch.stack(lqlist1,dim=2)
        lqs2 = torch.stack(lqlist2,dim=2)
        lqs3 = torch.stack(lqlist3,dim=2)


        B, C, N, H, W = lqs.shape
        assert C==self.nf
        n_tokens = lqs.shape[2:].numel()
        lq_flat = lqs.reshape(B, C, n_tokens).transpose(-1, -2)
        out = self.stmamba_f(lqs) # input:[B, C, N, H, W] out:[B, N*H*W, C]
        # video->image 
        out = out.reshape(B, -1, H, W)
        out = self.lrelu(self.conv1x1(out))
        #短序列1-3
        B_s, C_s, N_s, H_s, W_s = lqs1.shape
        assert C_s==self.nf
        out1 = self.stmamba_f1(lqs1) ## input:[B, C, N, H, W] out:[B, N*H*W, C]
        # video->image 
        out1 = out1.reshape(B_s, -1, H_s, W_s)
        out1 = self.lrelu(self.conv1x1_s(out1))
        #短序列3-5
        out2 = self.stmamba_f1(lqs2) ## input:[B, C, N, H, W] out:[B, N*H*W, C]
        # video->image 
        out2 = out2.reshape(B_s, -1, H_s, W_s)
        out2 = self.lrelu(self.conv1x1_s(out2))
        #短序列5-7
        out3 = self.stmamba_f1(lqs3) ## input:[B, C, N, H, W] out:[B, N*H*W, C]
        # video->image 
        out3 = out3.reshape(B_s, -1, H_s, W_s)
        out3 = self.lrelu(self.conv1x1_s(out3))

        # Mamba
        self.mamba_f.load_state_dict(self.stmamba_f.layers[-1].mixer.state_dict())
        out_f = self.mamba_f(lq_flat)
        out_f = out_f.transpose(1, 2).view(B, C, N, H, W)
        out_fs = torch.chunk(out_f,chunks=N,dim=2)
        out_fs = [torch.squeeze(i, dim=2) for i in out_fs]


        # concate
        outputs = []
        outputs.append(self.conv_last1(lqlist[0]) + self.conv1x1_(self.conv_blk1(torch.cat((out_fs[0], out, out1),dim=1))))
        outputs.append(self.conv_last1(lqlist[1]) + self.conv1x1_(self.conv_blk2(torch.cat((out_fs[1], out, out1),dim=1))))
        outputs.append(self.conv_last1(lqlist[2]) + self.conv1x1_(self.conv_blk3(torch.cat((out_fs[2], out, out1),dim=1))))
        outputs.append(self.conv_last1(lqlist[3]) + self.conv1x1_(self.conv_blk4(torch.cat((out_fs[3], out, out2),dim=1))))
        outputs.append(self.conv_last1(lqlist[4]) + self.conv1x1_(self.conv_blk5(torch.cat((out_fs[4], out, out3),dim=1))))
        outputs.append(self.conv_last1(lqlist[5]) + self.conv1x1_(self.conv_blk6(torch.cat((out_fs[5], out, out3),dim=1))))
        outputs.append(self.conv_last1(lqlist[6]) + self.conv1x1_(self.conv_blk7(torch.cat((out_fs[6], out, out3),dim=1))))
        return outputs


class GFM(nn.Module):
    '''基于Mamba的特征融合，不同尺度的特征混合模块'''
    def __init__(self, args):
        super(GFM, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.layers = self.args.layers
        self.depths = (2, 2, 2, 3)

        self.modules12 = nn.ModuleList(PCMLayer_offset(args, depth=self.depths[i],  first_layer=i == 0) for i in range(self.layers))
        self.modules21 = nn.ModuleList(PCMLayer_offset(args, depth=self.depths[i],  first_layer=i == 0) for i in range(self.layers))

        
        self.fusion = nn.Conv2d(2 * self.nf, self.nf, 1, 1)

        self.last_conv = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
    @staticmethod
    def fuse_features(modules, f1, f2):
        offset, feature = None, None
        for idx, sources in enumerate(zip(f1, f2)):
            sources = torch.cat(sources, dim=0)
            offset, feature = modules[idx](sources,feature,offset)
        return feature

    def forward(self, f1, f2):
        feature1 = self.fuse_features(self.modules12, f1, f2)
        feature2 = self.fuse_features(self.modules21, f2, f1)
        fused_feature = cat_conv(self.fusion, (feature1, feature2))
        
        fused_feature = self.last_conv(fused_feature)
        return fused_feature

class TFE(nn.Module):
    def __init__(self, args):
        super(TFE, self).__init__()
        self.args = args

        # motion offset
        layersLtROffset = []
        layersLtROffset.append(nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True))
        layersLtROffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.layersLtROffset = nn.Sequential(*layersLtROffset)
        self.conv1 = nn.Conv2d(128, 64, 3, 1, 1, bias=True)

        layersRtLOffset = []
        layersRtLOffset.append(nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True))
        layersRtLOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.layersRtLOffset = nn.Sequential(*layersRtLOffset)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # fusio
        layersFusion = []
        layersFusion.append(nn.Conv2d(320, 320, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(320, 320, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(320, 320, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(320, 64, 1, 1, 0, bias=True))
        self.layersFusion = nn.Sequential(*layersFusion)

    def forward(self,l, m, r):
        # x = torch.stack([l,m],dim=2)
        LtROffset = self.layersLtROffset(torch.stack([l,m],dim=2))
        B,C,N,H,W = LtROffset.shape
        LtROffset = LtROffset.reshape(B,-1,H,W)
        LtROffset = self.conv1(LtROffset)
        RtLOffset = self.layersLtROffset(torch.stack([m,r],dim=2))
        RtLOffset = RtLOffset.reshape(B,-1,H,W)
        RtLOffset = self.conv2(RtLOffset)
        en_feature = self.layersFusion(torch.cat([l, LtROffset, m, RtLOffset, r],dim=1))

        return en_feature+m

