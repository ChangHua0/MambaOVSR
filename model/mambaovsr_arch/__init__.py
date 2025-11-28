from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import Upsample

from ..util import RGBOrYUV

from .module import head, feature_extract, feature_recon, tail,Upsample,GFM,MSMM,TFE

class MambaOVSR(nn.Module):
    def __init__(self, args):
        super(MambaOVSR, self).__init__()
        self.args = args
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)
        self.upsample_mode = 'bilinear'

        self.head = head(args)
        self.fe = feature_extract(args)
        self.ff = GFM(args)
        self.fs = TFE(args)
        self.fp = MSMM(args)
        self.fr = feature_recon(args)
        self.upsample = Upsample(scale=self.args.upscale_factor, num_feat=self.args.nf)
        self.tail = tail(args)

    def merge_hf(self, lf, hf):
        return F.interpolate(lf, scale_factor=self.factor, mode='bilinear', align_corners=False) + hf

    def head_fe(self, x_or_yuv: RGBOrYUV):
        x = self.head(x_or_yuv)
        return self.fe(x)

    def fp_fr_tail(self, lf, all_frames):

        hfs = self.fp(*lf)
        hr = []
        hf1 = self.fr(hfs[0])
        hr1 = self.upsample(hf1)
        hr.append(hr1)
        hf2 = self.fr(hfs[1])
        hr2 = self.upsample(hf2)
        hr.append(hr2)
        hf3 = self.fr(hfs[2])
        hr3 = self.upsample(hf3)
        hr.append(hr3)
        hf4 = self.fr(hfs[3])
        hr4 = self.upsample(hf4)
        hr.append(hr4)
        hf5 = self.fr(hfs[4])
        hr5 = self.upsample(hf5)
        hr.append(hr5)
        hf6 = self.fr(hfs[5])
        hr6 = self.upsample(hf6)
        hr.append(hr6)
        hf7 = self.fr(hfs[6])
        hr7 = self.upsample(hf7)
        hr.append(hr7)

        if all_frames:
            outs = tuple(self.tail(i) for i in (*hr, hf2, hf4, hf6))
        else:
            outs = tuple(self.tail(i) for i in hr)
        return outs
    
    def forward_batch(self, lf0: RGBOrYUV, lf2: RGBOrYUV, lf4: RGBOrYUV, lf6: RGBOrYUV, all_frames=True, stop_at_conf=False):
        lf0s, lf2s, lf4s, lf6s = self.head_fe(lf0), self.head_fe(lf2), self.head_fe(lf4), self.head_fe(lf6)
        lf1 = self.ff(lf0s, lf2s)
        lf3 = self.ff(lf2s, lf4s)
        lf5 = self.ff(lf4s, lf6s)

        lf1 = self.fs(lf0s[-1], lf1, lf2s[-1])
        lf3 = self.fs(lf2s[-1], lf3, lf4s[-1])
        lf5 = self.fs(lf4s[-1], lf5, lf6s[-1])
        
        if stop_at_conf:  # TODO detect frame difference and exit if too big
            return
        lf = (lf0s[-1], lf1, lf2s[-1], lf3, lf4s[-1], lf5, lf6s[-1]) 
        return self.fp_fr_tail(lf, all_frames) 


    def forward_sequence(self, x_or_yuv: RGBOrYUV, all_frames=False):
        ls = self.head_fe(x_or_yuv)
        n = ls[0].shape[0]
        lf1, _ = self.ff([layer[:n - 1] for layer in ls], [layer[1:] for layer in ls])
        lf = (ls[-1][:n - 1], lf1, ls[-1][1:])
        return self.mu_fr_tail(lf, all_frames)

    # This is for symbolic tracing for sparsity
    def pseudo_forward_sparsity(self, lf0, lf1, lf2):
        hf0, *_ = self.mu(lf0, lf1, lf2, all_frames=True)
        return self.fr(hf0)

    def forward(self, lf0: RGBOrYUV, lf2: Union[RGBOrYUV, None] = None, lf4: Union[RGBOrYUV, None] = None, lf6: Union[RGBOrYUV, None] = None, sparsity_ex=None, /, batch_mode='batch',
            **kwargs):
        if batch_mode == '_no_use_sparsity_pseudo':
            return self.pseudo_forward_sparsity(lf0, lf2, sparsity_ex)
        if batch_mode == 'batch':
            outs = self.forward_batch(lf0, lf2, lf4, lf6, **kwargs)
        elif batch_mode == 'sequence':
            outs = self.forward_sequence(lf0, **kwargs)
        else:
            raise ValueError(f"Invalid batch_mode: {batch_mode}")
        return tuple(outs)
