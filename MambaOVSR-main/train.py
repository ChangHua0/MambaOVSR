import math
import logging
import os
import pathlib
import time

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cuda
import torch.backends.cudnn
import torchvision.utils
from pytorch_msssim import SSIM

from model import MambaOVSR
from model.util import converter, normalizer
import dataset
from mambaovsr.model import model_arg, model_arg_mamba
from mambaovsr.run import train_arg


import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_args = model_arg_mamba(nf=64,
                       groups=8,
                       img_size=(64, 64),
                       upscale_factor=2,
                       format='yuv420',
                       layers=4, # 4
                       )

train_args = train_arg(
    # size=(128, 128),
    size=(64, 64),
    pretrained="",
    dataset_type="seplet",
    dataset_indexes=[
        "/opera_data/trainlist.txt" # 数据集索引
    ],
    preview_interval=100,
    seed=0,
    lr=0.001,
    start_epoch=1,
    end_epoch=21,
    sparsity=True,
    batch_size=6,
    autocast=False,
    loss_type='rmse',
    save_path='', # 模型保存路径
    save_prefix='seplet', # 模型保存前缀
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


preview_interval = 100 \
    if (len(train_args.dataset_indexes) == 1 or math.gcd(100, len(train_args.dataset_indexes)) == 1) \
    else 101

save_prefix = f'{train_args.save_prefix}_{train_args.size}_{model_args.upscale_factor}x_l{model_args.layers}'
# breakpoint()
save_path = pathlib.Path(train_args.save_path)

nrow = 1 if train_args.size[0] * 9 > train_args.size[1] * 16 else 3

torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)

formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

logger = logging.getLogger('train_progress')
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

logger_init = logging.getLogger('initialization')
logger_init.addHandler(ch)
logger_init.setLevel(logging.DEBUG)

cvt = converter()
norm = normalizer()

dataset_types = {
    'seplet':dataset.ImageSequenceDataset,
    'triplet': dataset.ImageSequenceDataset,
    'video': dataset.VideoFrameDataset
}
Dataset = dataset_types[train_args.dataset_type]
if len(train_args.dataset_indexes) == 1:
    ds_train = Dataset(train_args.dataset_indexes[0],
                       train_args.size,
                       model_args.upscale_factor,
                       augment=True,
                       seed=train_args.seed)
else:
    ds_train = dataset.InterleavedDataset(*[
        Dataset(dataset_index,
                train_args.size,
                model_args.upscale_factor,
                augment=True,
                seed=train_args.seed + i)
        for i, dataset_index in enumerate(train_args.dataset_indexes)])

# train_sampler = DistributedSampler(ds_train)
ds_train = DataLoader(ds_train,
                      num_workers=1,
                      batch_size=train_args.batch_size,
                      shuffle=True,  # Video dataset friendly
                      drop_last=True)
                    #   sampler=train_sampler)



model = MambaOVSR(model_args)


model.train()
model_updated = False
num_params = 0
for param in model.parameters():
    num_params += param.numel()
logger_init.info(f"Model has {num_params} parameters.")


if train_args.pretrained:
    if not os.path.exists(train_args.pretrained):
        logger_init.warning(f"Pretrained weight {train_args.pretrained} not exist.")
    state_dict = torch.load(train_args.pretrained, map_location=lambda storage, loc: storage)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        logger_init.warning(f"Unknown parameters ignored: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        logger_init.warning(f"Missing parameters not initialized: {load_result.missing_keys}")
    logger_init.info("Pretrained weights loaded.")

model.cuda()    

optimizer = optim.Adamax(model.parameters(), lr=train_args.lr, betas=(0.9, 0.999), eps=1e-8)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40000, eta_min=1e-7)

num_params_train = 0
for group in optimizer.param_groups:
    for params in group.get('params', []):
        num_params_train += params.numel()
logger_init.info(f"Model has {num_params} parameters to train.")

if train_args.sparsity:
    print("ASP have none")


epsilon = (1 / 255) ** 2 

def rmse(a, b):
    return torch.mean(torch.sqrt((a - b) ** 2 + epsilon))


ssim_module = SSIM(data_range=1.0, nonnegative_ssim=True).cuda()

def ssim(a, b):
    return 1 - ssim_module(a, b)


def recursive_cuda(li, force_data_dtype):
    if isinstance(li, (list, tuple)):
        return tuple(recursive_cuda(i, force_data_dtype) for i in li)
    else:
        if force_data_dtype is not None:
            return li.cuda(non_blocking=True).to(force_data_dtype)
        else:
            return li.cuda(non_blocking=True)


def train(epoch):
    
    epoch_loss = 0
    total_iter = len(ds_train)
   
    
    with tqdm.tqdm(total=total_iter, desc=f"Epoch {epoch}") as progress:
    
        for it, data in enumerate(ds_train):
            optimizer.zero_grad()

            def compute_loss(force_data_dtype=None):

                (hf0, hf1, hf2, hf3, hf4, hf5, hf6), (lf0, lf1, lf2, lf3, lf4, lf5,lf6) = recursive_cuda(data, force_data_dtype)

                if Dataset.pix_type == 'yuv':
                    target = [cvt.yuv2rgb(*inp) for inp in (hf0, hf1, hf2, hf3, hf4, hf5, hf6, lf1, lf3, lf5)]
                else:
                    target = [hf0, hf1, hf2, hf3, hf4, hf5, hf6, lf1, lf3, lf5]

                if it % preview_interval == 0:
                    if Dataset.pix_type == 'yuv':
                        org = [F.interpolate(cvt.yuv2rgb(y[0:1], uv[0:1]),
                                             scale_factor=(model_args.upscale_factor, model_args.upscale_factor),
                                             mode='nearest').detach().float().cpu()
                               for y, uv in (lf0, lf1, lf2, lf3, lf4, lf5,lf6)]
                    else:
                        org = [F.interpolate(lf[0:1],
                                             scale_factor=(model_args.upscale_factor, model_args.upscale_factor),
                                             mode='nearest').detach().float().cpu()
                               for lf in (lf0, lf1, lf2, lf3, lf4, lf5,lf6)]

                if Dataset.pix_type == 'rgb':
                    lf0, lf2, lf4, lf6 = cvt.rgb2yuv(lf0), cvt.rgb2yuv(lf2), cvt.rgb2yuv(lf4), cvt.rgb2yuv(lf6)

                t0 = time.perf_counter()
                lf0, lf2, lf4, lf6 = norm.normalize_yuv_420(*lf0), norm.normalize_yuv_420(*lf2), norm.normalize_yuv_420(*lf4), norm.normalize_yuv_420(*lf6)
                outs = model(lf0, lf2, lf4, lf6, batch_mode='batch')

                t1 = time.perf_counter()
                actual = [cvt.yuv2rgb(*norm.denormalize_yuv_420(*out)) for out in outs]

                if train_args.loss_type == 'rmse':
                    loss = [rmse(a, t)  for a, t in zip(actual, target)]
                elif train_args.loss_type == 'ssim':
                    loss = [ssim(a, t)  for a, t in zip(actual, target)]
                else:
                    raise ValueError("Unknown loss type: " + train_args.loss_type)

                assert not any(torch.any(torch.isnan(i)).item() for i in loss)

                t2 = time.perf_counter()

                return loss, t1 - t0, t2 - t1
            

            if train_args.autocast:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, t_forward, t_loss = compute_loss(torch.float16)
            else:
                loss, t_forward, t_loss = compute_loss() 
            

            total_loss = sum(loss)
            epoch_loss += total_loss.item()

            t3 = time.perf_counter()
            total_loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.view_as(param)
            optimizer.step()
            scheduler.step()
            t_backward = time.perf_counter() - t3

            global model_updated
            model_updated = True

            progress.set_postfix(ordered_dict={
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                "f": f"{t_forward:.4f}s",
                "l": f"{t_loss:.4f}s",
                "b": f"{t_backward:.4f}s",
            })
            progress.update()

    logger.info(f"Epoch {epoch} Complete: Avg. Loss: {epoch_loss / total_iter:.4f}")


def save_model(epoch):
    if epoch == -1:
        name = "snapshot"
    else:
        name = f"epoch_{epoch}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_path = save_path / f"{save_prefix}_{name}.pth"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Checkpoint saved to {output_path}")


if __name__ == '__main__':

    try:
        for epoch in range(train_args.start_epoch, train_args.end_epoch):
            train(epoch)
            save_model(epoch)
    except KeyboardInterrupt:
        if model_updated:
            save_model(-1)
  