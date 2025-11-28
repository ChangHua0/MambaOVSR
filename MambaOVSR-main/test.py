import math
import logging
import os
import time
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cuda
import torch.backends.cudnn
import torchvision.utils
from pytorch_msssim import SSIM
import cv2
from model import MambaOVSR
from model.util import converter, normalizer
import dataset
from cycmunet.model import model_arg,model_arg_mamba
from cycmunet.run import test_arg
import utils.util as util

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_args = model_arg_mamba(nf=64,
                       groups=8,
                       img_size = (128, 160),
                       upscale_factor=2, # 2,4,8
                       format='yuv420',
                       layers=4,
                       )

mode="medium"
test_args = test_arg(
    # size=(128, 128),
    # size=(64, 64),
    size=(160, 128),
    checkpoint='',
    dataset_type="seplet",
    dataset_indexes=[
        "opera_data/medium_testlist.txt", # 数据集索引
    ],
    preview_interval=100,
    seed=0,
    batch_size=1,
    fp16=True,
)

data_mode = ''
save_folder = '/{}'.format(data_mode)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

force_data_dtype = torch.float16 if test_args.fp16 else None

# --------------------------------------
# Start of code

preview_interval = 1 \
    if (len(test_args.dataset_indexes) == 1 or math.gcd(100, len(test_args.dataset_indexes)) == 1) \
    else 1

nrow = 1 if test_args.size[0] * 9 > test_args.size[1] * 16 else 3

torch.manual_seed(test_args.seed)
torch.cuda.manual_seed(test_args.seed)

formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

logger = logging.getLogger('test_progress')
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

logger_init = logging.getLogger('initialization')
logger_init.addHandler(ch)
logger_init.setLevel(logging.DEBUG)

cvt = converter()
norm = normalizer()

dataset_types = {
    'seplet': dataset.ImageSequenceDatasetTest,# ImageSequenceDataset
    'triplet': dataset.ImageSequenceDatasetTest,
    'video': dataset.VideoFrameDataset
}
Dataset = dataset_types[test_args.dataset_type]

# breakpoint()
if len(test_args.dataset_indexes) == 1:
    ds_test = Dataset(test_args.dataset_indexes[0],
                      test_args.size,
                      model_args.upscale_factor,
                      mode = mode,
                      augment=False,
                      seed=test_args.seed)
else:
    ds_test = dataset.InterleavedDataset(*[
        Dataset(dataset_index,
                test_args.size,
                model_args.upscale_factor,
                # augment=True,
                augment=False,
                seed=test_args.seed + i)
        for i, dataset_index in enumerate(test_args.dataset_indexes)])
ds_test = DataLoader(ds_test,
                     num_workers=0,
                     batch_size=test_args.batch_size,
                    #  shuffle=Dataset.want_shuffle,  # Video dataset friendly
                     drop_last=True)

model = MambaOVSR(model_args)
model.eval()
num_params = 0
for param in model.parameters():
    num_params += param.numel()
logger_init.info(f"Model has {num_params} parameters.")

if not os.path.exists(test_args.checkpoint):
    logger_init.error(f"Checkpoint weight {test_args.checkpoint} not exist.")
    exit(1)
state_dict = torch.load(test_args.checkpoint, map_location=lambda storage, loc: storage)
state_dict.pop('fp.stmamba_f.rope.freqs_cos',None)
state_dict.pop('fp.stmamba_f.rope.freqs_sin',None)
state_dict.pop('fp.stmamba_f1.rope.freqs_cos',None)
state_dict.pop('fp.stmamba_f1.rope.freqs_sin',None)
state_dict.pop('fp.stmamba_f2.rope.freqs_cos',None)
state_dict.pop('fp.stmamba_f2.rope.freqs_sin',None)
state_dict.pop('fp.stmamba_f3.rope.freqs_cos',None)
state_dict.pop('fp.stmamba_f3.rope.freqs_sin',None)
load_result = model.load_state_dict(state_dict, strict=False)
if load_result.unexpected_keys:
    logger_init.warning(f"Unknown parameters ignored: {load_result.unexpected_keys}")
if load_result.missing_keys:
    logger_init.warning(f"Missing parameters not initialized: {load_result.missing_keys}")
logger_init.info("Checkpoint loaded.")

model = model.cuda()
if force_data_dtype:
    model = model.to(force_data_dtype)

epsilon = (1 / 255) ** 2


def rmse(a, b):
    return torch.mean(torch.sqrt((a - b) ** 2 + epsilon))


ssim_module = SSIM(data_range=1.0, nonnegative_ssim=True).cuda()


def ssim(a, b):
    # breakpoint()
    # return 1 - ssim_module(a, b) # 差异
    return ssim_module(a, b)

def psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.double()
    img2 = img2.double()
    # print(img1)
    # print('img1-2')
    # print(img2)
    mse = F.mse_loss(img1,img2,reduction='mean')
    # print(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10( 1 / math.sqrt(mse))


def recursive_cuda(li, force_data_dtype):
    if isinstance(li, (list, tuple)):
        return tuple(recursive_cuda(i, force_data_dtype) for i in li)
    else:
        if force_data_dtype is not None:
            return li.cuda().to(force_data_dtype)
        else:
            return li.cuda()

def normalize_tensor(tensor):
    tensor = tensor.to(torch.float32)
    
    min_val = tensor.min()
    max_val = tensor.max()
    
    if min_val < 0 or max_val > 1:
        tensor = (tensor - min_val) / (max_val - min_val)
    
    return tensor

def tensor2img_fast(tensor, rgb2bgr=False, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

if __name__ == '__main__':
    with torch.no_grad():
        total_loss = [0.0] * 6
        total_iter = len(ds_test)
        
        counter = 0
        with tqdm.tqdm(total=total_iter, desc=f"Test") as progress:
            # breakpoint()
            for it, data in enumerate(ds_test):
                sub_folder_name = data[-1][0]
                data = data[:2]
                # breakpoint()
                print(it)
                # (hf0, hf1, hf2), (lf0, lf1, lf2) = recursive_cuda(data, force_data_dtype)
                (hf0, hf1, hf2, hf3, hf4, hf5, hf6), (lf0, lf1, lf2, lf3, lf4, lf5,lf6) = recursive_cuda(data, force_data_dtype)
                if Dataset.pix_type == 'yuv':
                    # target = [cvt.yuv2rgb(*inp) for inp in (hf0, hf1, hf2, lf1)]
                    target = [cvt.yuv2rgb(*inp) for inp in (hf0, hf1, hf2, hf3, hf4, hf5, hf6, lf1 ,lf3,lf5)]
                    input = [cvt.yuv2rgb(*inp) for inp in (lf0, lf2, lf4, lf6)]

                else:
                    # target = [hf0, hf1, hf2, lf1]
                    target = [hf0, hf1, hf2, hf3, hf4, hf5, hf6, lf1 ,lf3,lf5]
                    input = [lf0, lf2, lf4, lf6]

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
                    # lf0, lf2 = cvt.rgb2yuv(lf0), cvt.rgb2yuv(lf2)
                    lf0, lf2, lf4, lf6 = cvt.rgb2yuv(lf0), cvt.rgb2yuv(lf2), cvt.rgb2yuv(lf4), cvt.rgb2yuv(lf6)

                t0 = time.perf_counter()
                # lf0, lf2 = norm.normalize_yuv_420(*lf0), norm.normalize_yuv_420(*lf2)
                lf0, lf2, lf4, lf6 = norm.normalize_yuv_420(*lf0), norm.normalize_yuv_420(*lf2), norm.normalize_yuv_420(*lf4), norm.normalize_yuv_420(*lf6)
                # outs = model(lf0, lf2, batch_mode='batch')
                outs = model(lf0, lf2, lf4, lf6, batch_mode='batch')

                t1 = time.perf_counter()
                t_forward = t1 - t0
                actual = [cvt.yuv2rgb(*norm.denormalize_yuv_420(*out)).float().to(dtype=torch.float16) for out in outs]
                out = [i[0:1].detach().float().cpu() for i in actual]
                ref = [i[0:1].detach().float().cpu() for i in target]
                ori = [i[0:1].detach().float().cpu() for i in input]
                # print(type(out))
                # breakpoint()

                outa = [tensor.squeeze(0) for tensor in out]
                refa = [tensor.squeeze(0) for tensor in ref]
                oria = [tensor.squeeze(0) for tensor in ori]
                
                save_sub_folder = os.path.join(save_folder, sub_folder_name)
                util.mkdirs(save_sub_folder)

                for idx, tout in enumerate(outa): # hf0, hf1, hf2, hf3, hf4, hf5, hf6, lf1 ,lf3,lf5
                    # indexit=it*3+idx+1
                    indexit=idx+1

                    image = tensor2img_fast(tout)
                    image = torchvision.transforms.ToPILImage()(tout)
                    image.save(save_sub_folder+"/{}.png".format(indexit))


                rmse_loss = [rmse(a, t).item() for a, t in zip(actual, target)]
                # breakpoint
                ssim_loss = [ssim(a, t).item() for a, t in zip(actual, target)]
                psnr_loss = [psnr(a, t) for a, t in zip(actual, target)]
                
                # AVG
                AVG_rmse = rmse_loss[:7]
                AVG_ssim = ssim_loss[:7]
                AVG_psnr = psnr_loss[:7]
                # VFI
                VFI_rmse = [AVG_rmse[i] for i in range(len(AVG_rmse)) if i % 2 != 0]
                VFI_ssim = [AVG_ssim[i] for i in range(len(AVG_ssim)) if i % 2 != 0]
                VFI_psnr = [AVG_psnr[i] for i in range(len(AVG_psnr)) if i % 2 != 0]

                t2 = time.perf_counter()
                t_loss = t2 - t1

                rmse_h = sum(AVG_rmse) / 7
                rmse_l = sum(VFI_rmse) / 3
                ssim_h = sum(AVG_ssim) / 7
                ssim_l = sum(VFI_ssim) / 3
                psnr_h = sum(AVG_psnr) / 7
                psnr_l = sum(VFI_psnr) / 3


                total_loss[0] += rmse_h
                total_loss[1] += rmse_l
                total_loss[2] += ssim_h
                total_loss[3] += ssim_l
                total_loss[4] += psnr_h
                total_loss[5] += psnr_l

                progress.set_postfix(ordered_dict={
                    "rmse_h": f"{rmse_h:.4f}",
                    "rmse_l": f"{rmse_l:.4f}",
                    "ssim_h": f"{ssim_h:.4f}",
                    "ssim_l": f"{ssim_l:.4f}",
                    "psnr_h": f"{psnr_h:.4f}",
                    "psnr_l": f"{psnr_l:.4f}",
                    "f": f"{t_forward:.4f}s",
                    "l": f"{t_loss:.4f}s",
                })
                progress.update()
                counter = counter + 1

        logger.info(f"Test Complete: "
                    f"RMSE HQ: {total_loss[0] / total_iter:.4f} "
                    f"RMSE LQ: {total_loss[1] / total_iter:.4f} "
                    f"SSIM HQ: {total_loss[2] / total_iter:.4f} "
                    f"SSIM LQ: {total_loss[3] / total_iter:.4f} "
                    f"PSNR HQ: {total_loss[4] / total_iter:.4f} "
                    f"PSNR LQ: {total_loss[5] / total_iter:.4f}")
