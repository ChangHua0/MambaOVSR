import glob
import itertools
import pathlib
import random
from typing import List

import torch.utils.data as data
import numpy as np
import torchvision.transforms
from PIL import Image, ImageFilter



class ImageSequenceDatasetTest(data.Dataset):
    want_shuffle = True
    pix_type = 'rgb'

    def __init__(self, index_file, patch_size, scale_factor, mode, augment, seed=0):
        # breakpoint()
        self.dataset_base = pathlib.Path(index_file).parent
        self.sequences = [i for i in open(index_file, 'r', encoding='utf-8').read().split('\n')
                          if i if not i.startswith('#')]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.mode = mode
        self.rand = random.Random(seed)
        self.transform = torchvision.transforms.ToTensor()

    # 从指定路径加载图像，并返回一个包含所有图像的列表
    def _load_sequence(self, path):
        # breakpoint()
        GT_path = self.dataset_base / self.mode / "sequences" / path 
        LR_path = self.dataset_base / self.mode / "sequences_LR" / path 
        # LR_path = self.dataset_base / self.mode / "sequences_LR_X4" / "sequences" /path 
        # path = self.dataset_base / path 
        
        
        # files = glob.glob("*.png", root_dir=path)
        GT_files = sorted(glob.glob(str(GT_path)+"/*.png")) # 排序，7帧图像按次序输入
        LR_files = sorted(glob.glob(str(LR_path)+"/*.png")) # 排序，7帧图像按次序输入
        # files = glob.glob(str(path)+"/*.png")
        # breakpoint()
        assert len(GT_files) > 1
        assert len(LR_files) > 1
        GT_images = [Image.open(file) for file in GT_files]
        LR_images = [Image.open(file) for file in LR_files]

        # for i, img in enumerate(images):
        #     print(f"Image {i + 1}: {img.size}")
        
        # #breakpoint()
        # for img in images:
        #     print(img.size)
        # if not all(i.size != images[0].size for i in images[1:]):
        #     raise ValueError("sequence has different dimensions")
        first_image = GT_images[0]
        for image in GT_images[1:]:
            if first_image.size != image.size:
                raise ValueError("Images in the list are not all equal.")
        first_image = LR_images[0]
        for image in LR_images[1:]:
            if first_image.size != image.size:
                raise ValueError("Images in the list are not all equal.")
        # for i, img in enumerate(images):
        #     print(f"Image {i + 1}: {img.size}")
        
        # breakpoint()
        
        return GT_images,LR_images

    # 将输入的图像列表裁剪到指定的大小
    def _prepare_images(self, images: List[Image.Image]):
        w, h = images[0].size
        f = self.scale_factor
        sw, sh = self.patch_size
        sw, sh = sw * f, sh * f
        # print(h,sh,w,sw)
        #breakpoint()
        # assert h >= sh and w >= sw
        # # 确定裁剪的起始位置
        # dh, dw = self.rand.randint(0, h - sh), self.rand.randint(0, w - sw)
        # images = [i.crop((dw, dh, dw + sw, dh + sh)) for i in images]
        if h >= sh and w >= sw:
            dh, dw = self.rand.randint(0, h - sh), self.rand.randint(0, w - sw)
            images = [i.crop((dw, dh, dw + sw, dh + sh)) for i in images]
        else: # 如果原始图像尺寸的高或者宽 小于 最后输出的尺寸 就使用resize方法拉到指定尺寸
            images = [i.resize((sw, sh)) for i in images]
        return images

    # 图像翻转
    trans_groups = {
        'none': [None],
        'rotate': [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        'mirror': [None, Image.FLIP_LEFT_RIGHT],
        'flip': [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180],
        'all': [None] + [e.value for e in Image.Transpose],  # Image.Transpose 是一个枚举类，包含7种图像转置操作
    }

    trans_names = [e.name for e in Image.Transpose]

    def _augment_images(self, images: List[Image.Image], trans_mode='all'):
        trans_action = 'none'
        trans_op = self.rand.choice(self.trans_groups[trans_mode])
        if trans_op is not None:
            images = [i.transpose(trans_op) for i in images]
            trans_action = self.trans_names[trans_op]
        return images, trans_action

    scale_filters = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

    def _scale_images(self, images: List[Image.Image]):
        f = self.scale_factor
        return [i.resize((i.width // f, i.height // f), self.rand.choice(self.scale_filters)) for i in images]

    def _degrade_images(self, images: List[Image.Image]):
        degrade_action = None
        # decision = self.rand.randrange(4)
        decision = self.rand.randrange(3)
        if decision == 1:
            degrade_action = 'box'
            percent = 0.5 + 0.5 * self.rand.random()
            images = [Image.blend(j, j.copy().filter(ImageFilter.BoxBlur(1)), percent) for j in images]
        elif decision == 2:
            degrade_action = 'gaussian'
            radius = self.rand.random()
            images = [j.filter(ImageFilter.GaussianBlur(radius)) for j in images] # 对图像进行高斯模糊
        # elif decision == 3:
        #     degrade_action = 'halo'
        #     percent = 0.5 + 0.5 * self.rand.random()
        #     images = [Image.blend(i,
        #                           i.resize((i.width // 2, i.height // 2), resample=Image.LANCZOS)
        #                           .resize(i.size, resample=Image.BILINEAR), percent)
        #               for i in images]

        return images, degrade_action

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        original,LR_images = self._load_sequence(self.sequences[idx])
        #sequence = self._prepare_images(sequence)  # crop to requested size
        # original, _ = self._augment_images(sequence)  # flip and rotates
        # if self.augment==True: # 训练
        #     original, _ = self._augment_images(sequence)  # flip and rotates
        # elif self.augment==False: # 测试
        #     original, _ = self._augment_images(sequence, trans_mode='none')

        # lfs_pred = [np.array(lf.resize((lf.width // self.scale_factor, lf.height // self.scale_factor), Image.LANCZOS))
        #             for lf in original[1::2]] # 插帧位置的帧 1,3,5
        # # print(lfs_pred[0][4])
        # lfs_deg = self._scale_images(original[::2]) # 超分位置的帧 0,2,4,6
        # # 在训练阶段对需要超分的图像进行退化处理
        # if self.augment==True:
        #     lfs_deg, _ = self._degrade_images(lfs_deg)
        # lfs_deg, _ = self._degrade_images(lfs_deg)
        # print(len(lfs_deg))
        # degraded = [i for i in itertools.zip_longest(lfs_deg, lfs_pred) if i is not None]
        # degradeda = [i for i in lfs_pred if i is not None]
        # degradedb = [i for i in lfs_deg if i is not None]

        # degraded = [(d, p) for d, p in itertools.zip_longest(lfs_deg, lfs_pred) if d is not None and p is not None]
        # degraded =np.array(degraded)
        # breakpoint()
        # degraded=[]

        original = [self.transform(i) for i in original]
        degraded = [self.transform(i) for i in LR_images]
        # degradeda = [self.transform(i) for i in degradeda]
        # degradedb = [self.transform(i) for i in degradedb]

        # for i in range(3):
        #     degraded.append(degradedb[i])  # 先加 list2 的元素
        #     degraded.append(degradeda[i])
        # degraded.append(degradedb[-1])
        
        
        return original, degraded, self.sequences[idx]


class ImageSequenceDataset(data.Dataset):
    want_shuffle = True
    pix_type = 'rgb'

    def __init__(self, index_file, patch_size, scale_factor, augment, seed=0):
        # breakpoint()
        self.dataset_base = pathlib.Path(index_file).parent
        self.sequences = [i for i in open(index_file, 'r', encoding='utf-8').read().split('\n')
                          if i if not i.startswith('#')]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.rand = random.Random(seed)
        self.transform = torchvision.transforms.ToTensor()

    # 从指定路径加载图像，并返回一个包含所有图像的列表
    def _load_sequence(self, path):
        # breakpoint()
        path = self.dataset_base / "sequences" / path 
        files = sorted(glob.glob(str(path)+"/*.png")) # 排序，7帧图像按次序输入
        # files = glob.glob(str(path)+"/*.png")
        # breakpoint()
        assert len(files) > 1
        images = [Image.open(file) for file in files]


        first_image = images[0]
        for image in images[1:]:
            if first_image.size != image.size:
                raise ValueError("Images in the list are not all equal.")

        
        return images

    # 将输入的图像列表裁剪到指定的大小
    def _prepare_images(self, images: List[Image.Image]):
        w, h = images[0].size
        f = self.scale_factor
        sw, sh = self.patch_size
        sw, sh = sw * f, sh * f
        # print(h,sh,w,sw)
        #breakpoint()
        # assert h >= sh and w >= sw
        # # 确定裁剪的起始位置
        # dh, dw = self.rand.randint(0, h - sh), self.rand.randint(0, w - sw)
        # images = [i.crop((dw, dh, dw + sw, dh + sh)) for i in images]
        if h >= sh and w >= sw:
            dh, dw = self.rand.randint(0, h - sh), self.rand.randint(0, w - sw)
            images = [i.crop((dw, dh, dw + sw, dh + sh)) for i in images]
        else: # 如果原始图像尺寸的高或者宽 小于 最后输出的尺寸 就使用resize方法拉到指定尺寸
            images = [i.resize((sw, sh)) for i in images]
        return images

    # 图像翻转
    trans_groups = {
        'none': [None],
        'rotate': [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        'mirror': [None, Image.FLIP_LEFT_RIGHT],
        'flip': [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180],
        'all': [None] + [e.value for e in Image.Transpose],  # Image.Transpose 是一个枚举类，包含7种图像转置操作
    }

    trans_names = [e.name for e in Image.Transpose]

    def _augment_images(self, images: List[Image.Image], trans_mode='all'):
        trans_action = 'none'
        trans_op = self.rand.choice(self.trans_groups[trans_mode])
        if trans_op is not None:
            images = [i.transpose(trans_op) for i in images]
            trans_action = self.trans_names[trans_op]
        return images, trans_action

    scale_filters = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

    def _scale_images(self, images: List[Image.Image]):
        f = self.scale_factor
        return [i.resize((i.width // f, i.height // f), self.rand.choice(self.scale_filters)) for i in images]

    def _degrade_images(self, images: List[Image.Image]):
        degrade_action = None
        # decision = self.rand.randrange(4)
        decision = self.rand.randrange(3)
        if decision == 1:
            degrade_action = 'box'
            percent = 0.5 + 0.5 * self.rand.random()
            images = [Image.blend(j, j.copy().filter(ImageFilter.BoxBlur(1)), percent) for j in images]
        elif decision == 2:
            degrade_action = 'gaussian'
            radius = self.rand.random()
            images = [j.filter(ImageFilter.GaussianBlur(radius)) for j in images] # 对图像进行高斯模糊
        return images, degrade_action

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self._load_sequence(self.sequences[idx])
        sequence = self._prepare_images(sequence)  # crop to requested size
        if self.augment==True: # 训练
            
            original, _ = self._augment_images(sequence)  # flip and rotates
        elif self.augment==False: # 测试
            original = sequence

        lfs_pred = [np.array(lf.resize((lf.width // self.scale_factor, lf.height // self.scale_factor), Image.LANCZOS))
                    for lf in sequence[1::2]] # 插帧位置的帧 1,3,5
        # print(lfs_pred[0][4])
        lfs_deg = self._scale_images(sequence[::2]) # 超分位置的帧 0,2,4,6
        # # 在训练阶段对需要超分的图像进行退化处理
        degradeda = [i for i in lfs_pred if i is not None]
        degradedb = [i for i in lfs_deg if i is not None]

        degraded=[]

        original = [self.transform(i) for i in sequence]
        degradeda = [self.transform(i) for i in degradeda]
        degradedb = [self.transform(i) for i in degradedb]

        for i in range(3):
            degraded.append(degradedb[i])  # 先加 list2 的元素
            degraded.append(degradeda[i])
        degraded.append(degradedb[-1])

        
        return original, degraded
