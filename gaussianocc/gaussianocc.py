from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange,repeat
import torch.nn.functional as F

from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

#? 作为定义Model的参考

@MODELS.register_module()
class GaussianOCC(BaseModel):

    def __init__(self,
                 ori_image_shape,
                 patch_size = 14,
                 **kwargs):
        super().__init__(**kwargs)


        self.patch_size = patch_size
        self.ori_image_shape = ori_image_shape

        print(cyan(f'successfully init Model!'))




    def prepare_inputs(self, inputs_dict, data_samples):
        num_views = data_samples[0].num_views
        inputs = inputs_dict['imgs']

        cam2img = []
        cam2ego = []
        ego2global = []
        img_aug_mat = []
        depth = []
        feats = []
        sem_segs = []

        rgb_gts = [] # 真实图像

        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})
            # cam2img.append(data_samples[i].cam2img)

            # normalize the standred format into intrinsics
            ori_h, ori_w = self.ori_image_shape # (900, 1600)
            intrinsics = data_samples[i].cam2img
            intrinsics[:, 0, 0] /= ori_w
            intrinsics[:, 1, 1] /= ori_h
            intrinsics[:, 0, 2] /= ori_w
            intrinsics[:, 1, 2] /= ori_h
            cam2img.append(intrinsics)

            data_samples[i].set_metainfo(
                {'cam2ego': data_samples[i].cam2ego[:num_views]})
            cam2ego.append(data_samples[i].cam2ego)
            ego2global.append(data_samples[i].ego2global)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat[:num_views]})
                img_aug_mat.append(data_samples[i].img_aug_mat)
            # todo depth
            depth.append(data_samples[i].depth)
            # todo rgb_gts
            rgb_gts.append(data_samples[i].img)
            if hasattr(data_samples[i], 'feats'): # todo 特征图
                feats.append(data_samples[i].feats)
            if hasattr(data_samples[i], 'sem_seg'):
                sem_segs.append(data_samples[i].sem_seg) # todo 分割图
        data_samples = dict(
            rgb_gts = rgb_gts,
            depth=depth,
            cam2img=cam2img,
            cam2ego=cam2ego,
            num_views=num_views,
            ego2global=ego2global,
            img_aug_mat=img_aug_mat if img_aug_mat else None)
        if feats:
            data_samples['feats'] = feats
        if sem_segs:
            data_samples['sem_segs'] = sem_segs
        for k, v in data_samples.items():
            if isinstance(v, torch.Tensor) or not isinstance(v, Iterable):
                continue
            if isinstance(v[0], torch.Tensor):
                data_samples[k] = torch.stack(v).to(inputs)
            else:
                data_samples[k] = torch.from_numpy(np.stack(v)).to(inputs)
        return inputs, data_samples

    def forward(self, inputs, data_samples, mode='loss'):
        inputs, data_samples = self.prepare_inputs(inputs, data_samples)
        bs, n, _, h, w = inputs.shape # (b,v,3,H,W)
        device = inputs.device

        # 将图像缩放为14的倍数
        concat = rearrange(inputs,"b v c h w -> (b v) c h w")
        resize_h, resize_w = h // self.patch_size * self.patch_size, w // self.patch_size * self.patch_size
        concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)
        resize  = rearrange(concat,"(b v) c h w -> b v c h w",b=bs)


        return
