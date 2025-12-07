from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from einops import rearrange,repeat
import torch.nn.functional as F
from math import isqrt

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gausstr.models.utils import flatten_multi_scale_feats

from gausstrv2.models.encoder import mv_feature_add_position,prepare_feat_proj_data_lists
from gausstrv2.models.encoder import warp_with_pose_depth_candidates
from gausstrv2.models.encoder import UNetModel
from gausstrv2.geometry import sample_image_grid,get_world_rays
from gausstrv2.models.encoder.common.gaussians import build_covariance
from gausstrv2.misc.sh_rotation import rotate_sh
from gausstrv2.models.utils.types import Gaussians

torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


# from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

@MODELS.register_module()
class GaussTRV2(BaseModel):

    def __init__(self,
                backbone,
                neck,
                pixel_gs,
                gauss_head,
                near,
                far,
                d_sh,
                ori_image_shape,
                use_checkpoint,
                 **kwargs):
        super().__init__(**kwargs)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.pixel_gs = MODELS.build(pixel_gs)

        self.gauss_head = MODELS.build(gauss_head)

        self.near = near
        self.far = far
        self.d_sh = d_sh
        self.ori_image_shape = ori_image_shape
        self.use_checkpoint = use_checkpoint

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

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False)
        else:
            img_feats = self.backbone(img)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped
    def plucker_embedder(
        self,
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker

    def forward(self, inputs, data_samples, mode='loss'):
        inputs, data_samples = self.prepare_inputs(inputs, data_samples)
        bs, n, _, h, w = inputs.shape # (b,v,3,H,W)
        device = inputs.device

        near = self.near
        near = torch.full((bs,n),near).to(device)
        far = self.far
        far = torch.full((bs,n),far).to(device)

        intrinsics = data_samples['cam2img'][...,:3,:3]
        extrinsics = data_samples['cam2ego']

        extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
        intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j") # 归一化的内参

        xy_ray, _ = sample_image_grid((h, w), device) # [0.1]网格点
        coordinates = repeat(xy_ray, "h w xy -> b v (h w) () () xy",b=bs,v=n)
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics) # (b v (h w) 1 1 3) (b v (h w) 1 1 3)

        origins = rearrange(origins,"b v (h w) srf spp c -> b v h w (srf spp c)", h=h,w=w)
        directions = rearrange(directions,"b v (h w) srf spp c -> b v h w (srf spp c)", h=h,w=w)
        pluckers = self.plucker_embedder(origins,directions) # (b v 6 h w)



        img_feats = self.extract_img_feat(img=inputs)
        pixel_gaussians = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_samples["depth"], # (b v h w)
                pluckers,
                origins,
                directions,
                intrinsics = data_samples['cam2img'][...,:3,:3],
                extrinsics = data_samples['cam2ego'],)
        '''
        import numpy as np
        # means3d = rearrange(means,"b v r srf spp xyz -> b (v r srf spp) xyz",)[0]
        means3d = pixel_gaussians.means[0]
        # 保存为 numpy
        np.save("means3d_gausstrv2_4.npy.npy", means3d.cpu().numpy())
        '''

        return self.gauss_head(
            pixel_gaussians = pixel_gaussians,
            inputs = inputs,
            image_shape=(h,w),
            near = near, far = far,
            mode=mode,
            **data_samples)

