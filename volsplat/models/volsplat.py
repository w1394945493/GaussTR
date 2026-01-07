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

import MinkowskiEngine as ME

from ..geometry import sample_image_grid,get_world_rays
from .encoder.common.me_fea import project_features_to_me
from .utils.types import Gaussians

torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


# from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor


    

@MODELS.register_module()
class VolSplat(BaseModel):

    def __init__(self,
                backbone,
                neck,
                sparse_unet,
                sparse_gs,
                gaussian_adapter,
            
                ori_image_shape,
                use_checkpoint,
                
                in_embed_dim,
                out_embed_dims,
                voxel_resolution,
                 **kwargs):
        super().__init__(**kwargs)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_embed_dim, out_embed_dims[0], 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        
        self.sparse_unet = MODELS.build(sparse_unet)
        self.gaussian_head = MODELS.build(sparse_gs)
        self.gaussian_adapter = MODELS.build(gaussian_adapter)

        self.ori_image_shape = ori_image_shape
        self.use_checkpoint = use_checkpoint
        
        self.voxel_resolution = voxel_resolution


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

        occ_gts = []

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

            if hasattr(data_samples[i].gt_pts_seg, 'semantic_seg'):
                occ_gts.append(data_samples[i].gt_pts_seg.semantic_seg) # todo occ占用图


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
        if occ_gts:
            data_samples['occ_gts'] = torch.cat(occ_gts) # todo occ占用真值
        for k, v in data_samples.items():
            if isinstance(v, torch.Tensor) or not isinstance(v, Iterable):
                continue
            if isinstance(v[0], torch.Tensor):
                data_samples[k] = torch.stack(v).to(inputs)
            else:
                data_samples[k] = torch.from_numpy(np.stack(v)).to(inputs)
        return inputs, data_samples

    def _sparse_to_batched(self, features, coordinates, batch_size, return_mask=False):

        device = features.device
        _, c = features.shape

        batch_features_list = []
        batch_sizes = []
        max_voxels = 0

        for batch_idx in range(batch_size):
            mask = coordinates[:, 0] == batch_idx
            batch_feats = features[mask]  # [N_i, C]
            batch_features_list.append(batch_feats)
            batch_sizes.append(batch_feats.shape[0])
            max_voxels = max(max_voxels, batch_feats.shape[0])

        # Create padded tensor [b, 1, N_max, C]
        batched_features = torch.zeros(batch_size, 1, max_voxels, c, device=device)

        # Create valid data mask [b, 1, N_max]
        if return_mask:
            valid_mask = torch.zeros(batch_size, 1, max_voxels, dtype=torch.bool, device=device)

        for batch_idx, batch_feats in enumerate(batch_features_list):
            n_voxels = batch_feats.shape[0]
            batched_features[batch_idx, 0, :n_voxels, :] = batch_feats
            if return_mask:
                valid_mask[batch_idx, 0, :n_voxels] = True

        if return_mask:
            return batched_features, valid_mask
        return batched_features

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
        
        depth = data_samples["depth"]  # (b v h w)
        
        # todo backbone特征提取
        multi_img_feats = self.extract_img_feat(img=inputs)
        img_feats = rearrange(multi_img_feats[0], "b v c h w -> (b v) c h w")
        img_feats = self.upsampler(img_feats) # (bv c h w)
        
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3)
        extrinsics = data_samples['cam2ego']        
        
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)
                img_feats,  # (bv c h w)
                depth=depth,
                voxel_resolution=self.voxel_resolution,
                b=bs, v=n,
                ) # sparse_input.C: (n,4) sparse_input.F: (n,128)         
        
        sparse_out = self.sparse_unet(sparse_input)   # 3D Sparse UNet

        if torch.equal(sparse_out.C, sparse_input.C) and sparse_out.F.shape[1] == sparse_input.F.shape[1]: # todo sparse_out.C: (N,4) 4(batch_indices,x,y,z)
            # Create new feature tensor
            new_features = sparse_out.F + sparse_input.F # todo 见论文 3(C).1) Feature Refinement 的 公式(8)

            sparse_out_with_residual = ME.SparseTensor(
                features=new_features,
                coordinate_map_key=sparse_out.coordinate_map_key,
                coordinate_manager=sparse_out.coordinate_manager
            )
        else:
            # Handle coordinate mismatch
            print("Warning: Input and output coordinates inconsistent, skipping residual connection")
            sparse_out_with_residual = sparse_out

        gaussians = self.gaussian_head(sparse_out_with_residual)
        del sparse_out_with_residual,sparse_out,sparse_input,new_features
        
        gaussian_params, valid_mask = self._sparse_to_batched(gaussians.F, gaussians.C, bs, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
        batched_points = self._sparse_to_batched(aggregated_points, gaussians.C, bs)  # [b, 1, N_max, 3]        

        opacity_raw = gaussian_params[..., :1]  # [b, 1, N_max, 1]
        opacity_raw = torch.where(
            valid_mask.unsqueeze(-1),  # [b, 1, N_max, 1]
            opacity_raw,
            torch.full_like(opacity_raw, -20.0)  # sigmoid(-20) ≈ 2e-9，
        )
        opacities = opacity_raw.sigmoid().unsqueeze(-1)  #[b, 1, N_max, 1, 1]
        raw_gaussians = gaussian_params[..., 1:]    #[b, 1, N_max, 37]
        raw_gaussians = rearrange(raw_gaussians,"... (srf c) -> ... srf c",srf=1,)
        
        gaussians = self.gaussian_adapter.forward(
            extrinsics = extrinsics, # (b v 4 4)
            intrinsics = intrinsics, # (b v 3 3)
            opacities = opacities,   # (b 1 n 1 1)
            raw_gaussians = rearrange(raw_gaussians,"b v r srf c -> b v r srf () c"), # (b 1 n 1 1 c)
            input_images =rearrange(inputs, "b v c h w -> (b v) c h w"), # (bv c h w)
            depth = depth, # (b v h w)
            coordidate = gaussians.C, # (n 4)
            points = batched_points, # (b 1 n 3)
            voxel_resolution = self.voxel_resolution, # 0.001        
        )
        
        gaussians = Gaussians(rearrange(gaussians.means,"b v r srf spp xyz -> b (v r srf spp) xyz"), # [b, 1, 256000, 1, 1, 3] -> [b, 256000, 3]
            rearrange(gaussians.scales,"b v r srf spp xyz -> b (v r srf spp) xyz"), # [b, 1, 256000, 1, 1, 3] -> [b, 256000, 3]
            rearrange(gaussians.rotations,"b v r srf spp d -> b (v r srf spp) d"), # [b, 1, 256000, 1, 1, 4] -> [b, 256000, 4]                             
            rearrange(gaussians.covariances,"b v r srf spp i j -> b (v r srf spp) i j",), # [2, 1, 256000, 1, 1, 3, 3] -> [2, 256000, 3, 3]
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"), # [2, 1, 256000, 1, 1, 3, 9] -> [2, 256000, 3, 9]            
            rearrange(gaussians.opacities,   "b v r srf spp -> b (v r srf spp)"), #[2, 1, 256000, 1, 1] -> [2, 256000]        
        ) 
                
        return

