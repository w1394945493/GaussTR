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

from mmdet.models import inverse_sigmoid

from .encoder.common.me_fea import project_features_to_me
from .utils.types import Gaussians

from .utils import flatten_multi_scale_feats, flatten_bsn_forward,cam2world
from .encoder.common.gaussians import build_covariance
from ..geometry.projection import sample_image_grid,get_world_rays


torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@MODELS.register_module()
class VolSplatV3(BaseModel):
    def __init__(self,
                 
                backbone,
                neck,    
                
                sparse_unet,
                sparse_gs,
                gaussian_adapter,
                decoder,
                
                use_checkpoint,
                voxel_resolution,
                **kwargs):
        super().__init__(**kwargs)
        
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        
        self.sparse_unet = MODELS.build(sparse_unet)
        self.gaussian_head = MODELS.build(sparse_gs)
        self.gaussian_adapter = MODELS.build(gaussian_adapter)
        
        self.decoder = MODELS.build(decoder)

        self.use_checkpoint = use_checkpoint
        self.voxel_resolution = voxel_resolution
        
        print(cyan(f'successfully init Model!'))

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        # if self.use_checkpoint and status != "test":
        #     img_feats = torch.utils.checkpoint.checkpoint(
        #                     self.backbone, img, use_reentrant=False)
        # else:
        #     img_feats = self.backbone(img)
        
        img_feats = self.backbone(img)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _sparse_to_batched(self, features, coordinates, batch_size, return_mask=False):

        device = features.device
        num_voxels, c = features.shape

        # todo -----------------------------------#
        batch_indices = coordinates[:, 0].long()
        v_counts = torch.bincount(batch_indices, minlength=batch_size) # [batch_size]
        max_voxels = v_counts.max().item()
        
        order = torch.arange(num_voxels, device=device)
        # 按照 batch_indices 排序后的偏移量
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        batch_offsets[1:] = torch.cumsum(v_counts, dim=0)
        # 计算每个点在自己 batch 内的相对位置索引 [0, 1, 2, ..., n_i-1]
        local_idx = order - batch_offsets[batch_indices]
        
        # 初始化稠密张量 [b, 1, N_max, C]
        batched_features = torch.zeros(batch_size, 1, max_voxels, c, device=device)
        batched_features[batch_indices, 0, local_idx] = features
        if return_mask:
            valid_mask = torch.zeros(batch_size, 1, max_voxels, dtype=torch.bool, device=device)
            valid_mask[batch_indices, 0, local_idx] = True
            return batched_features, valid_mask
        return batched_features

    def post_process(self,gaussian_params,valid_mask,batched_points):
        opacity_raw = gaussian_params[..., :1]  # [b, 1, N_max, 1]
        
        opacity_raw = torch.where(
            valid_mask.unsqueeze(-1),  # [b, 1, N_max, 1]
            opacity_raw,
            torch.full_like(opacity_raw, -20.0)  # sigmoid(-20) ≈ 2e-9，
        ) # todo 这里为了能保证多bs处理，将无效特征的透明度用-20填充了
        
        opacities = opacity_raw.sigmoid().unsqueeze(-1)  #[b, 1, N_max, 1, 1]
        raw_gaussians = gaussian_params[..., 1:]    #[b, 1, N_max, 37]
        raw_gaussians = rearrange(raw_gaussians,"... (srf c) -> ... srf c",srf=1,)
        
        # todo 预测高斯后处理
        gaussians = self.gaussian_adapter.forward(
            opacities = opacities,   # (b 1 n 1 1)
            raw_gaussians = rearrange(raw_gaussians, "b v r srf c -> b v r srf () c"), # (b 1 n 1 1 c)
            points = batched_points, # (b 1 n 3)
            voxel_resolution = self.voxel_resolution, #! 体素网格尺寸 
        )     
        
        return gaussians        
    
    
    def forward(self, mode='loss',**data):
        
        inputs = data['img']
        data_samples = data
    
        
        multi_img_feats = self.extract_img_feat(img=inputs)
        img_feats = rearrange(multi_img_feats[0], "b v c h w -> (b v) c h w")
        
        depth = data_samples["depth"]
        
        img_aug_mat = data_samples['img_aug_mat'] #! ori_img -> inputs
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) #! cam -> ori_img
        extrinsics = data_samples['cam2lidar']

        bs,n = intrinsics.shape[:2]
        
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)    
                img_feats,  # (bv c h w)
                depth=depth,
                
                voxel_resolution=self.voxel_resolution, 
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                )   
        
        
        # todo 3D UNet网络
        sparse_out = self.sparse_unet(sparse_input)
        
        # todo 残差连接
        if torch.equal(sparse_out.indices, sparse_input.indices) and sparse_out.features.shape[1] == sparse_input.features.shape[1]:
            new_features = sparse_out.features + sparse_input.features
            sparse_out_with_residual = sparse_out.replace_feature(new_features)
        else:
            print("Warning: Input and output coordinates inconsistent, skipping residual connection")
            sparse_out_with_residual = sparse_out
        
        gaussians = self.gaussian_head(sparse_out_with_residual)
        del sparse_out_with_residual,sparse_out,sparse_input,new_features
        
        gaussian_params, valid_mask = self._sparse_to_batched(gaussians.features, gaussians.indices, bs, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
        batched_points = self._sparse_to_batched(aggregated_points, gaussians.indices, bs)  # [b, 1, N_max, 3]        

        gaussians = self.post_process(gaussian_params,valid_mask,batched_points)
        return self.decoder(gaussians,data,mode=mode)
        
        
        
        
        
        
        
        
        
        
        
    