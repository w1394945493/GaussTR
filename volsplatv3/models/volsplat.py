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

from .encoder.common.me_fea import project_features_to_me,sparse_to_dense_mask
from .utils.types import Gaussians

from .utils import flatten_multi_scale_feats, flatten_bsn_forward,cam2world
from .encoder.common.gaussians import build_covariance
from ..geometry.projection import sample_image_grid,get_world_rays

import MinkowskiEngine as ME

torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@MODELS.register_module()
class VolSplat(BaseModel):
    def __init__(self,
                 
                backbone,
                neck,    
                
                sparse_unet,
                sparse_gs,
                gaussian_adapter,

                volume_gs,

                decoder,
                
                use_checkpoint,
                voxel_resolution,
                pc_range,
                **kwargs):
        super().__init__(**kwargs)
        
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        
        self.sparse_unet = MODELS.build(sparse_unet)
        self.gaussian_head = MODELS.build(sparse_gs)
        self.gaussian_adapter = MODELS.build(gaussian_adapter)

        self.volume_gs = MODELS.build(volume_gs)
        
        self.decoder = MODELS.build(decoder)

        self.use_checkpoint = use_checkpoint
        self.voxel_resolution = voxel_resolution
        self.pc_range = pc_range
        
        print(cyan(f'successfully init Model!'))

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False)
        else:
            img_feats = self.backbone(img)
        
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
        bs,n,_,h,w = inputs.shape
        data_samples = data
    
        img_feats = self.extract_img_feat(img=inputs)
        
        
        depth = data_samples["depth"] # (b v 112 200)
        
        img_aug_mat = data_samples['img_aug_mat'] #! ori_img -> inputs
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) #! cam -> ori_img
        extrinsics = data_samples['cam2lidar']

        bs, n = intrinsics.shape[:2]

        vol_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        # todo 体素化
        sparse_input, aggregated_points, counts,pixel_points,pixel_feats = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)    
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),  # (bv c h w)
                depth=depth,
                
                voxel_resolution=self.voxel_resolution, 
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                
                vol_range = vol_range, # todo 仅保留范围内的点
                # vol_range=None, # 
                pixel_flag=True
                )   
        
        '''
        # debug
        import pickle
        occ_gt = data_samples['occ_label']
        for b in range(bs):
            mask_3d = sparse_to_dense_mask(sparse_input, vol_range, self.voxel_resolution,bs=b)
        
            data={
                'mask': mask_3d,
                'occ_gt': occ_gt[b].cpu().numpy(),
                'vol_range': vol_range,
                'voxel_size':self.voxel_resolution,
            }
            save_path = f'mask_3d_{b}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        
        '''
        # todo 3D UNet网络
        sparse_out = self.sparse_unet(sparse_input)
        
        # todo 残差连接
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
        
        raw_gaussians = self.gaussian_head(sparse_out_with_residual)
        gaussians_feat = self._sparse_to_batched(sparse_out_with_residual.F, raw_gaussians.C, bs).squeeze(1) # (b 1 N_max 128) -> (b N_max 128)
        del sparse_out_with_residual,sparse_out,sparse_input,new_features
        
        gaussian_params, valid_mask = self._sparse_to_batched(raw_gaussians.F, raw_gaussians.C, bs, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
        batched_points = self._sparse_to_batched(aggregated_points, raw_gaussians.C, bs)  # [b, 1, N_max, 3]        
        gaussians = self.post_process(gaussian_params, valid_mask, batched_points)

        #------------------------------------------#
        # todo BEV/TPV网格 高斯预测
        x_start, y_start, z_start, x_end, y_end, z_end = self.pc_range # todo [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        candidate_pos_mask, candidate_feat_mask = [], []
        # positions = gaussians.means
        # feats = gaussians_feat
        
        positions = pixel_points  # 使用原始的像素特征
        feats = pixel_feats
        
        for b in range(bs):
            mask_pixel_i = (positions[b, :, 0] >= x_start) & (positions[b, :, 0] <= x_end) & \
                        (positions[b, :, 1] >= y_start) & (positions[b, :, 1] <= y_end) & \
                        (positions[b, :, 2] >= z_start) & (positions[b, :, 2] <= z_end)            
            candidate_pos_mask_i = positions[b][mask_pixel_i]
            candidate_feat_mask_i = feats[b][mask_pixel_i]
            candidate_pos_mask.append(candidate_pos_mask_i)
            candidate_feat_mask.append(candidate_feat_mask_i)  

c # 用于计算参考点在图像中归一化的二维坐标

        gaussians_bev = self.volume_gs(
                [img_feats[0]],
                candidate_pos_mask,
                candidate_feat_mask,
                img_meats,
                )  

        gaussians = Gaussians(
            torch.cat([gaussians.means,gaussians_bev.means],dim=1),
            torch.cat([gaussians.scales,gaussians_bev.scales],dim=1),
            torch.cat([gaussians.rotations,gaussians_bev.rotations],dim=1),
            torch.cat([gaussians.covariances,gaussians_bev.covariances],dim=1),
            torch.cat([gaussians.harmonics,gaussians_bev.harmonics],dim=1),
            torch.cat([gaussians.opacities,gaussians_bev.opacities],dim=1),
            torch.cat([gaussians.semantics,gaussians_bev.semantics],dim=1),
        )
        
        
        return self.decoder(gaussians, data, mode=mode)
        
        
        
        
        
        
        
        
        
        
        
    