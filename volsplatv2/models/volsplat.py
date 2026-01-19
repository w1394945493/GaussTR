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
    
    def forward(self, mode='loss',**data):
        
        inputs = data['img']
        data_samples = data

        bs, n, _, input_h, input_w = inputs.shape # (b,v,3,H,W)
        
        depth = data_samples["depth"]  # (b v h w)
        
        # todo backbone+FPN特征提取
        multi_img_feats = self.extract_img_feat(img=inputs)
        img_feats = rearrange(multi_img_feats[0], "b v c h w -> (b v) c h w")

        img_aug_mat = data_samples['img_aug_mat']
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) # todo 内参(相对于原图像)
        extrinsics = data_samples['cam2lidar'] # todo  
        
        f_h, f_w = img_feats.shape[-2:]
        d_h, d_w = depth.shape[-2:]
        
        
        # todo resize
        resize = torch.diag(torch.tensor([f_w/input_w, f_h/input_h],
                                        dtype=img_aug_mat.dtype,device = img_aug_mat.device))
        mat = torch.eye(4).to(img_aug_mat.device)            
        mat[:2,:2] = resize
        mat = repeat(mat,"i j -> () () i j")
        img_aug_mat = mat @ img_aug_mat
        
        if (d_w != f_w) or (d_h != f_h):
            depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False)
            
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)  #! 外参：确定是cam2lidar还是cam2ego               
                img_feats,  # (bv c h w)
                depth=depth,
                voxel_resolution=self.voxel_resolution,
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                ) # sparse_input.C: (n,4) sparse_input.F: (n,128)         
        
        # todo -----------------------------------------------------#
        # todo 3D UNet网络 进行细化
        sparse_out = self.sparse_unet(sparse_input)   # 3D Sparse UNet

        if torch.equal(sparse_out.C, sparse_input.C) and sparse_out.F.shape[1] == sparse_input.F.shape[1]: # todo sparse_out.C: (N,4) 4(batch_indices,x,y,z)
            # Create new feature tensor
            new_features = sparse_out.F + sparse_input.F # todo 见论文 3(C).1) Feature Refinement 的 公式(8) 残差细化连接

            sparse_out_with_residual = ME.SparseTensor(
                features=new_features,
                coordinate_map_key=sparse_out.coordinate_map_key,
                coordinate_manager=sparse_out.coordinate_manager
            )
        else:
            # Handle coordinate mismatch
            print("Warning: Input and output coordinates inconsistent, skipping residual connection")
            sparse_out_with_residual = sparse_out # sparse_input.C: (n,4) sparse_input.F: (n,128)   

        
        # todo ------------------------------------------------------------------------#
        # todo 高斯参数预测
        gaussians = self.gaussian_head(sparse_out_with_residual) 
        del sparse_out_with_residual,sparse_out,sparse_input,new_features
        
        # todo ----------------------#
        # todo  这里进行了逐batch处理
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
            opacities = opacities,   # (b 1 n 1 1)
            raw_gaussians = rearrange(raw_gaussians,"b v r srf c -> b v r srf () c"), # (b 1 n 1 1 c)
            points = batched_points, # (b 1 n 3)
            voxel_resolution = self.voxel_resolution, # 0.001 体素网格尺寸 
        )
    
        return self.decoder(gaussians,data,mode=mode)

