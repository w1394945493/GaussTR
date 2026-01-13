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
                
                
                in_embed_dim,
                out_embed_dims,
                voxel_resolution,
                vol_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                use_embed=False,
                num_embed=1800,
                embed_dim=128,
                
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

        self.use_embed = use_embed
        self.embed_points = None
        self.embed_feats = None
        if use_embed:
            self.num_embed = num_embed
            self.embed_points = nn.Embedding(num_embed, 3) 
            self.embed_feats = nn.Embedding(num_embed, in_embed_dim)
            with torch.no_grad():
                # 分别对 x, y, z 轴进行均匀分布初始化
                nn.init.uniform_(self.embed_points.weight[:, 0], vol_range[0], vol_range[3])
                nn.init.uniform_(self.embed_points.weight[:, 1], vol_range[1], vol_range[4])
                nn.init.uniform_(self.embed_points.weight[:, 2], vol_range[2], vol_range[5])                
                # 正态分布初始化
                nn.init.normal_(self.embed_feats.weight, std=0.02)

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

    def plucker_embedder(
        self,
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    # todo 重写BaseModel的_run_forward()
    def _run_forward(self, data, mode): 
        results = self(data, mode=mode)
        return results
    
    def forward(self, data, mode='loss'):
        
        inputs = data['img']
        data_samples = data

        bs, n, _, input_h, input_w = inputs.shape # (b,v,3,H,W)
        device = inputs.device
        
        depth = data_samples["depth"]  # (b v h w)
        depth = torch.clamp(depth,max=51.2)
        
        # todo backbone+FPN特征提取
        multi_img_feats = self.extract_img_feat(img=inputs)
        img_feats = rearrange(multi_img_feats[0], "b v c h w -> (b v) c h w")

        img_aug_mat = data_samples['img_aug_mat']
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) # todo 内参(相对于原图像)
        # extrinsics = data_samples['cam2ego'] 
        extrinsics = data_samples['cam2lidar'] # todo  
        
        # todo 将特征图上采样回原图像大小
        # img_feats = self.upsampler(img_feats) # (bv c h w)        
        f_h, f_w = img_feats.shape[-2:]
        d_h, d_w = depth.shape[-2:]
        if (d_w != f_w) or (d_h != f_h):
            resize = torch.diag(torch.tensor([f_w/input_w, f_h/input_h],
                                            dtype=img_aug_mat.dtype,device = img_aug_mat.device))
            mat = torch.eye(4).to(img_aug_mat.device)            
            mat[:2,:2] = resize
            mat = repeat(mat,"i j -> () () i j")
            img_aug_mat = mat @ img_aug_mat
            depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False)
        
        # todo -------------------------------------------------------#
        # todo 将像素特征转为体素特征
        if self.use_embed:
            embed_points = self.embed_points.weight.unsqueeze(0).expand(bs,-1,-1) # (embed,3) -> (bs,embed,3)
            embed_feats = self.embed_feats.weight.unsqueeze(0).expand(bs,-1,-1) # (embed,dim) -> (bs,embed,dim)
        else:
            embed_points, embed_feats = None, None
        
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)  #! 外参：确定是cam2lidar还是cam2ego               
                img_feats,  # (bv c h w)
                depth=depth,
                voxel_resolution=self.voxel_resolution,
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                
                embed_points = embed_points,
                embed_feats = embed_feats,
                
                ) # sparse_input.C: (n,4) sparse_input.F: (n,128)         
        
        # todo -----------------------------------------------------#
        # todo 3D UNet网络
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

        gaussians = self.gaussian_head(sparse_out_with_residual) # todo MLP层网络
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
            opacities = opacities,   # (b 1 n 1 1)
            raw_gaussians = rearrange(raw_gaussians,"b v r srf c -> b v r srf () c"), # (b 1 n 1 1 c)
            points = batched_points, # (b 1 n 3)
            voxel_resolution = self.voxel_resolution, # 0.001     
        )
    
        return self.decoder(gaussians,data,mode=mode)

