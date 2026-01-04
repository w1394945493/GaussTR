import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import imageio
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
import warnings
from einops import rearrange, einsum, repeat

from ...geometry import sample_image_grid, get_world_rays

from ..utils.utils import build_covariance
from ..utils.types import Gaussians

@MODELS.register_module()
class PixelGaussian(BaseModule):

    def __init__(self,
                 down_block=None,
                 mid_block=None,
                 up_block=None,
                 patch_sizes=None,
                 in_embed_dim=128,
                 out_embed_dims=[128, 256, 512, 512],
                 num_cams=6,
                 near=0.1,
                 far=1000.0,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.plucker_to_embed = nn.Linear(6, out_embed_dims[0])
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, out_embed_dims[0]))

        self.down_blocks = nn.ModuleList([])
        # in_channels = out_embed_dims[0] + 1 + 1 # concat pseudo depth and conf
        in_channels = out_embed_dims[0] + 1 # concat pseudo depth (without conf)
        for i, out_embed_dim in enumerate(out_embed_dims):
            is_final_block = i == len(out_embed_dims) - 1
            patch_size = patch_sizes[i] if patch_sizes is not None else None
            down_block.update(kv_compress_ratio=patch_size)
            down_block.update(attention_head_dim=out_embed_dim // down_block["num_attention_heads"])
            down_block.update(in_channels=in_channels)
            down_block.update(out_channels=out_embed_dim)
            down_block.update(add_downsample=not is_final_block)
            if i == 0:
                down_block.update(resnet_groups=1)
            else:
                down_block.update(resnet_groups=32)
            in_channels = out_embed_dim
            down_block_module = MODELS.build(down_block)
            self.down_blocks.append(down_block_module)

        # build middle block
        mid_block.update(in_channels=out_embed_dims[-1])
        mid_block.update(out_channels=out_embed_dims[-1])
        mid_block.update(attention_head_dim=out_embed_dims[-1] // mid_block["num_attention_heads"])
        self.mid_block = MODELS.build(mid_block)

        # build upsample blocks
        reversed_out_embed_dims = out_embed_dims[::-1]
        reversed_patch_sizes = patch_sizes[::-1] if patch_sizes is not None else [None] * len(out_embed_dims)
        out_channels = reversed_out_embed_dims[0]
        self.up_blocks = nn.ModuleList([])
        prev_output_channel = out_channels
        for i, (out_embed_dim, patch_size) in enumerate(zip(reversed_out_embed_dims, reversed_patch_sizes)):
            out_channels = reversed_out_embed_dims[i]
            in_channels = reversed_out_embed_dims[i]
            is_final_block = i == len(reversed_out_embed_dims) - 1
            up_block.update(attention_head_dim=out_embed_dim // up_block["num_attention_heads"])
            up_block.update(kv_compress_ratio=patch_size)
            up_block.update(in_channels=in_channels)
            up_block.update(prev_output_channel=prev_output_channel)
            up_block.update(out_channels=out_channels)
            up_block.update(add_upsample=not is_final_block)
            up_block_module = MODELS.build(up_block)
            self.up_blocks.append(up_block_module)
            prev_output_channel = out_channels

        # output & post-process
        self.num_cams = num_cams
        self.near = near
        self.far = far
        self.num_surfaces = 1

        self.upsampler = nn.Sequential(
            nn.Conv2d(in_embed_dim, out_embed_dims[0], 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        gs_channels = 3 + 1 + 3 + 4 + 3 # todo offset, opacity, scale, rotation, rgb
        self.gs_channels = gs_channels
        self.feature_norm = nn.GroupNorm(num_channels=out_embed_dims[0], num_groups=32, eps=1e-6)
        self.to_gaussians = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(out_embed_dims[0], gs_channels, 1),
        )
        self.opt_act = torch.sigmoid
        # todo scales(使用OmniScene中方法，对预测尺度进行处理)
        self.scale_act = lambda x: torch.exp(x) * 0.01 # todo log对数预测
        
        # todo MonoSplat/GaussianFormer等工作中的处理
        # self.scale_act = torch.sigmoid
        # self.scale_min = scale_min # todo 0.08
        # self.scale_max = scale_max # todo 0.64

        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def plucker_embedder(
        self,
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    # todo ----------------------------------#
    # todo (wys 10.17): 像素高斯预测
    # 把输入的多视角图像特征 + 几何信息(射线、深度、置信度等)转换为一批高斯点云(Gaussians)和特征表示，用于后续的体渲染或三维场景重建

    def forward(self,
                img_feats,
                depths_in,
                intrinsics,
                extrinsics,
                status="train"):
        """Forward training function."""
        h,w = depths_in.shape[-2:]
        bs, n = intrinsics.shape[:2]
        device = intrinsics.device
        # todo-----------------------------#
        # todo 0. 计算射线原点和方向                
        extrinsics_ = rearrange(extrinsics, "b v i j -> b v () () () i j")
        intrinsics_ = rearrange(intrinsics, "b v i j -> b v () () () i j") # 归一化的内参
                
        xy_ray, _ = sample_image_grid((h, w), device, normal=False)
        coordinates = repeat(xy_ray, "h w xy -> b v (h w) () () xy",b=bs,v=n)
        origins, directions = get_world_rays(coordinates, extrinsics_, intrinsics_) # (b v (h w) 1 1 3) (b v (h w) 1 1 3)
        origins = rearrange(origins,"b v (h w) srf spp c -> b v h w (srf spp c)", h=h,w=w)
        directions = rearrange(directions,"b v (h w) srf spp c -> b v h w (srf spp c)", h=h,w=w)
        pluckers = self.plucker_embedder(origins,directions)          
        
        # todo-----------------------------#
        # todo 1. 特征编码
        # upsample 4x downsampled img features to original size
        img_feats = self.upsampler(img_feats) # todo (bv c h/4 w/4) -> (bv c h w)
        img_feats = rearrange(img_feats, "(b v) c h w -> b v h w c", b=bs, v=self.num_cams) # todo (b v h w c)
        pluckers = rearrange(pluckers, "b v c h w -> b v h w c") # Pluckers: 射线的Plücker 坐标嵌入，编码几何方向
        plucker_embeds = self.plucker_to_embed(pluckers) # todo 全连接：编码 6 -> 128
        img_feats = img_feats + self.cams_embeds[None, :, None, None] + plucker_embeds # 每个像素特征 = 图像特征 + 相机ID + 射线几何方向信息
        img_feats = rearrange(img_feats, "b v h w c -> (b v) c h w")

        # todo-----------------------------#
        # todo 2. 深度与置信度拼接：使网络学习到每个像素点的可靠性与深度信息
        # rearrange pseudo depths and confs
        depths_in = rearrange(depths_in, "b v h w -> (b v) () h w") # todo 深度图 (b v h w) 这里使用的是度量深度(Metric 3D)！！！

        #? todo Metric 3D 未提供相应的深度置信度
        # confs_in = rearrange(confs_in, "b v h w -> (b v) () h w") # todo (b v h w)
        # todo depths_in
        # img_feats = torch.cat([img_feats, depths_in / 20.0, confs_in], dim=1) # 将深度与置信度作为附加通道嵌入
        img_feats = torch.cat([img_feats, depths_in / 20.0], dim=1)

        # todo-----------------------------#
        # todo 3. 编码器-解码器U-Net主体(多视图U-Net，见论文3.2节)
        # downsample
        sample = img_feats # (bs*6,128+1+1,h,w) 特征维度128+1维深度+1维深度置信度
        down_block_res_samples = (sample,)
        for block_id, down_block in enumerate(self.down_blocks):
            if self.use_checkpoint and status != "test":
                sample, res_samples = torch.utils.checkpoint.checkpoint(
                    down_block, sample, use_reentrant=False)
            else:
                sample, res_samples = down_block(sample)
            down_block_res_samples += res_samples
        # middile
        sample = self.mid_block(sample)
        # upsample
        for block_id, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            if self.use_checkpoint and status != "test":
                input_vars = (sample, res_samples)
                sample = torch.utils.checkpoint.checkpoint(
                    up_block, *input_vars, use_reentrant=False
                )
            else:
                sample = up_block(sample, res_samples)

        # todo 做了一个GruopNorm
        # rearrange features
        features = self.feature_norm(sample) # todo：features：得到每个像素位置的最终特征：用于解码高斯分布 (bs*6,128,h,w)
        bs = origins.shape[0]

        # todo-----------------------------#
        # todo 4. 高斯参数预测
        # post-process
        _, _, h, w = features.shape
        # todo：self.to_gaussians: 将多视角图像与射线几何信息转换为带颜色、旋转、尺度、透明度等属性的高斯点云，同时输出每个点的语义特征
        gaussians = self.to_gaussians(features) # todo: 高斯参数预测：(bs*v,128,h,w) -> (bs*v,14,h,w) 网络输出14维度向量
        gaussians = rearrange(gaussians, "(b v) (n c) h w -> b (v h w n) c",
                              b=bs, v=self.num_cams, n=1, c=self.gs_channels)

        offsets = gaussians[..., :3] # offsets: 每个高斯点相对射线采样点的偏移
        opacities = self.opt_act(gaussians[..., 3:4]) # opactites：透明度 # todo Sigmoid操作

        # todo Omni-Scene中的设计
        scales = self.scale_act(gaussians[..., 4:7]) # scales：空间尺度(控制体素体积) # todo e^(x) * 0.01

        # todo 参考MonoSplat、GaussianFormer中的设计
        # scales = self.scale_act(gaussians[..., 4:7]) # todo sigmoid归一化
        # scales = self.scale_min + (self.scale_max - self.scale_min) * scales # todo 将尺度限制在一定范围内

        rotations = self.rot_act(gaussians[..., 7:11]) # rotations：四元数旋转 # todo Normalize归一化操作
        rgbs = self.rgb_act(gaussians[..., 11:14]) # 颜色值 # todo sigmoid 操作
        # todo-----------------------------#
        # todo 5.空间点坐标计算
        depths_in = rearrange(depths_in, "(b v) c h w-> b (v h w) c", b=bs, v=self.num_cams)
        origins = rearrange(origins, "b v h w c -> b (v h w) c")
        origins = origins.unsqueeze(-2) # todo 射线原点

        #----------------------------------#
        # means计算过程：见论文3.2节式(3) # todo 位置计算：Omni-Scene中，没有计算深度值，而是直接使用的Metric 3D预测的度量深度
        directions = rearrange(directions, "b v h w c -> b (v h w) c")
        directions = directions.unsqueeze(-2)
        means = origins + directions * depths_in[..., None] # todo: 射线起点 + 深度 × 方向 得到空间位置
        means = rearrange(means, "b r n c -> b (r n) c")
        means = means + offsets # todo：再加上网络预测的offsets偏移量 在三维空间下的偏移量
        
        # todo ----------------------------#
        # todo 协方差计算(参照MonoSplat中的计算)
        covariances = build_covariance(scales, rotations) # todo shape: (b (v h w) 3 3)
        covariances = rearrange(covariances, "b (v h w) i j -> b v h w i j", v=self.num_cams, h=h, w=w)
        c2w_rotations = extrinsics[..., :3, :3] # todo 相机外参cam2ego (b v 3 3)
        c2w_rotations = rearrange(c2w_rotations,"b v i j -> b v () () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # todo 转换到自车坐标系下
        covariances = rearrange(covariances,"b v h w i j -> b (v h w) i j")


        # todo-----------------------------#
        # todo 6.整合输出
        pixel_gaussians = Gaussians(
            means=means, # (b N 3)
            covariances=covariances, # (b N 3 3)
            harmonics=rgbs, # (b N 3)
            scales=scales, # (b N 3)
            rotations=rotations, # (b N 4)
            opacities=opacities.squeeze(-1), # (b N 1) -> (b N)

        )
        if torch.isnan(means).any():
            a=1
        
        return pixel_gaussians

