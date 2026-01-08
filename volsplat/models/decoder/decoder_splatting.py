import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from einops import rearrange,repeat

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from . import rasterize_gaussians,render_cuda

@MODELS.register_module()
class DecoderSplatting(BaseModule):

    def __init__(self,
                 loss_lpips,
                 near,
                 far,
                 use_sh = True,
                 background_color=[0.0, 0.0, 0.0],
                 renderer_type = "vanilla",
                 ):
        super().__init__()

        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

        self.loss_lpips = MODELS.build(loss_lpips)
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg')) # todo mmseg
        self.use_sh = use_sh
        self.near = near
        self.far = far
        self.depth_limit = far
        self.renderer_type = renderer_type

    def forward(self,
                gaussians,
                data,
                mode='tensor',

                **kwargs):

        data_samples = data
        output_imgs = data_samples['output_img']
        h, w = output_imgs.shape[-2:]
        
        img_aug_mat = data_samples['output_img_aug_mat']
        cam2img = data_samples['output_cam2img']
        
        bs, n = cam2img.shape[:2] # todo: n: 视角数
        device = cam2img.device
        
        # todo 相对于输出图像的内参
        intrinsics = (img_aug_mat @ cam2img)[...,:3,:3] # (b v 3 3)
        # todo 归一化内参
        intrinsics[...,0,:] /= w
        intrinsics[...,1,:] /= h
        
        
        extrinsics = data_samples['output_cam2ego'] # (b v 4 4)
        
        
        means3d = gaussians.means # todo (b n 3)
        harmonics = gaussians.harmonics # todo (b n 3 d_sh) | (b n c), c=rgb
        opacities = gaussians.opacities # todo (b n)
        scales = gaussians.scales
        rotations = gaussians.rotations
        covariances = gaussians.covariances



        if self.renderer_type == "vanilla":

            near = repeat(torch.tensor([self.near],device=device),"1 -> b v",b=bs,v=n)
            far = repeat(torch.tensor([self.far],device=device),"1 -> b v",b=bs,v=n)
            
            colors, rendered_depth = render_cuda(
                extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
                image_shape = (h,w),
                near=rearrange(near, "b v -> (b v)"),
                far=rearrange(far, "b v -> (b v)"),
                background_color=repeat(self.background_color, "c -> (b v) c", b=bs, v=n),

                gaussian_means=repeat(means3d, "b g xyz -> (b v) g xyz", v=n),
                gaussian_sh_coefficients=
                    repeat(harmonics, "b g c d_sh -> (b v) g c d_sh", v=n) if self.use_sh else repeat(harmonics, "b g rgb -> (b v) g rgb ()", v=n),
                gaussian_opacities=repeat(opacities, "b g -> (b v) g", v=n),

                gaussian_scales=repeat(scales, "b g c -> (b v) g c", v=n) if covariances is None else None,
                gaussian_rotations=repeat(rotations, "b g c -> (b v) g c", v=n) if covariances is None else None,

                gaussian_covariances=repeat(covariances, "b g i j -> (b v) g i j", v=n) if covariances is not None else None,
                scale_invariant = False,
                use_sh= self.use_sh,
            )
            colors = rearrange(colors,'(bs n) c h w -> bs n c h w',bs=bs) # (b v c h w)
            rendered_depth = rearrange(rendered_depth,'(bs n) c h w -> bs n c h w',bs=bs).squeeze(2) # (b v h w)

        else:
            colors, rendered_depth = rasterize_gaussians(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                image_shape = (h,w),
                means3d=means3d,
                rotations=rotations,
                scales=scales,
                covariances=covariances,
                opacities=opacities.squeeze(-1),
                colors=rearrange(harmonics,"b g c d_sh -> b g d_sh c") if self.use_sh else harmonics, # todo (b n c d_sh)
                use_sh=self.use_sh,
                img_aug_mats=img_aug_mat,
                
                near_plane=self.near, # 
                far_plane=self.far,

                render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
                channel_chunk=32)

        # todo ---------------------------------------#
        # todo 推理
        if mode == 'predict':
            outputs = [{
                'depth_pred': rendered_depth, # (b v h w)
                'img_pred': colors, # (b v 3 112 200)
            }]
            return outputs

        losses = {}
        # todo ----------------------------------------#
        # todo 深度预测损失：
        # rendered_depth = rendered_depth.flatten(0,1) # todo ((b v) h w) v=6 h=112 w=192
        # depth = depth.flatten(0,1)  # todo ((b v) h w) depth: 来自Metric 3D生成的
        # losses['loss_depth'] = self.depth_loss(rendered_depth, depth)

        rgb = colors.flatten(0,1) # todo rgb.shape:torch.Size([6, 3, 112, 192])
        rgb_gt = data_samples['output_img']
        rgb_gt = rgb_gt.flatten(0,1) / 255. # todo rgb_gt.shape: torch.Size([6, 3, 112, 192])
        
        reg_loss = (rgb - rgb_gt) ** 2
        losses['loss_l2'] = reg_loss.mean()
        losses['loss_lpips'] = self.loss_lpips(rgb_gt, rgb)
        
        return losses

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion: # todo 这个没有用到
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion: # todo 只是用了l1 loss
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss






