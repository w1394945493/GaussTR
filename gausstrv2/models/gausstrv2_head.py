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

from .decoder import rasterize_gaussians,render_cuda



@MODELS.register_module()
class GaussTRV2Head(BaseModule):

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
                pixel_gaussians,
                inputs,
                image_shape,
                near,
                far,
                rgb_gts,
                depth,
                cam2img,
                cam2ego,
                img_aug_mat=None, # todo (b v 4 4)
                sem_segs=None,
                mode='tensor',
                **kwargs):

        bs, n = cam2img.shape[:2] # todo: n: 视角数
        h, w = image_shape

        means3d = pixel_gaussians.means
        harmonics = pixel_gaussians.harmonics # todo (b n c d_sh) | (b n c), c=rgb
        semantics = pixel_gaussians.semantics
        opacities = pixel_gaussians.opacities
        scales = pixel_gaussians.scales
        rotations = pixel_gaussians.rotations
        covariances = pixel_gaussians.covariances

        intrinsics = cam2img[...,:3,:3] # (b v 3 3)
        extrinsics = cam2ego # (b v 4 4)

        if self.renderer_type == "vanilla":

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

                gaussian_scales=repeat(scales, "b g c -> (b v) g c", v=n),
                gaussian_rotations=repeat(rotations, "b g c -> (b v) g c", v=n),

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
                opacities=opacities.squeeze(-1),
                colors=harmonics, # todo (b n c d_sh)
                use_sh=self.use_sh,
                img_aug_mats=img_aug_mat,

                near_plane=self.near,
                far_plane=self.far,

                render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
                channel_chunk=32)

        segs, r_depth = rasterize_gaussians(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape = (h,w),
            means3d=means3d,
            rotations=rotations,
            scales=scales,
            opacities=opacities.squeeze(-1),
            colors=semantics,

            use_sh=False,
            img_aug_mats=img_aug_mat,

            near_plane=self.near,
            far_plane=self.far,

            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32)


        # rendered_seg = rendered[:,3:-1]

        # todo ---------------------------------------#
        # todo 推理
        if mode == 'predict':
            seg_pred = rearrange(segs,'b v c h w -> b v h w c').softmax(-1).argmax(-1)

            outputs = [{
                'occ_pred': None,
                'depth_pred': rendered_depth, # (b v h w)
                'seg_pred':seg_pred,
                # 'seg_pred': None,
                'img_pred': colors,
                'img_gt': rgb_gts / 255.,
                # 'img_gt': inputs,

            }]
            return outputs


        losses = {}
        # todo 深度预测损失：
        rendered_depth = rendered_depth.flatten(0,1)
        # depth = depth.clamp(max=self.depth_limit)
        depth = depth.flatten(0,1)
        losses['loss_depth'] = 0.05 * self.depth_loss(rendered_depth, depth,
                                               criterion='l1')

        rgb = colors.flatten(0,1)
        # rgb_gt = inputs.flatten(0,1)
        rgb_gt = rgb_gts.flatten(0,1) / 255.
        reg_loss = (rgb - rgb_gt) ** 2

        losses['loss_mae'] = reg_loss.mean()

        losses['loss_lpips'] = self.loss_lpips(rgb_gt, rgb)

        probs = segs.flatten(0,1).flatten(2)
        target = sem_segs.flatten(0, 1).flatten(1).long()

        losses['loss_ce'] = F.cross_entropy(
            probs,
            target)

        return losses

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss


