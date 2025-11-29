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

from einops import rearrange


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .decoder import rasterize_gaussians



@MODELS.register_module()
class GaussTRV2Head(BaseModule):

    def __init__(self,
                 loss_lpips,
                 depth_limit=51.2,
                 ):
        super().__init__()

        self.loss_lpips = MODELS.build(loss_lpips)
        self.depth_limit = depth_limit
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg')) # todo mmseg

    def forward(self,
                pixel_gaussians,
                gt_imgs,
                image_shape,
                depth, # todo 真实深度图 (b v h w)
                cam2img,
                cam2ego,
                img_aug_mat=None, # todo (b v 4 4)
                sem_segs=None,
                mode='tensor',
                **kwargs):

        bs, n = cam2img.shape[:2] # todo: n: 视角数
        h, w = gt_imgs.shape[-2:]

        means3d = pixel_gaussians.means
        features = torch.cat([
                        pixel_gaussians.rgbs,
                        pixel_gaussians.semantic,
                    ],dim = -1)
        opacities = pixel_gaussians.opacities.unsqueeze(-1)
        scales = pixel_gaussians.scales
        rotations = pixel_gaussians.rotations
        covariances = pixel_gaussians.covariances
        semantices = pixel_gaussians.semantic.softmax(-1)

        rendered = rasterize_gaussians(
            means3d,
            features,
            opacities.squeeze(-1),
            scales,
            rotations,
            cam2img,
            cam2ego,
            img_aug_mats=img_aug_mat,
            # image_size=(900, 1600),
            image_shape = image_shape,

            near_plane=0.1,
            far_plane=100,

            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32).flatten(0, 1) # (b v c h w) ((b v) c h w)

        rendered_depth = rendered[:, -1] # todo ((b v) h w) 深度图
        rendered_rgb = rendered[:, :3] #  ((b v) c h w) -> ((b v) c-1 h w)
        rendered_seg = rendered[:,3:-1]

        # todo ---------------------------------------#
        # todo 推理
        if mode == 'predict':
            rendered_seg = rearrange(rendered_seg,'bsv c h w -> bsv h w c').softmax(-1).argmax(-1)
            seg_pred = rearrange(rendered_seg,'(bs v) h w -> bs v h w',bs=bs) # (b v h w)

            depth_pred = rearrange(rendered_depth,'(bs v) h w -> bs v h w',bs=bs) # (b v h w)
            img_pred = rearrange(rendered_rgb,'(bs v) c h w -> bs v c h w',bs=bs) # (b v 3 h w )

            outputs = [{
                'occ_pred': None,
                'depth_pred': depth_pred, # (b v h w)
                'seg_pred':seg_pred,
                'img_pred': img_pred,
                'img_gt': gt_imgs,

            }]
            return outputs


        losses = {}
        depth = depth.clamp(max=self.depth_limit)
        depth = depth.flatten(0,1)

        losses['loss_depth'] = self.depth_loss(rendered_depth, depth)

        rgb_target = gt_imgs.flatten(0,1)
        reg_loss = (rendered_rgb - rgb_target) ** 2
        losses['loss_mae'] = reg_loss.mean()

        losses['loss_lpips'] = self.loss_lpips(rgb_target,rendered_rgb)

        probs = rendered_seg.flatten(2).mT
        target = sem_segs.flatten(0, 1).flatten(1).long()

        losses['loss_ce'] = F.cross_entropy(
            probs.mT,
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


