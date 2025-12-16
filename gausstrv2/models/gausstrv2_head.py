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
from ..loss.lovzsz_softmax import lovasz_softmax_flat


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])

def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss

@MODELS.register_module()
class GaussTRV2Head(BaseModule):

    def __init__(self,
                 voxelizer,
                 loss_lpips,
                 near,
                 far,
                 use_sh = True,
                 background_color=[0.0, 0.0, 0.0],
                 num_classes = 18,
                 balance_cls_weight = True,
                manual_class_weight=[
                    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],

                 renderer_type = "vanilla",
                 ):
        super().__init__()

        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.voxelizer = MODELS.build(voxelizer)

        self.loss_lpips = MODELS.build(loss_lpips)
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg')) # todo mmseg
        self.use_sh = use_sh
        self.near = near
        self.far = far
        self.depth_limit = far

        self.num_classes = num_classes

        self.renderer_type = renderer_type

        self.occ_flag = False # todo 用于在指定epoch后进行occ估计
        if balance_cls_weight:
            if manual_class_weight is not None:
                self.class_weights = torch.tensor(manual_class_weight)
            else:
                class_freqs = nusc_class_frequencies
                self.class_weights = torch.from_numpy(1 / np.log(class_freqs[:num_classes] + 0.001))
            self.class_weights = num_classes * F.normalize(self.class_weights, 1, -1)
            print(self.__class__, self.class_weights)
        else:
            self.class_weights = torch.ones(num_classes)


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
                occ_gts=None,
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
                covariances=covariances,
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
            covariances=covariances,
            opacities=opacities.squeeze(-1),
            colors=semantics,

            use_sh=False,
            img_aug_mats=img_aug_mat,

            near_plane=self.near,
            far_plane=self.far,

            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32)

        # todo ---------------------------------------#
        # todo occ 占用预测
        if self.occ_flag:
            grid_feats, grid_density = self.voxelizer(
                            means3d=means3d,
                            opacities=opacities.unsqueeze(-1), # (b N) -> (b N 1)
                            features=semantics, # (b N n_class)
                            scales = scales,
                            rotations = rotations,
                            covariances=covariances) # todo

        # todo ---------------------------------------#
        # todo 推理
        if mode == 'predict':

            occ_preds = None
            if self.occ_flag:
                occ_preds = grid_feats.argmax(-1) # todo (b 200 200 16 18) -> (b 200 200 16)


            seg_preds = rearrange(segs,'b v c h w -> b v h w c').softmax(-1).argmax(-1)
            outputs = [{
                'occ_pred': occ_preds, # (b H W Z)
                'depth_pred': rendered_depth, # (b v h w)
                'seg_pred':seg_preds, # (b v h w)
                # 'seg_pred': None,
                'img_pred': colors,

            }]
            return outputs


        losses = {}
        # todo 深度预测损失：
        rendered_depth = rendered_depth.flatten(0,1) # todo ((b v) h w) v=6 h=112 w=192
        depth = depth.flatten(0,1)  # todo ((b v) h w) depth: 来自Metric 3D生成的
        losses['loss_depth'] = 0.05 * self.depth_loss(rendered_depth, depth,criterion='l1')

        rgb = colors.flatten(0,1) # todo rgb.shape:torch.Size([6, 3, 112, 192])
        rgb_gt = rgb_gts.flatten(0,1) / 255. # todo rgb_gt.shape: torch.Size([6, 3, 112, 192])
        reg_loss = (rgb - rgb_gt) ** 2
        losses['loss_l2'] = reg_loss.mean()

        losses['loss_lpips'] = self.loss_lpips(rgb_gt, rgb)

        # todo 2D分割图
        # probs = segs.flatten(0,1).flatten(2)
        # target = sem_segs.flatten(0, 1).flatten(1).long()

        # todo 3D占用图
        if self.occ_flag: # todo GaussianFormer中，使用了交叉熵损失和lovasz损失

            probs = rearrange(grid_feats,"b H W D C -> b C (H W D)") # todo probs.shape torch.Size([1, 18, 640000])
            target = rearrange(occ_gts,"b H W D ->b (H W D)").long() # todo target.shape torch.Size([1, 640000])
            # losses['loss_ce'] = F.cross_entropy(probs,target)
            losses['loss_ce'] = 10.0 * CE_ssc_loss(probs, target, self.class_weights.type_as(probs), ignore_index=255)

            lovasz_input = torch.softmax(probs, dim=1).transpose(1,2).flatten(0,1)
            target = occ_gts.flatten()
            ignore = 17 # todo 忽略天空类
            valid = (target != ignore)
            probas = lovasz_input[valid]
            labels = target[valid]
            losses['loss_lovasz'] = 1.0 * lovasz_softmax_flat(probas,labels)

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






