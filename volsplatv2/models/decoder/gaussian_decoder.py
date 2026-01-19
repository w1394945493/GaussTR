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

from . import rasterize_gaussians, render_cuda
from ...loss import CE_ssc_loss, lovasz_softmax
from ..encoder.common.gaussians import build_covariance       
@MODELS.register_module()
class GaussianDecoder(BaseModule):

    def __init__(self,
                 voxelizer,
                #  loss_lpips,
                 near,
                 far,
                 use_sh = True,
                 background_color=[0.0, 0.0, 0.0],
                 renderer_type = "vanilla",
                 num_class = 18,
                 with_empty = False,
                 empty_args=None,
                 manual_class_weight=[
                 1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                 1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                 1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],):
        super().__init__()

        self.voxelizer = MODELS.build(voxelizer)
        


        
        self.use_sh = use_sh
        self.near = near
        self.far = far
        self.renderer_type = renderer_type
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.lovasz_ignore = num_class - 1
        self.num_classes = num_class
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :]) # (3,) -> (1,1,3)
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :]) # (1, 1, 3)
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :]) # (1 1 4)
            self.register_buffer('empty_covs', build_covariance(self.empty_scale,self.empty_rot))
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :]) # (1 1 18)
            self.register_buffer('empty_opa', torch.ones(1)[None,:])  # (1 1)
        self.with_empty = with_empty
        self.class_weights = torch.tensor(manual_class_weight)
        
        
    def forward(self,
                gaussians,
                data,
                mode='tensor',

                **kwargs):

        data_samples = data
        
        means3d = gaussians.means # todo (b n 3)
        # harmonics = gaussians.harmonics # todo (b n 3 d_sh) | (b n c), c=rgb
        opacities = gaussians.opacities # todo (b n)
        scales = gaussians.scales
        rotations = gaussians.rotations
        covariances = gaussians.covariances
        features = gaussians.semantics # (b n num_class)

        if self.with_empty:
            assert features.shape[-1] == self.num_classes - 1
            bs = means3d.shape[0] # 获取当前 Batch Size

            empty_mean = self.empty_mean.expand(bs, -1, -1)     # (B, 1, 3)
            empty_covs = self.empty_covs.expand(bs, -1, -1, -1) # (B, 1, 3, 3) 如果是矩阵
            empty_sem = self.empty_sem.clone().expand(bs, -1, -1)       # (B, 1, num_classes)
            empty_opa = self.empty_opa.expand(bs, -1)       # (B, 1)         
            
            features = torch.cat([features, torch.zeros_like(features[..., :1])], dim=-1) 
            empty_sem[..., self.lovasz_ignore] += self.empty_scalar
            features = torch.cat([features, empty_sem], dim=1)
              
            means3d = torch.cat([means3d, empty_mean], dim=1)
            covariances = torch.cat([covariances,empty_covs],dim=1)
            opacities = torch.cat([opacities,empty_opa],dim=1)

        
        # todo 视图渲染
        # colors, rendered_depth = self.render_gaussians(extrinsics, intrinsics, 
        #                                                means3d, harmonics,opacities,scales,rotations,covariances,
        #                                                (h,w),device,)
        
        # todo --------------------------------------#
        # todo 占用预测：需要逐bs进行
        density, grid_feats = self.voxelizer(means3d, covariances, opacities, features) # (b,200,200,16) (b,200,200,16,18)
        occ_mask, occ_gt = data_samples['occ_cam_mask'],data_samples['occ_label'] # (b,200,200,16) (b,200,200,16)
        
        # todo ---------------------------------------#
        # todo 推理
        if mode == 'predict':
            probs = torch.softmax(grid_feats,dim=-1)
            occ_pred = probs.argmax(-1)
            outputs = [{
                # 'depth_pred': rendered_depth, # (b v h w)
                # 'img_pred': colors, # (b v 3 112 200)
                'occ_pred': occ_pred, # (b,200,200,16)
                'occ_mask': occ_mask, # (b,200,200,16)
                'occ_gt': occ_gt,     # (b,200,200,16)
                'gaussian': gaussians, # 
            }]
            return outputs

        losses = {}
        # todo ----------------------------------------#
        # todo 占用预测损失
        semantics = grid_feats[occ_mask].unsqueeze(0).transpose(1,2)
        sampled_label = occ_gt[occ_mask].unsqueeze(0)
        
        losses['loss_voxel_ce'] = 10.0 * \
            CE_ssc_loss(semantics,  # (b n_class n)
                        sampled_label,  # (b n)
                        self.class_weights.type_as(semantics), 
                        ignore_index=255,
                        )
        lovasz_input = torch.softmax(semantics, dim=1)
        losses['loss_voxel_lovasz'] = 1.0 * lovasz_softmax(
            lovasz_input.transpose(1, 2).flatten(0, 1), 
            sampled_label.flatten(), 
            ignore=self.lovasz_ignore) # todo 忽略背景类
        
        
        
        # todo ----------------------------------------#
        # todo 深度预测损失：
        # rendered_depth = rendered_depth.flatten(0,1) # todo ((b v) h w) v=6 h=112 w=192
        # depth = depth.flatten(0,1)  # todo ((b v) h w) depth: 来自Metric 3D生成的
        # losses['loss_depth'] = self.depth_loss(rendered_depth, depth)

        # rgb = colors.flatten(0,1) # todo rgb.shape:torch.Size([6, 3, 112, 192])
        # rgb_gt = data_samples['output_img']
        # rgb_gt = rgb_gt.flatten(0,1) / 255. # todo rgb_gt.shape: torch.Size([6, 3, 112, 192])
        
        # reg_loss = (rgb - rgb_gt) ** 2
        # losses['loss_l2'] = reg_loss.mean()
        # losses['loss_lpips'] = self.loss_lpips(rgb_gt, rgb)
        
        return losses



    def render_gaussians(self,
                         extrinsics,
                         intrinsics,
                         means3d,
                         harmonics,
                         opacities,
                         scales,
                         rotations,
                         covariances,
                         
                         image_shape,
                         device,
                         ):
        bs, n = extrinsics.shape[:2]
        h, w = image_shape
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
                
                near_plane=self.near,                 
                far_plane=self.far,

                render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
                channel_chunk=32)
        return colors, rendered_depth

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
