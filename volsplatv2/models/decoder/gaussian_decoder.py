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

# from . import rasterize_gaussians, render_cuda
from . import rasterize_gaussians
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
            # 构造 20x20x8=3200个空高斯用以填充背景
            voxel_size = empty_args['voxel_size']
            vol_range = empty_args['vol_range'] # [min_x, min_y, min_z, max_x, max_y, max_z]
            x_step = voxel_size * 10
            y_step = voxel_size * 10
            z_step = voxel_size * 2

            x_coords = torch.arange(vol_range[0] + x_step/2, vol_range[3], x_step)
            y_coords = torch.arange(vol_range[1] + y_step/2, vol_range[4], y_step)
            z_coords = torch.arange(vol_range[2] + z_step/2, vol_range[5], z_step)

            # 
            grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            means = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
            num_empty = means.shape[0]

            # 3. 尺度 3-sigma(radius = 3 * scale)
            # overlap_factor = 6.0
            overlap_factor = 6.0
            s_x = x_step / overlap_factor
            s_y = y_step / overlap_factor
            s_z = z_step / overlap_factor
                
            scales = torch.tensor([s_x, s_y, s_z]).repeat(num_empty, 1)
            # 4. 旋转 (单位四元数)
            rots = torch.tensor([1., 0., 0., 0.]).repeat(num_empty, 1)

            # 5. 注册到 buffer, 增加 Batch 维度
            self.register_buffer('empty_mean', means[None, :])           # (1, N_empty, 3)
            self.register_buffer('empty_scale', scales[None, :])         # (1, N_empty, 3)
            self.register_buffer('empty_rot', rots[None, :])             # (1, N_empty, 4)
            # 计算对应的协方差
            self.register_buffer('empty_covs', build_covariance(self.empty_scale, self.empty_rot))
            # 语义特征 (全0) 和 透明度 
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :].repeat(1, num_empty, 1)) 
            
            # self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            # self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float))
            # self.empty_scalar = nn.Parameter(torch.full((1, num_empty), 1.0, dtype=torch.float)) # (1,N)
            self.register_buffer('empty_scalar', torch.full((1, num_empty), 10.0, dtype=torch.float))

            # self.register_buffer('empty_opa', torch.ones(num_empty)[None, :]) # (1, N_empty)        
            init_opa_val = 0.1
            raw_val = torch.log(torch.tensor(init_opa_val / (1 - init_opa_val)))
            # self.empty_opa = nn.Parameter(torch.full((1, num_empty), raw_val, dtype=torch.float))    
            self.register_buffer('empty_opa', torch.full((1, num_empty), raw_val, dtype=torch.float))

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
            
            bs = means3d.shape[0] # 获取当前 Batch Size

            empty_mean = self.empty_mean.expand(bs, -1, -1)     # (B, n_bg, 3)
            empty_covs = self.empty_covs.expand(bs, -1, -1, -1) # (B, n_bg, 3, 3) 如果是矩阵
            empty_sem = self.empty_sem.expand(bs, -1, -1).clone()       # (B, n_bg, num_classes)
            
            empty_opa = torch.sigmoid(self.empty_opa).expand(bs, -1)       # (B, 1)         
            
            empty_scalar = F.softplus(self.empty_scalar) # 确保背景增益＞0
            empty_sem[..., self.lovasz_ignore] += empty_scalar.expand(bs, -1)

            
            features = torch.cat([features, empty_sem], dim=1)
              
            means3d = torch.cat([means3d, empty_mean], dim=1)
            covariances = torch.cat([covariances,empty_covs],dim=1)
            
            opacities = torch.cat([opacities,empty_opa],dim=1)


        # 视图渲染
        # colors, rendered_depth = self.render_gaussians(extrinsics, intrinsics, 
        #                                                means3d, harmonics,opacities,scales,rotations,covariances,
        #                                                (h,w),device,)
        
        # todo --------------------------------------#
        # todo 占用预测
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
                'gaussian': gaussians,
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
        if self.renderer_type == "gsplat":
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
        # elif self.renderer_type == "vanilla":

        #     near = repeat(torch.tensor([self.near],device=device),"1 -> b v",b=bs,v=n)
        #     far = repeat(torch.tensor([self.far],device=device),"1 -> b v",b=bs,v=n)
            
        #     colors, rendered_depth = render_cuda(
        #         extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
        #         intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
        #         image_shape = (h,w),
        #         near=rearrange(near, "b v -> (b v)"),
        #         far=rearrange(far, "b v -> (b v)"),
        #         background_color=repeat(self.background_color, "c -> (b v) c", b=bs, v=n),

        #         gaussian_means=repeat(means3d, "b g xyz -> (b v) g xyz", v=n),
        #         gaussian_sh_coefficients=
        #             repeat(harmonics, "b g c d_sh -> (b v) g c d_sh", v=n) if self.use_sh else repeat(harmonics, "b g rgb -> (b v) g rgb ()", v=n),
        #         gaussian_opacities=repeat(opacities, "b g -> (b v) g", v=n),

        #         gaussian_scales=repeat(scales, "b g c -> (b v) g c", v=n) if covariances is None else None,
        #         gaussian_rotations=repeat(rotations, "b g c -> (b v) g c", v=n) if covariances is None else None,

        #         gaussian_covariances=repeat(covariances, "b g i j -> (b v) g i j", v=n) if covariances is not None else None,
        #         scale_invariant = False,
        #         use_sh= self.use_sh,
        #     )
        #     colors = rearrange(colors,'(bs n) c h w -> bs n c h w',bs=bs) # (b v c h w)
        #     rendered_depth = rearrange(rendered_depth,'(bs n) c h w -> bs n c h w',bs=bs).squeeze(2) # (b v h w)        
        else:
            raise ValueError(f"Unsupported renderer type: {self.renderer_type}.")
            
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
