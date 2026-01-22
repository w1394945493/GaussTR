import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange,repeat

from mmdet3d.registry import MODELS
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from .utils import flatten_bsn_forward
from ..geometry import get_world_rays
from .encoder.common.gaussians import build_covariance
from ..loss import lovasz_softmax_flat,CE_ssc_loss
from .decoder import rasterize_gaussians,render_cuda
from .utils import cam2world,rotmat_to_quat,get_covariance


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

@MODELS.register_module()
class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 num_layers=2,
                 activation='relu',
                 mode=None,
                 range=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4
        output_dim = output_dim or input_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation
        self.range = range
        self.mode = mode

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = getattr(F, self.activation)(
                layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.mode is not None:
            if self.mode == 'sigmoid':
                x = F.sigmoid(x)

            if self.mode == 'normalize':
                x = F.normalize(x,dim=-1)

            if self.mode == 'softplus':
                x = F.softplus(x)

            if self.range is not None:
                x = self.range[0] + (self.range[1] - self.range[0]) * x
        return x

@MODELS.register_module()
class GaussTRHead(BaseModule):

    def __init__(self,
                 regress_head,
                 opacity_head,
                 scale_head,
                 rot_head,
                 semantic_head,
                 rgb_head,
                 voxelizer,
                 loss_lpips,
                 near,
                 far,
                 ori_image_shape,
                 depth_limit=51.2,
                 use_sh = True,
                 background_color=[0.0, 0.0, 0.0],
                 num_classes = 18,
                 
                 with_empty = False,
                 empty_args=None,
                 
                 
                 manual_class_weight=[
                    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],

                 ):
        super().__init__()

        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.regress_head = MODELS.build(regress_head)
        self.opacity_head = MODELS.build(opacity_head)
        self.scale_head = MODELS.build(scale_head)
        self.rot_head = MODELS.build(rot_head)
        self.semantic_head = MODELS.build(semantic_head)
        self.rgb_head = MODELS.build(rgb_head)
        self.voxelizer = MODELS.build(voxelizer)

        self.near = near
        self.far = far
        self.ori_image_shape = ori_image_shape
        self.depth_limit = depth_limit

        self.use_sh = use_sh

        self.class_weights = torch.tensor(manual_class_weight)
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg')) # todo mmseg
        
        self.num_classes = num_classes
        self.lovasz_ignore = num_classes - 1
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
            self.empty_scalar = nn.Parameter(torch.full((1, num_empty), 10.0, dtype=torch.float)) # (1,N)
            
            # self.register_buffer('empty_opa', torch.ones(num_empty)[None, :]) # (1, N_empty)        
            init_opa_val = 0.1
            raw_val = torch.log(torch.tensor(init_opa_val / (1 - init_opa_val)))
            # self.empty_opa = nn.Parameter(torch.full((1, num_empty), raw_val, dtype=torch.float))    
            self.register_buffer('empty_opa', torch.full((1, num_empty), raw_val, dtype=torch.float))
        
        self.with_empty = with_empty
        
        
        
    def forward(self,
                x,
                ref_pts,
                image_shape,
                depth,
                rgb_gts,
                cam2img,
                cam2ego,
                img_aug_mat=None,
                occ_gts=None,
                mode='tensor',
                **kwargs):


        bs, n = cam2img.shape[:2]

        x = x.reshape(bs, n, *x.shape[1:]) # (b,v,300,256)
        deltas = self.regress_head(x)

        ref_pts = (deltas[..., :2] + inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid() # (b v 300 2)

        depth = depth.clamp(max=self.depth_limit) # todo (b v 504 896)
        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None],ref_pts.unsqueeze(2) * 2 - 1) # (b v 1 1 300)
        sample_depth = sample_depth[:, :, 0, 0, :, None] # (b v 1 1 n) -> (b v n 1)
        points = torch.cat([
            ref_pts * torch.tensor(image_shape[::-1]).to(x),
            sample_depth * (1 + deltas[..., 2:3])], -1) # gausstr预测的偏移量：x,y: 像素坐标系下偏移 z: 真实的深度(m)下的偏移
        
        means = cam2world(points, cam2img, cam2ego) # (b v g 3)
        # scales = self.scale_head(x) * self.scale_transform(sample_depth, cam2img[..., 0, 0]).clamp(1e-6) # (b v g 3)
        scales = self.scale_head(x)
        covariances = flatten_bsn_forward(get_covariance, scales, cam2ego[..., None, :3, :3]) # (b v g 3 3)
        rotations = flatten_bsn_forward(rotmat_to_quat, cam2ego[..., :3, :3]) # (b v g 4)
        rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        
        opacities = self.opacity_head(x) # (b v g 1)
        semantics =self.semantic_head(x) # (b v g n_cls) n_cls = 17
        rgbs = self.rgb_head(x) # (b v g 3)


        means3d = rearrange(means,'b v g c -> b (v g) c') # (b (v g) 3)
        opacities = rearrange(opacities,'b v g c -> b (v g) c').squeeze(-1) # (b (v g))
        semantics = rearrange(semantics,'b v g c -> b (v g) c') # (b (v g) n_cls)
        rgbs = rearrange(rgbs,'b v g c -> b (v g) c') # (b (v g) 3)
        
        scales = rearrange(scales,'b v g c -> b (v g) c') # (b (v g) 3)
        rotations = rearrange(rotations,'b v g c -> b (v g) c') # (b (v g) 4)
        
        covariances = rearrange(covariances,'b v g i j -> b (v g) i j') # (b (v g) 3 3)

        if self.with_empty:
            bs = means3d.shape[0] # 获取当前 Batch Size
            empty_mean = self.empty_mean.expand(bs, -1, -1)     # (B, n_bg, 3)
            empty_covs = self.empty_covs.expand(bs, -1, -1, -1) # (B, n_bg, 3, 3) 如果是矩阵
            empty_sem = self.empty_sem.expand(bs, -1, -1).clone()       # (B, n_bg, num_classes)
            
            empty_opa = torch.sigmoid(self.empty_opa).expand(bs, -1)       # (B, 1)  

            empty_scalar = F.softplus(self.empty_scalar) # 确保背景增益＞0
            empty_sem[..., self.lovasz_ignore] += empty_scalar.expand(bs, -1)

            
            features = torch.cat([semantics, empty_sem], dim=1)
              
            means3d = torch.cat([means3d, empty_mean], dim=1)
            covariances = torch.cat([covariances,empty_covs],dim=1)
            
            opacities = torch.cat([opacities,empty_opa],dim=1)

        # todo occ占用预测
        density, grid_feats = self.voxelizer(means3d, covariances, opacities, features)
        
        if mode == 'predict':
            occ_preds = grid_feats.argmax(-1) # (b 200 200 16 18) -> (b 200 200 16) 与先softmax，在argmax结果是一致的
            outputs = [{
                'occ_pred': occ_preds,
            }]
            return outputs

        losses = {}

        probs = rearrange(grid_feats,"b H W D C -> b C (H W D)")
        target = rearrange(occ_gts,"b H W D ->b (H W D)").long()
        losses['loss_ce'] = 10.0 * CE_ssc_loss(probs, target, self.class_weights.type_as(probs), ignore_index=255)

        inputs = torch.softmax(probs, dim=1).transpose(1,2).flatten(0,1)
        target = occ_gts.flatten()
        ignore = self.lovasz_ignore
        valid = (target != ignore)
        probas = inputs[valid]
        labels = target[valid]
        losses['loss_lovasz'] = 1.0 * lovasz_softmax_flat(probas, labels)
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
    
    def scale_transform(self, depth, focal, multiplier=7.5):
        return depth * multiplier / focal.reshape(*depth.shape[:2], 1, 1) # todo 将深度转换到物理尺度
