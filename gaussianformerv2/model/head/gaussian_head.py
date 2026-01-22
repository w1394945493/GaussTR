import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix

from .gaussian_voxelizer import GaussSplatting3D,GaussSplatting3DCuda
from ..encoder.common.gaussians import build_covariance  


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self,
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        voxel_kwargs = None,
        dataset_type='nusc',
        empty_label=17,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        combine_geosem=False,
        **kwargs,
    ):
        super().__init__(init_cfg)

        self.num_classes = num_classes # todo 18
        
        voxel_size = voxel_kwargs['voxel_size']
        vol_min = torch.tensor(voxel_kwargs['vol_min'])
        vol_max = torch.tensor(voxel_kwargs['vol_max'])
        vol_range = torch.cat([vol_min, vol_max]) 

        # 网格形状 (200, 200, 16)
        dim_x = int((vol_max[0] - vol_min[0]) / voxel_size)
        dim_y = int((vol_max[1] - vol_min[1]) / voxel_size)
        dim_z = int((vol_max[2] - vol_min[2]) / voxel_size)
        grid_shape = (dim_x, dim_y, dim_z)  
               
        self.register_buffer('vol_range', vol_range)        
        self.voxel_size = voxel_size
        self.grid_shape = grid_shape

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
            overlap_factor =6.0
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
            self.empty_scalar = nn.Parameter(torch.full((1, num_empty), 1.0, dtype=torch.float)) # (1,N)
            
            # self.register_buffer('empty_opa', torch.ones(num_empty)[None, :]) # (1, N_empty)        
            init_opa_val = 0.1
            raw_val = torch.log(torch.tensor(init_opa_val / (1 - init_opa_val)))
            self.empty_opa = nn.Parameter(torch.full((1, num_empty), raw_val, dtype=torch.float))    
            # self.register_buffer('empty_opa', torch.full((1, num_empty), raw_val, dtype=torch.float))
            
        
        
        self.with_empty = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type: # todo random_1
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means3d = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        features = gaussians.semantics # b, g, c # todo (b 25600 17)
        opacities = gaussians.opacities.squeeze(-1) # b, g
        
        bs, g, _ = means3d.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means3d.dtype, device=means3d.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        covariances = torch.matmul(M.transpose(-1, -2), M)        
        
        if self.with_empty:
            empty_mean = self.empty_mean.expand(bs, -1, -1)     # (B, n_bg, 3)
            empty_covs = self.empty_covs.expand(bs, -1, -1, -1) # (B, n_bg, 3, 3) 如果是矩阵
            empty_sem = self.empty_sem.clone().expand(bs, -1, -1)       # (B, n_bg, num_classes)
            
            empty_opa = torch.sigmoid(self.empty_opa).expand(bs, -1)       # (B, 1)         
            
            empty_scalar = F.softplus(self.empty_scalar) # 确保背景增益＞0
            empty_sem[..., -1] += empty_scalar.expand(bs, -1)
            
            assert features.shape[-1] == self.num_classes - 1
            features = torch.cat([features, torch.zeros_like(features[..., :1])], dim=-1)
            
            features = torch.cat([features, empty_sem], dim=1)
              
            means3d = torch.cat([means3d, empty_mean], dim=1)
            covariances = torch.cat([covariances,empty_covs],dim=1)
            
            opacities = torch.cat([opacities,empty_opa],dim=1)

        return means3d, opacities, features,  covariances
    

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        prediction = []
        bin_logits = []
        density = []
        # todo ----------------------------------------------------------#
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            means3d, opacities, features, covs = self.prepare_gaussian_args(gaussians)
            
            means3d = means3d.squeeze(0)
            opacities = opacities.squeeze(0)
            features = features.squeeze(0)
            covs = covs.squeeze(0) 
            
            
            # grid_density, pred_feats = GaussSplatting3D.apply(
            #             means3d, covs, opacities, features, 
            #             self.vol_range, self.voxel_size, self.grid_shape
            #         ) # todo pred_feats: (200, 200, 16, 18)
            
            grid_density, pred_feats = GaussSplatting3DCuda.apply(
                        means3d, covs, opacities, features, 
                        self.vol_range, self.voxel_size, self.grid_shape
                    ) # todo pred_feats: (200, 200, 16, 18)

            logits = pred_feats.permute(3, 0, 1, 2).flatten(1).unsqueeze(0) # (200, 200, 16, 18) -> (18,640000) -> (1,18,6400)
            prediction.append(logits)
            final_prediction = logits.argmax(dim=1)
        
                        
        return {
            'pred_occ': prediction,
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation]
        }

