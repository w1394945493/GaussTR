import numpy as np
import torch, torch.nn as nn

from mmdet3d.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix
from .gaussian_voxelizer import GaussSplatting3D

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
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0) # todo nn.Parameter: 可训练参数
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :]) # todo register_buffer: 跟着模型走，但不训练的状态量
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
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
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c # todo (b 25600 17)
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        
        if self.with_emtpy: # todo True
            assert opacities.shape[-1] == self.num_classes - 1 # todo self.num_classes: 18
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else: # todo 拼接了一个全0列
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1) # todo cat -> (1 25600 18)
            means = torch.cat([means, self.empty_mean], dim=1) # todo self.empty_mean: (1 1 3) 内容是(0.,0.,-1)
            scales = torch.cat([scales, self.empty_scale], dim=1) # todo self.empty_scale: (1 1 3) 内容是(100 100 8)
            rotations = torch.cat([rotations, self.empty_rot], dim=1) # todo self.empty_rot: (1 1 4) 内容是(1 0 0 0)
            empty_sem = self.empty_sem.clone() # (1 1 18) # todo self.empty_sem: (1 1 18) 内容全为0
            empty_sem[..., self.empty_label] += self.empty_scalar # todo self.empty_scaler: 10.2920 self.empty_label: 17
            opacities = torch.cat([opacities, empty_sem], dim=1) # todo
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1) # todo self.empty_opa: (1 1 1) 内容为1



        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        covs = torch.matmul(M.transpose(-1, -2), M)
        return means, origi_opa, opacities, scales, covs
    
        # CovInv =covs.cpu().inverse().cuda() # b, g, 3, 3 # todo g = 25600 + 1 = 25601
        # return means, origi_opa, opacities, scales, CovInv

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
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            means3d, opacities, features, scales, covs = self.prepare_gaussian_args(gaussians)
            
            means3d = means3d.squeeze(0)
            opacities = opacities.squeeze(0)
            features = features.squeeze(0)
            covs = covs.squeeze(0) 
            
            _, pred_feats = GaussSplatting3D.apply(
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

