import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from einops import rearrange

from .utils import (apply_to_items, generate_grid, get_covariance,
                    quat_to_rotmat, unbatched_forward)





@MODELS.register_module()
class GaussianVoxelizer(nn.Module):

    def __init__(self,
                 vol_range,
                 voxel_size,
                 num_classes,
                 scale_multiplier=3,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 covariance_thresh=0):
        super().__init__()
        self.voxel_size = voxel_size # todo 0.4

        center_h, center_w, center_d = (vol_range[0] + vol_range[3]) / 2,(vol_range[1] + vol_range[4]) / 2,(vol_range[2] + vol_range[5]) / 2
        range_h, range_w, range_d = abs(vol_range[0] - vol_range[3]), abs(vol_range[1] - vol_range[4]), abs(vol_range[2] - vol_range[5])

        vol_range = torch.tensor(vol_range) # todo [-40 -40 -1 40 40 5.4]
        self.register_buffer('vol_range', vol_range)

        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        grid_coords = generate_grid(self.grid_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh


        self.num_classes = num_classes
        self.empty_label = num_classes - 1
        self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0) # todo nn.Parameter: 可训练参数

        self.register_buffer('empty_mean', torch.tensor([center_h, center_w, center_d])[None, :]) # todo register_buffer: 跟着模型走，但不训练的状态量
        self.register_buffer('empty_scale', torch.tensor([range_h, range_w, range_d])[None, :])
        self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, :])
        self.register_buffer('empty_cov', torch.diag(torch.tensor([range_h**2, range_w**2, range_d**2]))[None, :])
        self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, :])
        self.register_buffer('empty_opa', torch.ones(1)[None, :])

        pc_min = vol_range[:3].tolist()
        self.pc_min = pc_min
        H,W,D,_ = grid_coords.shape
        self.voxel_range = [H,W,D]
        import local_aggregate
        self.aggregator = local_aggregate.LocalAggregator(scale_multiplier=scale_multiplier, # todo scale_multiplier: 3 (±3σ范围)
                                                          H=H, W=W, D=D, # todo 体素空间 H:200 W:200 D:16
                                                          pc_min=pc_min, # todo [-40 -40 -1]
                                                          grid_size=voxel_size) # todo 体素尺寸: 0.4

    @unbatched_forward
    def forward(self,
                means3d,
                opacities,
                covariances=None,
                scales=None,
                rotations=None,
                **kwargs):
        if covariances is None:
            covariances = get_covariance(scales, quat_to_rotmat(rotations))

        gaussians = dict(
            means3d=means3d,
            opacities=opacities,
            covariances=covariances,
            scales = scales,
            **kwargs)

        '''
        import numpy as np
        # 保存为 numpy
        np.save("means3d_query.npy", means3d.cpu().numpy())
        np.save("vol_range.npy", self.vol_range.cpu().numpy())

        print("保存完毕: means3d.npy, vol_range.npy")
        '''
        # todo 对高斯点进行过滤
        if self.filter_gaussians:
            mask = opacities.squeeze(1) > self.opacity_thresh
            for i in range(3):
                # todo 剔除在体素空间范围之外的点
                mask &= (means3d[:, i] >= self.vol_range[i]) & (
                    means3d[:, i] <= self.vol_range[i + 3])
                # -------------------------------------------------#
                # 体素空间范围（float 坐标）避免超出范围
                voxel_coord = (means3d[:, i] - self.pc_min[i]) / self.voxel_size
                mask &= (voxel_coord >= 0) & (voxel_coord < self.voxel_range[i])

            if self.covariance_thresh > 0:
                cov_diag = torch.diagonal(covariances, dim1=1, dim2=2)
                mask &= ((cov_diag.min(1)[0] * 6) > self.covariance_thresh)
            gaussians = apply_to_items(lambda x: x[mask], gaussians)

        # todo 将离散的3D高斯分布转换成体素
        return self.splat_into_3d(
            self.grid_coords, # todo 网格坐标：(L W H 3)
            **gaussians,
            vol_range=self.vol_range,
            voxel_size=self.voxel_size)

    def splat_into_3d(self,
                    grid_coords,
                    means3d,
                    opacities,
                    scales,
                    covariances,
                    vol_range,
                    voxel_size,
                    features=None,
                    eps=1e-6):

        H,W,D = self.grid_shape
        grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                            device=grid_coords.device)
        grid_feats = torch.zeros((*grid_coords.shape[:-1],self.num_classes),
                        device=grid_coords.device)

        features = torch.cat([features,torch.zeros_like(features[...,:1])],dim=-1) # 17 -> 18
        empty_sem = self.empty_sem.clone()
        empty_sem[..., self.empty_label] += self.empty_scalar
        features = torch.cat([features,empty_sem],dim=0)

        means3d = torch.cat([means3d, self.empty_mean], dim=0)
        covariances = torch.cat([covariances,self.empty_cov],dim=0)
        opacities = torch.cat([opacities,self.empty_opa],dim=0)
        scales = torch.cat([scales,self.empty_scale],dim=0)



        if means3d.size(0) == 0:
            return grid_density, grid_feats

        sampled_xyz = rearrange(grid_coords,"H W D xyz -> (H W D) xyz").unsqueeze(0)
        means3d = means3d.unsqueeze(0)
        covariances_inv = covariances.inverse().unsqueeze(0)
        opacities = opacities.squeeze(-1).unsqueeze(0)
        features = features.unsqueeze(0)
        scales = scales.unsqueeze(0)
        semantics = self.aggregator(sampled_xyz,means3d,opacities,features,scales,covariances_inv)
        grid_feats = rearrange(semantics,"(H W D) dim -> H W D dim",H=H,W=W,D=D) # todo (200 200 16 n_classes)
        return grid_density, grid_feats

        # sigmas = torch.sqrt(covariances.diagonal(dim1=-2, dim2=-1))
        # factors = 3 * torch.tensor([-1, 1]).to(sigmas.device).view(2, 1).expand(2, means3d.size(0)).T
        # bounds_all = means3d[:, None, :] + factors[:, :, None] * sigmas[:, None, :]


        # bounds_all = bounds_all.clamp(vol_range[:3], vol_range[3:])  # (N,2,3)
        # bounds_indices = ((bounds_all - vol_range[:3]) / voxel_size).long()
        # covariances_inv = covariances.inverse()
        # for g in range(means3d.size(0)):
        #     bounds = bounds_all[g] # (2 3)
        #     if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
        #             ((bounds < vol_range[None, 3:]).max(0).values.min())):
        #         continue
        #     bounds = bounds_indices[g].int().tolist() # (2 3) -> list[[x_start y_start z_start] [x_end y_end z_end]]
        #     slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)]) # slices: 三元组：((x_start,x_end,None),(y_start,y_end,None),(z_start,z_end,None)) None表示默认步长为1

        #     diff = grid_coords[slices] - means3d[g]
        #     cov_inv = covariances_inv[g] # (3 3)
        #     maha_dist = (diff.unsqueeze(-2) @ cov_inv
        #                 @ diff.unsqueeze(-1)).squeeze(-1) # (x y z 1 3) @ (3 3) @ (x y z 3 1)
        #     density = opacities[g] * torch.exp(-0.5 * maha_dist)
        #     grid_density[slices] += density
        #     grid_feats[slices] += density * features[g]

        # grid_feats /= grid_density.clamp(eps)
        # return grid_density, grid_feats