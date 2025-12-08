import torch
import torch.nn as nn
from einops import rearrange
from mmdet3d.registry import MODELS

from .utils import (apply_to_items, generate_grid, get_covariance,
                    quat_to_rotmat, unbatched_forward)





@MODELS.register_module()
class GaussianVoxelizer(nn.Module):

    def __init__(self,
                 vol_range,
                 voxel_size,
                 scale_multiplier=3,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 covariance_thresh=0):
        super().__init__()
        self.voxel_size = voxel_size # todo 0.4
        vol_range = torch.tensor(vol_range) # todo [-40 -40 -1 40 40 5.4]
        self.register_buffer('vol_range', vol_range)

        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        grid_coords = generate_grid(self.grid_shape, offset=0.5) # todo 网格点
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh

        # todo (wys) 参考GaussianFormer 高斯到体素化
        pc_min = vol_range[:3].tolist()
        H,W,D,_ = grid_coords.shape
        import local_aggregate
        self.aggregator = local_aggregate.LocalAggregator(scale_multiplier=scale_multiplier,
                                                          H=H, W=W, D=D,
                                                          pc_min=pc_min,
                                                          grid_size=voxel_size)
        self.aggregator.requires_grad_(False)

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
                mask &= (means3d[:, i] >= self.vol_range[i]) & (
                    means3d[:, i] <= self.vol_range[i + 3])
            if self.covariance_thresh > 0:
                cov_diag = torch.diagonal(covariances, dim1=1, dim2=2)
                mask &= ((cov_diag.min(1)[0] * 6) > self.covariance_thresh)
            gaussians = apply_to_items(lambda x: x[mask], gaussians)

        # todo 高斯到体素投影
        return self.splat_into_3d(
            self.grid_coords, # todo 网格坐标：(L W H 3)
            **gaussians,
            vol_range=self.vol_range,
            voxel_size=self.voxel_size)


    def splat_into_3d(self,
                    grid_coords,
                    means3d,
                    opacities,
                    covariances,
                    vol_range,
                    voxel_size,
                    features=None,
                    chunk = 128,
                    eps=1e-6):

        # todo --------------------------------------------#
        # todo 参考GaussianFormer中的体素化模块
        sampled_xyz = rearrange(grid_coords,"H W D xyz -> (H W D) xyz").unsqueeze(0) # todo (1 200x200x16 3)
        cov_inv = covariances.inverse().unsqueeze(0) # todo (1 n 3 3)
        means = means3d.unsqueeze(0) # todo (1 n 3)
        opacities = opacities.unsqueeze(0).squeeze(-1) # todo (1 n)
        features = features.unsqueeze(0) # todo (1 n n_classes)
        scales = torch.sqrt(covariances.diagonal(dim1=1, dim2=2)).unsqueeze(0) # todo (1 n 3)

        # semantics = self.aggregator(
        #     sampled_xyz,
        #     means,
        #     opacities,
        #     features,
        #     scales,
        #     cov_inv,
        # ) # todo (200x200x16 n_classes)

        b,n,c = features.shape # b=1, n=Gaussians, c=n_classes
        V = sampled_xyz.shape[1]            # V = H*W*D
        grid_feats = torch.zeros(V, c, device=features.device)
        for i in range(0, n, chunk):
            s = slice(i, i + chunk)

            # 分块处理 Gaussians
            means_chunk     = means[:, s]        # (1, chunk, 3)
            opac_chunk      = opacities[:, s]    # (1, chunk)
            feats_chunk     = features[:, s]     # (1, chunk, C)
            scales_chunk    = scales[:, s]       # (1, chunk, 3)
            cov_inv_chunk   = cov_inv[:, s]      # (1, chunk, 3, 3)

            # 调用原 aggregator
            sem = self.aggregator(
                sampled_xyz,
                means_chunk,
                opac_chunk,
                feats_chunk,
                scales_chunk,
                cov_inv_chunk,
            )  # (V, C)

            grid_feats += sem # 累加到 (V, C)
        H,W,D = self.grid_shape
        grid_feats = rearrange(grid_feats,"(H W D) dim -> H W D dim",H=H,W=W,D=D) # todo (200 200 16 n_classes)
        return grid_feats


        # todo 将一组3D高斯点撒进体素网格，累加每个体素出的密度，并将特征按密度加权累计后归一化输出
        grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                                device=grid_coords.device) # todo (200 200 16 1)
        if features is not None:
            grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                    device=grid_coords.device) # todo (200 200 16 n_classes)


        # todo 逐点循环，高斯点数量很多时会很慢(待优化)
        for g in range(means3d.size(0)): # todo means3d: (n,3) n: 高斯点数量
            # todo 计算sigma、bounding box(用1D标准差近似球/椭球支撑域)
            sigma = torch.sqrt(torch.diag(covariances[g])) # todo 提取协方差的对焦元素 (3)
            factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma) # todo 取 ±3σ作为截断范围(≈99.7%高斯质量)
            bounds = means3d[g, None] + factor * sigma[None] # todo 得到高斯在世界坐标系下的轴向最小/最大坐标 (2,3) ±3σ
            if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                    ((bounds < vol_range[None, 3:]).max(0).values.min())):
                continue # todo 若高斯的3sigma区间完全落在体素体积外，则跳过该高斯

            # todo 计算切片范围：找到与该高斯相交的voxels区域
            bounds = bounds.clamp(vol_range[:3], vol_range[3:]) # todo 把bounds限制到体积边界范围内
            bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist() # todo 转换成体素索引 idx = (x-x_min) / voxel_size
            slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])

            # todo 计算该高斯对每个voxel的贡献：计算每个体素中心与高斯中心的马氏距离
            diff = grid_coords[slices] - means3d[g] # (1 1 1 3)
            maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse() # todo 这里使用的协方差的逆
                        @ diff.unsqueeze(-1)).squeeze(-1) # todo 计算每个体素中心与高斯中心的马氏距离
            density = opacities[g] * torch.exp(-0.5 * maha_dist) # todo 并使用高斯权重
            # todo 向3D网格累计密度，区域内所有体素都加上该高斯的权重
            grid_density[slices] += density

            if features is not None:
                grid_feats[slices] += density * features[g]

        if features is None:
            return grid_density
        grid_feats /= grid_density.clamp(eps)
        return grid_density, grid_feats