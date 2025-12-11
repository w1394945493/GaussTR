import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from mmdet3d.registry import MODELS

from torch_scatter import scatter_add, scatter_max

from tqdm import tqdm

from .utils import (apply_to_items, generate_grid, get_covariance,
                    quat_to_rotmat, unbatched_forward)

def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)



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

        # # todo --------------------------------------------#
        # # todo 参考GaussianFormer中的体素化模块
        # sampled_xyz = rearrange(grid_coords,"H W D xyz -> (H W D) xyz").unsqueeze(0) # todo (1 200x200x16 3)
        # cov_inv = covariances.inverse().unsqueeze(0) # todo (1 n 3 3)
        # means = means3d.unsqueeze(0) # todo (1 n 3)
        # opacities = opacities.unsqueeze(0).squeeze(-1) # todo (1 n)
        # features = features.unsqueeze(0) # todo (1 n n_classes)
        # scales = torch.sqrt(covariances.diagonal(dim1=1, dim2=2)).unsqueeze(0) # todo (1 n 3)

        # semantics = self.aggregator(
        #     sampled_xyz,
        #     means,
        #     opacities,
        #     features,
        #     scales,
        #     cov_inv,
        # ) # todo (200x200x16 n_classes)
        # H,W,D = self.grid_shape
        # grid_feats = rearrange(semantics,"(H W D) dim -> H W D dim",H=H,W=W,D=D) # todo (200 200 16 n_classes)

        # return grid_feats,None

        # todo ----------------------------------#
        # todo 参照AnySplat中的体素化操作
        voxel_indices = ((means3d - vol_range[:3]) / voxel_size).round().int() # 转为体素索引
        voxel_indices = torch.clamp(voxel_indices, min=torch.tensor(0, device=voxel_indices.device), max=torch.tensor([H-1, W-1, D-1], device=voxel_indices.device))

        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        ) # todo unique_voxels: 唯一的体素索引(去重后的结果) inverse_indices: 原始的体素索引对应到unique_voxels中的索引，返回和voxel_indices长度一样的张量，表示每个voxel_indices中每个元素在unique_voxels中的位置

        conf_flat = opacities.flatten() # todo 使用透明度作为置信度依据
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0) # todo scatter_max: 按照指定的索引对张量进行聚合操作 scatter_max(src,index,dim): src: 你想要聚合的张量 index：和src相同大小的张量 dim：制定者那个维度聚合
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices]) # todo 计算各个位置的透明度 - 该位置最大透明度 的对数值
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0) # todo 按照指定索引进行求和

        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1) # todo 作为权重 (num_gaussians,1)
        weighted_feats = features.squeeze(1) * weights # todo (num_gaussians,num_classes)

        feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )

        H,W,D = self.grid_shape # 200 200 16
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                device=grid_coords.device)
        grid_feats = rearrange(grid_feats,'H W D C -> (H W D) C')
        flat_indices = unique_voxels[:, 0] * (W * D) + unique_voxels[:, 1] * D + unique_voxels[:, 2]
        grid_feats = scatter_add(feats, flat_indices, dim=0, out=grid_feats)

        grid_feats = rearrange(grid_feats,'(H W D) C -> H W D C',H=H,W=W,D=D)

        return grid_feats, None

        # todo GaussTR
        # todo 将一组3D高斯点撒进体素网格，累加每个体素出的密度，并将特征按密度加权累计后归一化输出
        grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                                device=grid_coords.device) # todo (200 200 16 1)

        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                device=grid_coords.device) # todo (200 200 16 n_classes)
        sigmas = torch.sqrt(covariances.diagonal(dim1=-2, dim2=-1))
        factors = 3 * torch.tensor([-1, 1]).to(sigmas.device).view(2, 1).expand(2, means3d.size(0)).T  # (num_gaussian, 2)
        bounds_all = means3d[:, None, :] + factors[:, :, None] * sigmas[:, None, :]
        bounds_all = bounds_all.clamp(vol_range[:3], vol_range[3:])  # (N,2,3)
        bounds_indices = ((bounds_all - vol_range[:3]) / voxel_size).long()
        covariances_inv = covariances.inverse()

        # todo 逐点循环，高斯点数量很多时会很慢(待优化): 仍然速度很慢：6x112x192个高斯点处理需几分钟以上
        for g in tqdm(range(means3d.size(0))): # todo means3d: (n,3) n: 高斯点数量
            # 计算sigma、bounding box(用1D标准差近似球/椭球支撑域)
            # sigma = torch.sqrt(torch.diag(covariances[g])) # 提取协方差的对焦元素 (3)
            # factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma) # 取 ±3σ作为截断范围(≈99.7%高斯质量)
            # bounds = means3d[g, None] + factor * sigma[None] # 得到高斯在世界坐标系下的轴向最小/最大坐标 (2,3) ±3σ

            bounds = bounds_all[g] # (2 3)
            if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                    ((bounds < vol_range[None, 3:]).max(0).values.min())):
                continue # 若高斯的3sigma区间完全落在体素体积外，则跳过该高斯

            # 找到与该高斯相交的voxels区域
            # bounds = bounds.clamp(vol_range[:3], vol_range[3:]) # 把bounds限制到体积边界范围内
            # bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist() # 转换成体素索引 idx = (x-x_min) / voxel_size

            bounds = bounds_indices[g].int().tolist() # (2 3) -> list[[x_start y_start z_start] [x_end y_end z_end]]
            slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)]) # slices: 三元组：((x_start,x_end,None),(y_start,y_end,None),(z_start,z_end,None)) None表示默认步长为1

            # 计算每个体素中心与高斯中心的马氏距离
            diff = grid_coords[slices] - means3d[g] # (x,y,z,3) -(3) = (x y z 3)
            # cov_inv = covariances[g].inverse()
            cov_inv = covariances_inv[g] # (3 3)
            maha_dist = (diff.unsqueeze(-2) @ cov_inv
                        @ diff.unsqueeze(-1)).squeeze(-1) # (x y z 1 3) @ (3 3) @ (x y z 3 1)

            density = opacities[g] * torch.exp(-0.5 * maha_dist) # todo 并使用高斯权重
            # 向3D网格累计密度，区域内所有体素都加上该高斯的权重
            grid_density[slices] += density

            if features is not None:
                grid_feats[slices] += density * features[g]

        if features is not None:
            grid_feats /= grid_density.clamp(eps)

        return grid_feats, grid_density