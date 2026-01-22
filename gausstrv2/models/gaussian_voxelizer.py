import torch
import torch.nn as nn
import triton
import triton.language as tl
from mmdet3d.registry import MODELS

import gauss_splatting_cuda
from .utils import unbatched_forward,apply_to_items

class GaussSplatting3DCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3d, covs, opacities, features, vol_range, voxel_size, grid_shape):
        """
        means3d: [N, 3]
        covs: [N, 3, 3]
        opacities: [N]
        features: [N, D]
        vol_range: [3] (min_x, min_y, min_z)
        voxel_size: float
        grid_shape: tuple (dim_x, dim_y, dim_z)
        """
        N = means3d.shape[0]
        n_dims = features.shape[1]
        device = means3d.device

        # 1. 预处理：计算协方差逆和半径 (保持 Triton 中的逻辑)
        # 注意：这里需要确保数据在内存上是连续的
        inv_covs = torch.inverse(covs).contiguous()
        
        # 计算半径 (取协方差对角线方差，按 3 sigma 原则)
        variances = torch.diagonal(covs, dim1=-2, dim2=-1)
        radii = 3.0 * torch.sqrt(variances).contiguous()

        # 2. 初始化输出网格
        # grid_density: [dim_x, dim_y, dim_z]
        # grid_feats: [dim_x, dim_y, dim_z, n_dims]
        grid_density = torch.zeros(grid_shape, device=device, dtype=torch.float32)
        grid_feats = torch.zeros((*grid_shape, n_dims), device=device, dtype=torch.float32)
        # grid_feats[..., -1] = 1e-5
        
        # 3. 调用 CUDA 前向传播
        # 注意传参顺序要和 splatting_cuda.cpp 中的 m.def("forward", ...) 一致
        gauss_splatting_cuda.forward(
            means3d.contiguous(),
            inv_covs.view(N, 9).contiguous(),
            opacities.contiguous(),
            radii.contiguous(),
            features.contiguous(),
            grid_density,
            grid_feats,
            float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
            float(voxel_size)
        )

        # 4. 归一化特征
        eps = 1e-6
        grid_feats_norm = grid_feats / grid_density.unsqueeze(-1).clamp(min=eps)

        # 5. 保存给反向传播用的变量
        ctx.save_for_backward(means3d, inv_covs, opacities, radii, features, grid_density, grid_feats_norm)
        ctx.vol_range = vol_range
        ctx.voxel_size = voxel_size
        ctx.eps = eps

        return grid_density, grid_feats_norm

    @staticmethod
    def backward(ctx, grad_grid_density, grad_grid_feats):
        """
        grad_grid_density: [dim_x, dim_y, dim_z]
        grad_grid_feats: [dim_x, dim_y, dim_z, n_dims]
        """
        # 1. 恢复前向传播的数据
        means3d, inv_covs, opacities, radii, features, grid_density, grid_feats_norm = ctx.saved_tensors
        vol_range = ctx.vol_range
        voxel_size = ctx.voxel_size
        eps = ctx.eps
        N = means3d.shape[0]
        n_dims = features.shape[1]

        # 2. 初始化梯度张量 (初始化为 0)
        grad_means = torch.zeros_like(means3d)
        grad_inv_covs = torch.zeros((N, 9), device=means3d.device)
        grad_opacities = torch.zeros_like(opacities)
        grad_features = torch.zeros_like(features)

        # 3. 调用 CUDA 反向传播
        # 注意传参顺序要和 splatting_cuda.cpp 中的 m.def("backward", ...) 一致
        gauss_splatting_cuda.backward(
            grad_features,
            grad_opacities,
            grad_means,
            grad_inv_covs,
            grid_density,
            grid_feats_norm,
            grad_grid_density.contiguous(),
            grad_grid_feats.contiguous(),
            means3d.contiguous(),
            inv_covs.view(N, 9).contiguous(),
            opacities.contiguous(),
            radii.contiguous(),
            features.contiguous(),
            float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
            float(voxel_size),
            float(eps)
        )

        # 4. 将 inv_covs 的梯度转回 covs 的梯度
        # 根据矩阵求导法则: d(inv(A)) = -inv(A) @ d(A) @ inv(A)
        # 所以 d(L)/d(A) = -inv(A).T @ d(L)/d(inv(A)) @ inv(A).T
        inv_covs_reshaped = inv_covs.view(N, 3, 3)
        grad_inv_covs_reshaped = grad_inv_covs.view(N, 3, 3)
        
        # 对于对称矩阵 A^-1: dL/dA = -A^-1 @ (dL/dA^-1) @ A^-1
        grad_covs = -torch.bmm(torch.bmm(inv_covs_reshaped, grad_inv_covs_reshaped), inv_covs_reshaped)

        # 返回的梯度顺序必须和 forward 的参数顺序一一对应，不需要梯度的参数返回 None
        # means3d, covs, opacities, features, vol_range, voxel_size, grid_shape
        return grad_means, grad_covs, grad_opacities, grad_features, None, None, None




@MODELS.register_module()
class GaussianVoxelizer(nn.Module):
    def __init__(self,
                 vol_range,
                 voxel_size,
                 filter_gaussians=True,
                 opacity_thresh=0.6,
                 covariance_thresh=1.5e-2,                                
                 ):
        super().__init__()
        self.voxel_size = voxel_size                 
        vol_range = torch.tensor(vol_range)
        self.register_buffer('vol_range', vol_range)
        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        self.filter_gaussians = filter_gaussians
    
    @unbatched_forward
    def forward(self,means3d,covariances,opacities,features):
        
        gaussians = dict(
            means3d=means3d, # (n,3)
            covs=covariances,
            opacities=opacities,
            features=features,)
        '''
        import numpy as np
        # 保存为 numpy
        np.save("means3d.npy", means3d.detach().cpu().numpy())
        np.save("vol_range.npy", self.vol_range.detach().cpu().numpy())

        print("保存完毕: means3d.npy, vol_range.npy")
        '''
        
        if self.filter_gaussians: # todo 剔除一下透明度过低的高斯点(多bs时，可能是为对齐bs的填充高斯)
            mask = opacities > 1e-6
            for i in range(3):
                mask &= (means3d[:, i] >= self.vol_range[i]) & (
                        means3d[:, i] <= self.vol_range[i + 3])
            gaussians = apply_to_items(lambda x: x[mask], gaussians)
            
        # todo cuda 版本
        density, grid_feats = GaussSplatting3DCuda.apply(
            gaussians['means3d'], 
            gaussians['covs'], 
            gaussians['opacities'], 
            gaussians['features'], # (n dims)
            self.vol_range, 
            self.voxel_size, 
            self.grid_shape
        )
        return density, grid_feats    