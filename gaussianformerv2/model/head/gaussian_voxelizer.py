import time

import torch
import triton
import triton.language as tl
import gauss_splatting_cuda


def splat_into_3d(grid_coords, # 体素网格坐标 (N,3)
                  means3d,     # 高斯点位置 (N,3) N=6x300 6个相机，300个query高斯
                  opacities,   # 透明度  (N)
                  covariances, # 协方差  (N,3,3)
                  vol_range,   # 体素范围 
                  voxel_size, # 体素尺寸 0.4
                  features=None,
                  eps=1e-6):
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                               device=grid_coords.device)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                 device=grid_coords.device) # (x,y,z,n_cls)
    for g in range(means3d.size(0)):
        
        sigma = torch.sqrt(torch.diag(covariances[g]))
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
        
        bounds = bounds.clamp(vol_range[:3], vol_range[3:])
        bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist()
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])

        diff = grid_coords[slices] - means3d[g]
        maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
                     @ diff.unsqueeze(-1)).squeeze(-1)
        density = opacities[g] * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density
        if features is not None:
            grid_feats[slices] += density * features[g]
    if features is None:
        return grid_density
    grid_feats /= grid_density.clamp(eps)
    return grid_density, grid_feats

@triton.jit
def _splat_fwd_kernel_opt(
    means_ptr, inv_covs_ptr, opacities_ptr,
    radii_ptr, features_ptr,
    grid_density_ptr, grid_feats_ptr,
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size,
    N_gaussians,n_dims, 
    dim_x, dim_y, dim_z,
):
    idx = tl.program_id(0)
    if idx >= N_gaussians: return
    
    # 均值x,y,z
    mx = tl.load(means_ptr + idx * 3 + 0)  
    my = tl.load(means_ptr + idx * 3 + 1)  
    mz = tl.load(means_ptr + idx * 3 + 2)

    # 协方差逆
    c_ptr = inv_covs_ptr + idx * 9
    c0, c1, c2 = tl.load(c_ptr + 0), tl.load(c_ptr + 1), tl.load(c_ptr + 2)
    c3, c4, c5 = tl.load(c_ptr + 3), tl.load(c_ptr + 4), tl.load(c_ptr + 5)
    c6, c7, c8 = tl.load(c_ptr + 6), tl.load(c_ptr + 7), tl.load(c_ptr + 8)
    
    # 透明度
    opac = tl.load(opacities_ptr + idx)
    # 半径
    # 加载三个轴向的独立半径
    rx = tl.load(radii_ptr + idx * 3 + 0)
    ry = tl.load(radii_ptr + idx * 3 + 1)
    rz = tl.load(radii_ptr + idx * 3 + 2)
  
    ix_s = (mx - rx - vol_min_x) / voxel_size
    ix_e = (mx + rx - vol_min_x) / voxel_size
    ix_start = tl.maximum(0, ix_s.to(tl.int32))
    ix_end   = tl.minimum(dim_x, ix_e.to(tl.int32) + 1)

    iy_s = (my - ry - vol_min_y) / voxel_size
    iy_e = (my + ry - vol_min_y) / voxel_size
    iy_start = tl.maximum(0, iy_s.to(tl.int32))
    iy_end   = tl.minimum(dim_y, iy_e.to(tl.int32) + 1)

    iz_s = (mz - rz - vol_min_z) / voxel_size
    iz_e = (mz + rz - vol_min_z) / voxel_size
    iz_start = tl.maximum(0, iz_s.to(tl.int32))
    iz_end   = tl.minimum(dim_z, iz_e.to(tl.int32) + 1)    

    for x in range(ix_start, ix_end): 
        dx = x * voxel_size + vol_min_x - mx 
        # 预计算 x 轴相关的中间变量
        xc0 = dx * c0; xc1 = dx * c1; xc2 = dx * c2 
        
        for y in range(iy_start, iy_end):
            dy = y * voxel_size + vol_min_y - my
            # 预计算 y 轴相关的中间变量
            yc3 = dy * c3; yc4 = dy * c4; yc5 = dy * c5
                
            for z in range(iz_start, iz_end):
                dz = z * voxel_size + vol_min_z - mz
                mahal = (dx * (xc0 + dy * c3 + dz * c6) +
                         dy * (dx * c1 + yc4 + dz * c7) +
                         dz * (dx * c2 + dy * c5 + dz * c8))                
                
                
                density = opac * tl.exp(-0.5 * mahal)
                # todo 计算网格索引
                offset = x * (dim_y * dim_z) + y * dim_z + z
                
                # todo 1. 累加密度
                tl.atomic_add(grid_density_ptr + offset, density)
                
                # todo 2. 累加特征
                for f in range(n_dims):
                    feat_val = tl.load(features_ptr + idx * n_dims + f)
                    tl.atomic_add(grid_feats_ptr + offset * n_dims + f, density * feat_val)

@triton.jit
def _splat_bwd_kernel_opt(
    grad_features_ptr, grad_opacities_ptr,  grad_means_ptr, grad_inv_covs_ptr,      # 输出：高斯特征、透明度、位置、协方差逆梯度
    grid_density_ptr, grid_feats_ptr,              # 输入：体素网格密度、体素网格特征
    grad_grid_density_ptr, grad_grid_feats_ptr,    # 输入： 损失对体素网格透明度、特征的梯度
    means_ptr, inv_covs_ptr, opacities_ptr, radii_ptr, features_ptr,  # 输入：高斯点原始位置、协方差逆、透明度、半径、特征
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size,
    N_gaussians,n_dims, 
    dim_x, dim_y, dim_z,
    eps: tl.constexpr = 1e-6
):
    idx = tl.program_id(0)
    if idx >= N_gaussians: return    
    
    # 均值x,y,z
    mx = tl.load(means_ptr + idx * 3 + 0)  
    my = tl.load(means_ptr + idx * 3 + 1)  
    mz = tl.load(means_ptr + idx * 3 + 2)

    # 协方差逆
    c_ptr = inv_covs_ptr + idx * 9
    c0, c1, c2 = tl.load(c_ptr + 0), tl.load(c_ptr + 1), tl.load(c_ptr + 2)
    c3, c4, c5 = tl.load(c_ptr + 3), tl.load(c_ptr + 4), tl.load(c_ptr + 5)
    c6, c7, c8 = tl.load(c_ptr + 6), tl.load(c_ptr + 7), tl.load(c_ptr + 8)
    
    # 透明度
    opac = tl.load(opacities_ptr + idx)
    
    # 半径
    # 加载三个轴向的独立半径
    rx = tl.load(radii_ptr + idx * 3 + 0)
    ry = tl.load(radii_ptr + idx * 3 + 1)
    rz = tl.load(radii_ptr + idx * 3 + 2)    

    ix_s = (mx - rx - vol_min_x) / voxel_size
    ix_e = (mx + rx - vol_min_x) / voxel_size
    ix_start = tl.maximum(0, ix_s.to(tl.int32))
    ix_end   = tl.minimum(dim_x, ix_e.to(tl.int32) + 1)

    iy_s = (my - ry - vol_min_y) / voxel_size
    iy_e = (my + ry - vol_min_y) / voxel_size
    iy_start = tl.maximum(0, iy_s.to(tl.int32))
    iy_end   = tl.minimum(dim_y, iy_e.to(tl.int32) + 1)

    iz_s = (mz - rz - vol_min_z) / voxel_size
    iz_e = (mz + rz - vol_min_z) / voxel_size
    iz_start = tl.maximum(0, iz_s.to(tl.int32))
    iz_end   = tl.minimum(dim_z, iz_e.to(tl.int32) + 1)    

    # 透明度梯度累加
    grad_opac_acc = 0.0
    
    # 均值梯度累加
    grad_mx_acc = 0.0
    grad_my_acc = 0.0
    grad_mz_acc = 0.0

    # 协方差逆梯度累加 (3x3 矩阵)
    gic0, gic1, gic2 = 0.0, 0.0, 0.0
    gic3, gic4, gic5 = 0.0, 0.0, 0.0
    gic6, gic7, gic8 = 0.0, 0.0, 0.0
    
    for x in range(ix_start, ix_end): 
        dx = x * voxel_size + vol_min_x - mx 
        # 预计算 x 轴相关的中间变量
        xc0 = dx * c0; xc1 = dx * c1; xc2 = dx * c2 
        
        for y in range(iy_start, iy_end):
            dy = y * voxel_size + vol_min_y - my
            # 预计算 y 轴相关的中间变量
            yc3 = dy * c3; yc4 = dy * c4; yc5 = dy * c5
                
            for z in range(iz_start, iz_end):
                dz = z * voxel_size + vol_min_z - mz
                mahal = (dx * (xc0 + dy * c3 + dz * c6) +
                         dy * (dx * c1 + yc4 + dz * c7) +
                         dz * (dx * c2 + dy * c5 + dz * c8))                
                
                exp_term = tl.exp(-0.5 * mahal)
                density = opac * exp_term
                
                # 计算网格索引
                offset = x * (dim_y * dim_z) + y * dim_z + z
                
                # 加载该位置的网格密度
                grid_density_val = tl.load(grid_density_ptr + offset)   
                grid_density_val = tl.maximum(grid_density_val, eps)
                
                g_density_from_loss = tl.load(grad_grid_density_ptr + offset)
                
                # 计算当前体素对均值梯度的中间变量
                # 1. 空间导数向量: Σ^-1 * (x_j - μ_i)
                dir_x = dx * c0 + dy * c1 + dz * c2
                dir_y = dx * c3 + dy * c4 + dz * c5
                dir_z = dx * c6 + dy * c7 + dz * c8
                
                v_combined_scalar = g_density_from_loss * opac
                
                # 密度梯度 路径 (1): 处理 grad_grid_density 的直接贡献
                grad_opac_acc += g_density_from_loss * exp_term
                            
                # 特征维度循环
                for f in range(n_dims): # todo n_dims: 特征维度
                    g_grid = tl.load(grad_grid_feats_ptr + offset * n_dims + f)   # 外界传回的梯度
                    F_grid = tl.load(grid_feats_ptr + offset * n_dims + f) # 此时网格位置的平均特征
                    f_i = tl.load(features_ptr + idx * n_dims + f) # 当前高斯点的特征
                    
                    # todo 1. 高斯特征梯度(不受分母 clamp 导数影响)
                    grad_gauss_feat = g_grid * density / grid_density_val
                    tl.atomic_add(grad_features_ptr + idx * n_dims + f, grad_gauss_feat)                    
                    
                    # 分子贡献项 (始终存在)
                    feat_grad_scalar = g_grid * (f_i / grid_density_val)
                    # 分母贡献项 (仅当 density 未被 clamp 时存在)
                    if grid_density_val > eps:
                        feat_grad_scalar -= g_grid * (F_grid / grid_density_val)
                    
                    # todo 2. 高斯透明度梯度    
                    # 路径 (2): 处理 grad_grid_feats 的商法则贡献                
                    grad_opac_acc += feat_grad_scalar * exp_term
                    
                    # todo 3. 均值(位置)梯度
                    v_combined_scalar += feat_grad_scalar * opac
                        
                grad_mx_acc += v_combined_scalar * exp_term * dir_x
                grad_my_acc += v_combined_scalar * exp_term * dir_y
                grad_mz_acc += v_combined_scalar * exp_term * dir_z
                
                S = (v_combined_scalar * exp_term ) * (-0.5)
                gic0 += S * dx * dx
                gic1 += S * dx * dy
                gic2 += S * dx * dz
                gic3 += S * dy * dx
                gic4 += S * dy * dy
                gic5 += S * dy * dz
                gic6 += S * dz * dx
                gic7 += S * dz * dy
                gic8 += S * dz * dz                
                
    tl.atomic_add(grad_opacities_ptr + idx, grad_opac_acc)
    tl.atomic_add(grad_means_ptr + idx * 3 + 0, grad_mx_acc)
    tl.atomic_add(grad_means_ptr + idx * 3 + 1, grad_my_acc)
    tl.atomic_add(grad_means_ptr + idx * 3 + 2, grad_mz_acc)

    out_gic_ptr = grad_inv_covs_ptr + idx * 9
    tl.atomic_add(out_gic_ptr + 0, gic0); tl.atomic_add(out_gic_ptr + 1, gic1); tl.atomic_add(out_gic_ptr + 2, gic2)
    tl.atomic_add(out_gic_ptr + 3, gic3); tl.atomic_add(out_gic_ptr + 4, gic4); tl.atomic_add(out_gic_ptr + 5, gic5)
    tl.atomic_add(out_gic_ptr + 6, gic6); tl.atomic_add(out_gic_ptr + 7, gic7); tl.atomic_add(out_gic_ptr + 8, gic8) 
      
class GaussSplatting3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3d, covs, opacities, features, vol_range, voxel_size, grid_shape):
        device = means3d.device
        N = means3d.shape[0]
        n_dims = features.shape[1]
        inv_covs = torch.inverse(covs)        
        variances = torch.diagonal(covs, dim1=-2, dim2=-1)
        radii = 3.0 * torch.sqrt(variances)
        
        
        grid_density = torch.zeros(grid_shape, device=device)
        grid_feats = torch.zeros((*grid_shape, n_dims), device=device)     
        grid_feats[..., -1] = 1e-5  # todo 初始化最后一维为极小数，确保默认预测为背景，避免预测第0类       
        _splat_fwd_kernel_opt[(N,)](
            means3d, inv_covs.reshape(N, 9), opacities, radii, features,
            grid_density, grid_feats,
            float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
            float(voxel_size), 
            N, n_dims,
            int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        )
        grid_feats = grid_feats / grid_density.unsqueeze(-1).clamp(1e-6)  
        
        # 存储必要的张量
        ctx.save_for_backward(means3d, opacities, features, inv_covs, radii, grid_density, grid_feats)
        # 存储普通数值
        ctx.vol_range = vol_range
        ctx.voxel_size = voxel_size
        ctx.grid_shape = grid_shape
               
        return grid_density, grid_feats

    @staticmethod
    def backward(ctx, grad_grid_density, grad_grid_feats): # 损失对前向输出的梯度
        # 恢复前向向量
        means3d, opacities, features, inv_covs, radii, grid_density, grid_feats = ctx.saved_tensors
        N = means3d.shape[0]
        n_dims = features.shape[1]        
        
        # 初始化梯度
        grad_features = torch.zeros_like(features)
        grad_opacities = torch.zeros_like(opacities)
        grad_means = torch.zeros_like(means3d)
        grad_inv_covs = torch.zeros((N, 9), device=means3d.device) 
              
        _splat_bwd_kernel_opt[(N,)](
            grad_features, grad_opacities, grad_means, grad_inv_covs,  # 输出   
            grid_density, grid_feats,      # 输入:  体素网格状态
            grad_grid_density.contiguous(), grad_grid_feats.contiguous(),  # 输入： 损失对体素网格透明度、特征的梯度
            means3d, inv_covs.reshape(N, 9), opacities, radii, features,  # 输入：高斯点原始位置、协方差逆、透明度、半径、特征
            float(ctx.vol_range[0]), float(ctx.vol_range[1]), float(ctx.vol_range[2]),
            float(ctx.voxel_size),
            N, n_dims,
            int(ctx.grid_shape[0]), int(ctx.grid_shape[1]), int(ctx.grid_shape[2])
        )       
        # 将逆协方差梯度转换为原始协方差梯度
        grad_inv_covs = grad_inv_covs.view(N, 3, 3)
        grad_covs = -inv_covs @ grad_inv_covs @ inv_covs
        
        return grad_means, grad_covs, grad_opacities, grad_features, None, None, None


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
        grid_feats[..., -1] = 1e-5 
        
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

