import time

import torch
import triton
import triton.language as tl


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
                
                # todo 
                for f in range(n_dims):
                    feat_val = tl.load(features_ptr + idx * n_dims + f)
                    tl.atomic_add(grid_feats_ptr + offset * n_dims + f, density * feat_val)

@triton.jit
def _splat_bwd_kernel_opt(
    grad_features_ptr, grad_opacities_ptr,         # 输出：高斯特征、透明度梯度
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

    grad_opac_acc = 0.0
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
                
                # 路径 (1): 处理 grad_grid_density 的直接贡献
                g_density_from_loss = tl.load(grad_grid_density_ptr + offset)
                grad_opac_acc += g_density_from_loss * exp_term
                            
                for f in range(n_dims): # todo n_dims: 特征维度
                    g_grid = tl.load(grad_grid_feats_ptr + offset * n_dims + f)   # 外界传回的梯度
                    F_grid = tl.load(grid_feats_ptr + offset * n_dims + f) # 此时网格位置的平均特征
                    f_i = tl.load(features_ptr + idx * n_dims + f) # 当前高斯点的特征
                    
                    # todo 1. 高斯特征梯度(不受分母 clamp 导数影响)
                    grad_gauss_feat = g_grid * density / grid_density_val
                    tl.atomic_add(grad_features_ptr + idx * n_dims + f, grad_gauss_feat)                    
                    # todo 2. 高斯透明度梯度    
                    # 路径 (2): 处理 grad_grid_feats 的商法则贡献                
                    # grad_opac_acc += g_grid * (exp_term * (f_i - F_grid) / grid_density_val)
                    # 分子贡献项 (始终存在)
                    grad_opac_acc += g_grid * (exp_term * f_i / grid_density_val)
                    # 分母贡献项 (仅当 density 未被 clamp 时存在)
                    if grid_density_val > eps:
                        grad_opac_acc -= g_grid * (exp_term * F_grid / grid_density_val)
                        

    tl.atomic_add(grad_opacities_ptr + idx, grad_opac_acc)

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
              
        _splat_bwd_kernel_opt[(N,)](
            grad_features, grad_opacities, # 输出   
            grid_density, grid_feats,      # 输入:  体素网格状态
            grad_grid_density.contiguous(), grad_grid_feats.contiguous(),  # 输入： 损失对体素网格透明度、特征的梯度
            means3d, inv_covs.reshape(N, 9), opacities, radii, features,  # 输入：高斯点原始位置、协方差逆、透明度、半径、特征
            float(ctx.vol_range[0]), float(ctx.vol_range[1]), float(ctx.vol_range[2]),
            float(ctx.voxel_size),
            N, n_dims,
            int(ctx.grid_shape[0]), int(ctx.grid_shape[1]), int(ctx.grid_shape[2])
        )        
        
        return None, None, grad_opacities, grad_features, None, None, None


if __name__=='__main__':
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    N = 1800
    voxel_size = 0.4   # 网格尺寸
    n_class = 18

    vol_min = torch.tensor([-40.0, -40.0, -1.0], device=device)
    vol_max = torch.tensor([40.0, 40.0, 5.4], device=device)
    vol_range = torch.cat([vol_min, vol_max]) 

    # 网格形状 (200, 200, 16)
    dim_x = int((vol_max[0] - vol_min[0]) / voxel_size)
    dim_y = int((vol_max[1] - vol_min[1]) / voxel_size)
    dim_z = int((vol_max[2] - vol_min[2]) / voxel_size)
    grid_shape = (dim_x, dim_y, dim_z) 
    
    means3d = (torch.rand((N, 3), device=device) * (vol_max - vol_min) + vol_min).requires_grad_(True)
    L = torch.randn((N, 3, 3), device=device) * 0.1
    covs = (torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 0.1).requires_grad_(True)
    opacities = torch.rand((N,), device=device).requires_grad_(True)
    
    features = torch.rand((N, n_class), device=device).requires_grad_(True)
    
    # 1. 生成真值标签：形状为 (200, 200, 16)，值在 [0, 17]
    target_labels = torch.randint(0, n_class, grid_shape, device=device)
    print('target_labels.shape:',target_labels.shape)

    # 损失
    criterion = torch.nn.CrossEntropyLoss()
    
    device = means3d.device
    N = means3d.shape[0]
    inv_covs = torch.inverse(covs)        
    variances = torch.diagonal(covs, dim1=-2, dim2=-1)      
    radii = 3.0 * torch.sqrt(variances)
    
    
    # todo -------------------------------------------------#
    # todo triton并行处理
    
    # todo 第一次运行时，包含编译用时，耗时会较长
    torch.cuda.synchronize()
    t0 = time.time()

    _ = GaussSplatting3D.apply(means3d, covs, opacities, features, vol_range, voxel_size, grid_shape)

    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"triton编译用时: {t1-t0}") 


    features_triton = features.clone().detach().requires_grad_(True)
    opacities_triton = opacities.clone().detach().requires_grad_(True)
    
    torch.cuda.synchronize()
    t0 = time.time()
    grid_density_triton, grid_feats_triton = GaussSplatting3D.apply(means3d, covs, opacities_triton, features_triton, vol_range, voxel_size, grid_shape)
    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"triton第二次调用用时: {t1-t0}") 

    
    
    # todo---------------------------------------#
    # todo 原方法
    features_ori = features.clone().detach().requires_grad_(True)
    opacities_ori = opacities.clone().detach().requires_grad_(True)
    
    
    lin_x = torch.linspace(vol_min[0], vol_max[0] - voxel_size, dim_x, device=device)
    lin_y = torch.linspace(vol_min[1], vol_max[1] - voxel_size, dim_y, device=device)
    lin_z = torch.linspace(vol_min[2], vol_max[2] - voxel_size, dim_z, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    torch.cuda.synchronize()
    t0 = time.time()
    grid_density_ori, grid_feats_ori = splat_into_3d(
            grid_coords=grid_coords,
            means3d=means3d,
            opacities=opacities_ori,
            covariances=covs,
            vol_range=vol_range,
            voxel_size=voxel_size,
            features=features_ori
        )
    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"splat_into_3d()用时: {t1-t0}") 

    grid_density_ori = grid_density_ori.squeeze(-1)
    
    torch.testing.assert_close(
        grid_density_triton,
        grid_density_ori,
        rtol=1e-4,
        atol=1e-6
    )
    print("forward denisty -> 对齐")

    torch.testing.assert_close(
        grid_feats_triton,
        grid_feats_ori,
        rtol=1e-4,
        atol=1e-6
    )
    print("forward feats -> 对齐")
    
    input_triton = grid_feats_triton.permute(3, 0, 1, 2).unsqueeze(0)  # (200, 200, 16, 18) -> 转换为 (1, 18, 200, 200, 16)
    target_triton = target_labels.unsqueeze(0) # 变为 (1, 200, 200, 16)
    loss_triton = criterion(input_triton, target_triton)
    print(f'loss triton: {loss_triton.item()}')
    loss_triton.backward()

    input_ori = grid_feats_ori.permute(3, 0, 1, 2).unsqueeze(0)
    target_ori = target_labels.unsqueeze(0)
    loss_ori = criterion(input_ori, target_ori)
    print(f'loss ori: {loss_ori.item()}')
    loss_ori.backward()
    
    # 3. 对比 特征 的梯度
    try:
        torch.testing.assert_close(features_triton.grad, features_ori.grad, rtol=1e-4, atol=1e-5)
        print("Backward 特征梯度验证 -> [通过]")
        
        print(f"features_triton的梯度是否有 NaN: {torch.isnan(features_triton.grad).any()}")
        print(f"features_triton的梯度最大值: {features_triton.grad.max().item()}")

        print(f"features_ori的梯度是否有 NaN: {torch.isnan(features_ori.grad).any()}")
        print(f"features_ori的梯度最大值: {features_ori.grad.max().item()}")        
        # ----------------------------------------------------------------------#        
    except AssertionError as e:
        print("Backward 特征梯度验证 -> [失败]")
        print(e)
        
    try:
        torch.testing.assert_close(opacities_triton.grad, opacities_ori.grad, rtol=1e-4, atol=1e-5)
        print(f"opacities_triton的梯度是否有 NaN: {torch.isnan(opacities_triton.grad).any()}")
        print(f"opacities_triton的梯度最大值: {opacities_triton.grad.max().item()}")
        
        print(f"opacities_ori的梯度是否有 NaN: {torch.isnan(opacities_ori.grad).any()}")
        print(f"opacities_ori的梯度最大值: {opacities_ori.grad.max().item()}") 

    except AssertionError as e:
        print("Backward 透明度梯度验证 -> [失败]")
        print(e)