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
    
    # 特征
    v_size = voxel_size    
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
        dx = x * v_size + vol_min_x - mx 
        # 预计算 x 轴相关的中间变量
        xc0 = dx * c0; xc1 = dx * c1; xc2 = dx * c2 
        
        for y in range(iy_start, iy_end):
            dy = y * v_size + vol_min_y - my
            # 预计算 y 轴相关的中间变量
            yc3 = dy * c3; yc4 = dy * c4; yc5 = dy * c5
                
            for z in range(iz_start, iz_end):
                dz = z * v_size + vol_min_z - mz
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
                    tl.atomic_add(grid_feats_ptr + offset * n_class + f, density * feat_val)
                
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

    device = means3d.device
    N = means3d.shape[0]
    inv_covs = torch.inverse(covs)        
    variances = torch.diagonal(covs, dim1=-2, dim2=-1)
    # radii = 3.0 * torch.sqrt(variances.max(dim=-1)[0])        
    radii = 3.0 * torch.sqrt(variances)
    
    
    # todo -------------------------------------------------#
    # todo 
    grid_density_triton = torch.zeros(grid_shape, device=device)
    grid_feats_triton = torch.zeros((*grid_shape, n_class), device=device)
    # todo 第一次编译耗时较长
    torch.cuda.synchronize()
    t0 = time.time()
    _splat_fwd_kernel_opt[(N,)](
        means3d, inv_covs.reshape(N, 9), opacities, radii,features,
        grid_density_triton, grid_feats_triton,
        float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
        float(voxel_size), 
        N, n_class,
        int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    )  
    eps = 1e-6
    grid_feats_triton /= grid_density_triton.unsqueeze(-1).clamp(eps)
    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"triton编译用时: {t1-t0}") 

    grid_density_triton = torch.zeros(grid_shape, device=device)
    grid_feats_triton = torch.zeros((*grid_shape, n_class), device=device)
    
    torch.cuda.synchronize()
    t0 = time.time()
    _splat_fwd_kernel_opt[(N,)](
        means3d, inv_covs.reshape(N, 9), opacities, radii, features,
        grid_density_triton, grid_feats_triton,
        float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
        float(voxel_size), 
        N, n_class, 
        int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    )
    eps = 1e-6
    grid_feats_triton /= grid_density_triton.unsqueeze(-1).clamp(eps)   
    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"triton第二次调用用时: {t1-t0}") 

    # todo---------------------------------------#
    # todo 调用原方法
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
            opacities=opacities,
            covariances=covs,
            vol_range=vol_range,
            voxel_size=voxel_size,
            features=features
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
    print("denisty 对齐")

    torch.testing.assert_close(
        grid_feats_triton,
        grid_feats_ori,
        rtol=1e-4,
        atol=1e-6
    )
    print("feats 对齐")