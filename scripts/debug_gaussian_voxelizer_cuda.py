import time
import torch
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
        grid_feats[..., -1] = 1e-5
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
            means3d.contiguous(), # todo 必须要保证 存储连续 .contiguous() 做了slice等操作，就会导致不连续，修改操作(in-place modify)不会破坏连续性
            inv_covs.view(N, 9).contiguous(),
            opacities.contiguous(),
            radii.contiguous(),
            features.contiguous(), # todo means、inv_covs等都是输入参数，前向传播过程固定不变，对应const float*
            grid_density, # todo 输出参数：对应 float*, 不应当有const
            grid_feats,   # todo 原厂保证：使用torch.zeros等新创建的张量，Pytorch默认在显存中新开辟一块完全连续的空间
            float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
            float(voxel_size)
        )

        # 4. 归一化特征
        eps = 1e-6
        grid_feats_norm = grid_feats / grid_density.unsqueeze(-1).clamp(min=eps)

        # 5. 保存给反向传播用的变量
        ctx.save_for_backward(means3d, inv_covs, opacities, radii, features, grid_density, grid_feats_norm) # todo 在backward函数里一定注意不要修改save_tensors这些变量
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

        # 返回的梯度顺序必须和 forward 的参数顺序一一对应
        # means3d, covs, opacities, features, vol_range, voxel_size, grid_shape
        # 不需要梯度的参数返回 None
        return grad_means, grad_covs, grad_opacities, grad_features, None, None, None



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
    
    means3d = (torch.rand((N, 3), device=device) * (vol_max - vol_min) + vol_min)
    L = torch.randn((N, 3, 3), device=device) * 0.1
    covs = (torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 0.1)
    opacities = torch.rand((N,), device=device)
    
    features = torch.rand((N, n_class), device=device)
    
    # 1. 生成真值标签：形状为 (200, 200, 16)，值在 [0, 17]
    target_labels = torch.randint(0, n_class, grid_shape, device=device)
    print('target_labels.shape:',target_labels.shape)

    # 损失
    criterion = torch.nn.CrossEntropyLoss()
    
    device = means3d.device
    N = means3d.shape[0]

    features_cuda = features.clone().detach().requires_grad_(True)
    opacities_cuda = opacities.clone().detach().requires_grad_(True)
    means3d_cuda = means3d.clone().detach().requires_grad_(True)
    covs_cuda = covs.clone().detach().requires_grad_(True)
    
    torch.cuda.synchronize()
    t0 = time.time()
    grid_density_cuda, grid_feats_cuda = GaussSplatting3DCuda.apply(means3d_cuda, covs_cuda, opacities_cuda, features_cuda, vol_range, voxel_size, grid_shape)
    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"cuda调用用时: {t1-t0}") 

    # todo---------------------------------------#
    # todo 原方法
    features_ori = features.clone().detach().requires_grad_(True)
    opacities_ori = opacities.clone().detach().requires_grad_(True)
    means3d_ori = means3d.clone().detach().requires_grad_(True)
    covs_ori = covs.clone().detach().requires_grad_(True)
    
    lin_x = torch.linspace(vol_min[0], vol_max[0] - voxel_size, dim_x, device=device)
    lin_y = torch.linspace(vol_min[1], vol_max[1] - voxel_size, dim_y, device=device)
    lin_z = torch.linspace(vol_min[2], vol_max[2] - voxel_size, dim_z, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    torch.cuda.synchronize()
    t0 = time.time()
    grid_density_ori, grid_feats_ori = splat_into_3d(
            grid_coords=grid_coords,
            means3d=means3d_ori,
            opacities=opacities_ori,
            covariances=covs_ori,
            vol_range=vol_range,
            voxel_size=voxel_size,
            features=features_ori
        )
    t1 = time.time()   
    torch.cuda.synchronize()
    print(f"splat_into_3d()用时: {t1-t0}") 

    grid_density_ori = grid_density_ori.squeeze(-1)
    
    torch.testing.assert_close(
        grid_density_cuda,
        grid_density_ori,
        rtol=1e-4,
        atol=1e-6
    )
    print("forward denisty -> 对齐")

    torch.testing.assert_close(
        grid_feats_cuda,
        grid_feats_ori,
        rtol=1e-4,
        atol=1e-6
    )
    print("forward feats -> 对齐")
    
    input_cuda = grid_feats_cuda.permute(3, 0, 1, 2).unsqueeze(0)  # (200, 200, 16, 18) -> 转换为 (1, 18, 200, 200, 16)
    target_cuda = target_labels.unsqueeze(0) # 变为 (1, 200, 200, 16)
    loss_cuda = criterion(input_cuda, target_cuda)
    loss_cuda = loss_cuda + grid_density_cuda.mean()
    print(f'loss triton: {loss_cuda.item()}')
    loss_cuda.backward()

    input_ori = grid_feats_ori.permute(3, 0, 1, 2).unsqueeze(0)
    target_ori = target_labels.unsqueeze(0)
    loss_ori = criterion(input_ori, target_ori)
    loss_ori = loss_ori + grid_density_ori.mean()
    print(f'loss ori: {loss_ori.item()}')
    loss_ori.backward()
    
    
    
    # 3. 对比 特征 的梯度
    try:
        torch.testing.assert_close(features_cuda.grad, features_ori.grad, rtol=1e-4, atol=1e-5)
        print("Backward 特征梯度验证 -> [通过]")
        
        print(f"features_cuda的梯度是否有 NaN: {torch.isnan(features_cuda.grad).any()}")
        print(f"features_cuda的梯度最大值: {features_cuda.grad.max().item()}")

        print(f"features_ori的梯度是否有 NaN: {torch.isnan(features_ori.grad).any()}")
        print(f"features_ori的梯度最大值: {features_ori.grad.max().item()}")        
        # ----------------------------------------------------------------------#        
    except AssertionError as e:
        print("Backward 特征梯度验证 -> [失败]")
        print(e)
        
    try:
        torch.testing.assert_close(opacities_cuda.grad, opacities_ori.grad, rtol=1e-4, atol=1e-5)
        
        print("Backward 透明度梯度验证 -> [通过]")
        
        print(f"opacities_cuda的梯度是否有 NaN: {torch.isnan(opacities_cuda.grad).any()}")
        print(f"opacities_cuda的梯度最大值: {opacities_cuda.grad.max().item()}")
        
        print(f"opacities_ori的梯度是否有 NaN: {torch.isnan(opacities_ori.grad).any()}")
        print(f"opacities_ori的梯度最大值: {opacities_ori.grad.max().item()}") 

    except AssertionError as e:
        print("Backward 透明度梯度验证 -> [失败]")
        print(e)

    try:
        torch.testing.assert_close(means3d_cuda.grad, means3d_ori.grad, rtol=1e-4, atol=1e-5)
        
        print("Backward 均值梯度验证 -> [通过]")
        
        print(f"means3d_cuda的梯度是否有 NaN: {torch.isnan(means3d_cuda.grad).any()}")
        print(f"means3d_cuda的梯度最大值: {means3d_cuda.grad.max().item()}")
        
        print(f"means3d_ori的梯度是否有 NaN: {torch.isnan(means3d_ori.grad).any()}")
        print(f"means3d_ori的梯度最大值: {means3d_ori.grad.max().item()}") 

    except AssertionError as e:
        print("Backward 均值梯度验证 -> [失败]")
        print(e)
    
    try:
        torch.testing.assert_close(covs_cuda.grad, covs_ori.grad, rtol=1e-4, atol=1e-5)
        
        print("Backward 协方差梯度验证 -> [通过]")
        
        print(f"covs_cuda的梯度是否有 NaN: {torch.isnan(covs_cuda.grad).any()}")
        print(f"covs_cuda的梯度最大值: {covs_cuda.grad.max().item()}")
        
        print(f"covs_ori的梯度是否有 NaN: {torch.isnan(covs_ori.grad).any()}")
        print(f"covs_ori的梯度最大值: {covs_ori.grad.max().item()}") 

    except AssertionError as e:
        print("Backward 协方差梯度验证 -> [失败]")
        print(e)