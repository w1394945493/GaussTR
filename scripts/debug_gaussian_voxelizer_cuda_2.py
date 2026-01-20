import time
import torch
import gauss_splatting_cuda
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw) # todo 将四元数转换为(3 3)的旋转矩阵
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i") # 转置
        @ rearrange(rotation, "... i j -> ... j i")
    )


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
        grid_feats_norm = grid_feats / grid_density.unsqueeze(-1).clamp(min=eps) # todo 这里还有一步归一化

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



def split_global_empty_gaussian(empty_args, grid_shape, voxel_size, vol_min, device):
    """
    将一个巨型背景高斯分裂为多个覆盖局部的中型高斯
    """
    dim_x, dim_y, dim_z = grid_shape
    
    # 设定分裂密度：例如每 20 个体素放一个子高斯
    # 这个值决定了并行度。200/20 = 10, 总共 10x10x1 = 100个子点
    step = 20 
    
    # 生成局部中心点坐标
    x_range = torch.arange(step//2, dim_x, step, device=device) * voxel_size + vol_min[0]
    y_range = torch.arange(step//2, dim_y, step, device=device) * voxel_size + vol_min[1]
    z_range = torch.tensor([empty_args['mean'][2]], device=device) # Z轴通常薄，可以不分
    
    grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
    new_means = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    num_subs = new_means.shape[0]
    
    # 关键：计算子高斯的 Scale 
    # 为了保证平滑覆盖，子高斯的 Scale 应该是 step * voxel_size 的 1.5 倍左右（保证重叠）
    sub_scale_val = step * voxel_size * 1.5
    new_scales = torch.full((num_subs, 3), sub_scale_val, device=device)
    new_scales[:, 2] = empty_args['scale'][2] # Z 轴保持原样
    
    # 构造协方差
    new_covs = torch.diag_embed(new_scales**2)
    
    # 保持属性
    new_opacities = torch.ones((num_subs,), device=device)
    
    return new_means, new_covs, new_opacities

def test_speed(with_empty=False, num_iters=10):
    device = torch.device("cuda")
    N = 18000
    voxel_size = 0.5
    n_class = 18
    vol_min = torch.tensor([-50.0, -50.0, -5.0], device=device)
    vol_max = torch.tensor([50.0, 50.0, 3.0], device=device)
    vol_range = torch.cat([vol_min, vol_max])
    
    dim_x, dim_y, dim_z = 200, 200, 16 # 固定形状方便对比
    grid_shape = (dim_x, dim_y, dim_z)
    
    # 初始化基础数据
    means3d = (torch.rand((N, 3), device=device) * (vol_max - vol_min) + vol_min)
    L = torch.randn((N, 3, 3), device=device) * 0.1
    covs = (torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 0.1)
    opacities = torch.rand((N,), device=device)
    features = torch.rand((N, n_class), device=device)

    # 在你的脚本中使用：
    if with_empty:
        # 1. 为了铺满背景，步长也要相应减小
        # 如果每个点覆盖 10 个网格(5.0 units)，步长设为 5.0 可以实现完美衔接
        step_units = 5.0 
        
        # 2. 生成中心点网格
        x_coords = torch.arange(vol_min[0] + step_units/2, vol_max[0], step_units, device=device)
        y_coords = torch.arange(vol_min[1] + step_units/2, vol_max[1], step_units, device=device)
        # Z 轴我们铺设两层，确保纵深也有覆盖
        z_coords = torch.linspace(vol_min[2] + 1.0, vol_max[2] - 1.0, 2, device=device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        e_means = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        num_e = e_means.shape[0]
        
        # 3. 设置精确的 Scale
        # XY 覆盖 10 个网格 -> scale = 10 / 12 = 0.833
        # Z  覆盖 2 个网格 -> scale = 2 / 12 = 0.166
        e_scale_xy = 0.833
        e_scale_z = 0.166
        
        e_scales = torch.tensor([e_scale_xy, e_scale_xy, e_scale_z], device=device).repeat(num_e, 1)
        
        # 4. 构造属性
        e_covs = torch.diag_embed(e_scales**2)
        e_opas = torch.ones((num_e,), device=device) * 0.3 # 背景淡淡的就好
        e_feats = torch.zeros((num_e, n_class), device=device)
        
        
        
        # 合并
        means3d = torch.cat([means3d, e_means])
        features = torch.cat([features, e_feats])
        covs = torch.cat([covs, e_covs])
        opacities = torch.cat([opacities, e_opas])


    # 预热 (Warm up)
    for _ in range(3):
        _ = GaussSplatting3DCuda.apply(means3d, covs, opacities, features, vol_range, voxel_size, grid_shape)
    
    torch.cuda.synchronize()
    
    
    
    t_start = time.time()
    
    # 正式测试
    for _ in range(num_iters):
        grid_density, grid_feats = GaussSplatting3DCuda.apply(
            means3d, covs, opacities, features, vol_range, voxel_size, grid_shape
        )
        
    torch.cuda.synchronize()
    t_end = time.time()
    
    avg_time = (t_end - t_start) / num_iters
    return avg_time

if __name__ == '__main__':
    print("开始性能测试...")
    
    time_normal = test_speed(with_empty=False)
    print(f"👉 [with_empty=False] 平均耗时: {time_normal*1000:.4f} ms")
    
    
    
    time_empty = test_speed(with_empty=True)
    print(f"👉 [with_empty=True ] 平均耗时: {time_empty*1000:.4f} ms")
    
    diff = time_empty / time_normal
    print(f"\n性能差异: 开启 empty 后变慢了 {diff:.2f} 倍")



