import torch
import triton
import triton.language as tl
import time


# todo 原代码
def splat_into_3d(grid_coords, # todo occ占用感知空间 [-40, -40, -1, 40, 40, 5.4]
                  means3d,     # todo 高斯点位置 (N,3) N=6x300 6个相机，300个query高斯
                  opacities,   # todo 透明度
                  covariances, # todo 协方差
                  vol_range,   # todo 体素范围 
                  voxel_size): # todo 体素尺寸 0.4
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1), device=grid_coords.device)
    inv_covs = torch.inverse(covariances) # 原方法内部求逆
    
    for g in range(means3d.size(0)):
        sigma = torch.sqrt(torch.diag(covariances[g]))
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        
        # 边界检查
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
            
        bounds = bounds.clamp(vol_range[:3], vol_range[3:])
        # 注意：这里原代码转 int 再 tolist，可能会有微小舍入误差
        b_idx = ((bounds - vol_range[:3]) / voxel_size).int()
        slices = tuple([slice(b_idx[0, i], b_idx[1, i] + 1) for i in range(3)])

        diff = grid_coords[slices] - means3d[g]
        # 马氏距离计算
        maha_dist = (diff.unsqueeze(-2) @ inv_covs[g] @ diff.unsqueeze(-1)).squeeze(-1)
        density = opacities[g] * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density
        
    return grid_density.squeeze(-1)

# todo 装饰器，分别负责性能优化和代码编译
# todo 当给代码加上@triton.jit，Triton编译器会接管该函数，不再是普通的python代码，而是一段硬件指令模板
# todo 原理：把python语法翻译成PTX(Nvidia的汇编语言)。只有通过jit编译的代码，才能直接在GPU的流式多处理器(SM)上跑起来

# todo @triton.autotune: 自动调优 
# todo 这里提供了三组方案：A方案 BLOCK_SIZE=128, 4个线程束；B方案：BLOCK_SIZE=256, 4个线程束；C方案：BLOCK_SIZE=512, 8个线程束
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
#     ],
#     key=['N_gaussians'],
# )
@triton.jit
def _splat_fwd_kernel_opt(
    means_ptr, inv_covs_ptr, opacities_ptr,
    radii_ptr,  # 新增：每个点的动态半径
    grid_ptr,
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size,
    N_gaussians, 
    dim_x, dim_y, dim_z,
    # BLOCK_SIZE: tl.constexpr  # todo 告诉编译器，这个值在编译时就定死了（比如 128），不是运行中会变的变量。
):
    idx = tl.program_id(0) # todo 
    if idx >= N_gaussians: return

    # todo 在@triton.jit装饰的函数里，只能使用tl.xxx(Triton语言)，而不能使用普通的torch或numpy
    # todo 从显存搬用数据到寄存器：tl.load: 把数据搬用到线程的寄存器上机械能计算
    # 1. 显式加载 3D 均值（避免解包错误）
    mx = tl.load(means_ptr + idx * 3 + 0)  # todo 定义高斯中心 idx*3: 每个点有(x,y,z)三个坐标，第idx的数据从idx*3的位置开始
    my = tl.load(means_ptr + idx * 3 + 1)  # todo +0/1/2: 分别取出该点的x,y,z坐标
    mz = tl.load(means_ptr + idx * 3 + 2)


    # todo 不直接写 m=means[idx]: Triton(和CUDA)的底层访存逻辑：GPU并不理解"对象"或"切片"，它只理解地址 means_ptr + offset: 最直接，最快的访问方式
    # 2. 显式加载逆协方差矩阵 9 个分量
    c_base = idx * 9                            # todo idx*9: 高斯点的形状3x3，仓库里把这些矩阵摊平放成一排，9个数代表椭球在3D空间内
    c0 = tl.load(inv_covs_ptr + c_base + 0) 
    c1 = tl.load(inv_covs_ptr + c_base + 1)
    c2 = tl.load(inv_covs_ptr + c_base + 2)
    c3 = tl.load(inv_covs_ptr + c_base + 3)
    c4 = tl.load(inv_covs_ptr + c_base + 4)
    c5 = tl.load(inv_covs_ptr + c_base + 5)
    c6 = tl.load(inv_covs_ptr + c_base + 6)
    c7 = tl.load(inv_covs_ptr + c_base + 7)
    c8 = tl.load(inv_covs_ptr + c_base + 8)

    opac = tl.load(opacities_ptr + idx)
    if opac < 0.001: return 

    # 加载动态半径 (Python 端计算好的 3 * max_scale)
    radius = tl.load(radii_ptr + idx)
    v_size = voxel_size
    
    
    # 计算包围盒逻辑    # todo 确定当前这个高斯点影响3D网格中的哪些格子？
    # todo 从"物理世界"映射到"网格索引" 高斯点的坐标是物理长度，而网格存储需要的是整数索引
    ix_s = (mx - radius - vol_min_x) / v_size  # todo mx-radius：算出高斯点在物理空间中影响的最左侧边界；-vol_min_x: 物理空间坑你不是从0开始，减去偏移量
    ix_e = (mx + radius - vol_min_x) / v_size  
    ix_start = tl.maximum(0, ix_s.to(tl.int32)) # todo 确定整数范围(ix_start到ix_end) tl.maximum(0,...)防止高斯点的一部分飘到网格外面(索引不能为负)
    
    # ix_end   = tl.minimum(dim_x, tl.math.ceil(ix_e).to(tl.int32)) # todo 防止太大，超出网格的最大范围(防止内存越界导致程序崩溃)
    ix_end = tl.minimum(dim_x, ix_e.to(tl.int32) + 1)
    
    
    iy_s = (my - radius - vol_min_y) / v_size
    iy_e = (my + radius - vol_min_y) / v_size
    iy_start = tl.maximum(0, iy_s.to(tl.int32))
    # iy_end   = tl.minimum(dim_y, tl.math.ceil(iy_e).to(tl.int32))
    iy_end   = tl.minimum(dim_y, iy_e.to(tl.int32) + 1)

    iz_s = (mz - radius - vol_min_z) / v_size
    iz_e = (mz + radius - vol_min_z) / v_size
    iz_start = tl.maximum(0, iz_s.to(tl.int32))
    # iz_end   = tl.minimum(dim_z, tl.math.ceil(iz_e).to(tl.int32))
    iz_end   = tl.minimum(dim_z, iz_e.to(tl.int32)+1)

    # 3. 循环展开与部分预计算
    for x in range(ix_start, ix_end): # todo 三层局部嵌套循环：遍历3D局部网格，依次访问每一个可能被高斯点覆盖的小方格
        dx = x * v_size + vol_min_x - mx # todo 计算相对距离 d=(dx,dy,dz) 这里是计算当前格子中心到高斯点球心的物理距离
        # 预计算 x 轴相关的中间变量
        xc0 = dx * c0; xc1 = dx * c1; xc2 = dx * c2 # todo 预计算优化：为了省时，
        
        for y in range(iy_start, iy_end):
            dy = y * v_size + vol_min_y - my
            # 预计算 y 轴相关的中间变量
            yc3 = dy * c3; yc4 = dy * c4; yc5 = dy * c5
            
            # 这里的常数项计算可以减少内层循环压力
            for z in range(iz_start, iz_end):
                dz = z * v_size + vol_min_z - mz
                # todo ----------------------------------#
                # 完整的马氏距离计算
                mahal = (dx * (xc0 + dy * c3 + dz * c6) +
                         dy * (dx * c1 + yc4 + dz * c7) +
                         dz * (dx * c2 + dy * c5 + dz * c8)) # todo  d^T Σ^-1 d
                
                density = opac * tl.exp(-0.5 * mahal)
                
                offset = x * (dim_y * dim_z) + y * dim_z + z
                tl.atomic_add(grid_ptr + offset, density) # todo 可能有多个高斯点同时覆盖了同一格子，把贡献累加起来

# todo 反向传播 -> 溯源
# todo 前向传播：高斯点如何贡献到网格 -> 反向传播：如果希望网格里某个数值大一些，如何调整高斯点的位置，形状和透明度
@triton.jit
def _splat_bwd_kernel(
    grad_grid_ptr,
    means_ptr, inv_covs_ptr, opacities_ptr,
    radii_ptr,
    grad_means_ptr, grad_inv_covs_ptr, grad_opacities_ptr,
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size,
    N_gaussians,
    dim_x, dim_y, dim_z
):
    idx = tl.program_id(0)
    if idx >= N_gaussians: return

    # 加载均值
    mx = tl.load(means_ptr + idx * 3 + 0)
    my = tl.load(means_ptr + idx * 3 + 1)
    mz = tl.load(means_ptr + idx * 3 + 2)
    
    # 显式加载逆协方差分量
    c_base = idx * 9
    c0=tl.load(inv_covs_ptr+c_base+0); c1=tl.load(inv_covs_ptr+c_base+1); c2=tl.load(inv_covs_ptr+c_base+2)
    c3=tl.load(inv_covs_ptr+c_base+3); c4=tl.load(inv_covs_ptr+c_base+4); c5=tl.load(inv_covs_ptr+c_base+5)
    c6=tl.load(inv_covs_ptr+c_base+6); c7=tl.load(inv_covs_ptr+c_base+7); c8=tl.load(inv_covs_ptr+c_base+8)
    
    opac = tl.load(opacities_ptr + idx)

    dmx, dmy, dmz = 0.0, 0.0, 0.0
    dopac = 0.0
    dc0=0.0; dc1=0.0; dc2=0.0; dc3=0.0; dc4=0.0; dc5=0.0; dc6=0.0; dc7=0.0; dc8=0.0

    radius = tl.load(radii_ptr + idx) # 使用前向相同的半径
    
    # 同样使用显式类型转换规避指针错误
    ix_start = tl.maximum(0, ((mx - radius - vol_min_x) / voxel_size).to(tl.int32))
    ix_end   = tl.minimum(dim_x, (tl.math.ceil((mx + radius - vol_min_x) / voxel_size)).to(tl.int32))
    iy_start = tl.maximum(0, ((my - radius - vol_min_y) / voxel_size).to(tl.int32))
    iy_end   = tl.minimum(dim_y, (tl.math.ceil((my + radius - vol_min_y) / voxel_size)).to(tl.int32))
    iz_start = tl.maximum(0, ((mz - radius - vol_min_z) / voxel_size).to(tl.int32))
    iz_end   = tl.minimum(dim_z, (tl.math.ceil((mz + radius - vol_min_z) / voxel_size)).to(tl.int32))

    for x in range(ix_start, ix_end):
        dx = x * voxel_size + vol_min_x - mx
        for y in range(iy_start, iy_end):
            dy = y * voxel_size + vol_min_y - my
            for z in range(iz_start, iz_end):
                dz = z * voxel_size + vol_min_z - mz
                
                offset = x * (dim_y * dim_z) + y * dim_z + z
                grad_out = tl.load(grad_grid_ptr + offset)
                
                mahal = (dx * (dx*c0 + dy*c3 + dz*c6) +
                         dy * (dx*c1 + dy*c4 + dz*c7) +
                         dz * (dx*c2 + dy*c5 + dz*c8))
                
                exp_term = tl.exp(-0.5 * mahal)
                common_factor = grad_out * opac * exp_term # todo 避免重复计算，提升运行效率
                
                # todo --------------------------------------------#
                # todo 前向传播中，公式简写为 density = opacity * exp(-0.5*mahal)，
                # todo 反向传播中，拿到一个grad_out(网格点的梯度)，需要计算出对各个参数的导数
                # todo (1) 透明度的梯度：密度(density)对透明度(opacity)是线性关系
                # 梯度累加 
                dopac += grad_out * exp_term  # todo 直观理解：如果要求网格密度更高，透明度更高一些
                # todo (2) 位置的梯度： 
                dmx += common_factor * (dx*c0 + dy*c3 + dz*c6)
                dmy += common_factor * (dx*c1 + dy*c4 + dz*c7)
                dmz += common_factor * (dx*c2 + dy*c5 + dz*c8)
                # todo (3) 形状的梯度
                mult = -0.5 * common_factor
                dc0 += mult * dx * dx; dc1 += mult * dx * dy; dc2 += mult * dx * dz
                dc3 += mult * dy * dx; dc4 += mult * dy * dy; dc5 += mult * dy * dz
                dc6 += mult * dz * dx; dc7 += mult * dz * dy; dc8 += mult * dz * dz

    # 写回梯度
    tl.store(grad_opacities_ptr + idx, dopac)
    tl.store(grad_means_ptr + idx * 3 + 0, dmx)
    tl.store(grad_means_ptr + idx * 3 + 1, dmy)
    tl.store(grad_means_ptr + idx * 3 + 2, dmz)
    
    g_c_base = idx * 9
    tl.store(grad_inv_covs_ptr + g_c_base + 0, dc0); tl.store(grad_inv_covs_ptr + g_c_base + 1, dc1)
    tl.store(grad_inv_covs_ptr + g_c_base + 2, dc2); tl.store(grad_inv_covs_ptr + g_c_base + 3, dc3)
    tl.store(grad_inv_covs_ptr + g_c_base + 4, dc4); tl.store(grad_inv_covs_ptr + g_c_base + 5, dc5)
    tl.store(grad_inv_covs_ptr + g_c_base + 6, dc6); tl.store(grad_inv_covs_ptr + g_c_base + 7, dc7)
    tl.store(grad_inv_covs_ptr + g_c_base + 8, dc8)

# todo torch.autograd.Function: 定义一个拥有自定义求导逻辑的运算单元
class GaussianSplat3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                means3d, inv_covs, opacities, vol_range, voxel_size, grid_shape):
        # todo 第一个参数ctx：自动定义的
        # means3d: (N, 3), inv_covs: (N, 3, 3), opacities: (N,)
        device = means3d.device
        N = means3d.shape[0]

        
        
        eps = 1e-6
        tiny_eps = 1e-10
        safe_inv = inv_covs + torch.eye(3, device=device).unsqueeze(0) * eps
        covs = torch.inverse(safe_inv)        
        # 取对角线方差
        variances = torch.diagonal(covs, dim1=-2, dim2=-1)
        # 确保 sqrt 的输入永远为正
        max_v = torch.clamp(variances.max(dim=-1)[0], min=tiny_eps)
        radii = 3.0 * torch.sqrt(max_v)        
        
        
        grid_density = torch.zeros(grid_shape, device=device)
        # todo Triton框架操作：定义并行规模:告诉GPU，启动N个并行线程，每个线程负责处理一个高斯点
        # todo []: Triton的语法糖，用来指定执行配置 (N,): 1D的网格布局 元组(维度扩展)
        # todo 当写下[(N,)]时，相当于告诉GPU有多少并行任务；
        
        
        _splat_fwd_kernel_opt[(N,)](
            means3d, inv_covs.reshape(N, 9), opacities,
            radii, # 传入动态半径
            grid_density,
            float(vol_range[0]), float(vol_range[1]), float(vol_range[2]),
            float(voxel_size), 
            N, int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        )
        
        ctx.save_for_backward(means3d, inv_covs, opacities, radii.detach())
        ctx.vol_info = (vol_range, voxel_size, grid_shape)
        return grid_density

    @staticmethod
    def backward(ctx, grad_output):
        means3d, inv_covs, opacities, radii = ctx.saved_tensors
        vol_range, voxel_size, grid_shape = ctx.vol_info
        
        grad_means = torch.zeros_like(means3d)
        grad_inv_covs = torch.zeros_like(inv_covs)
        grad_opacities = torch.zeros_like(opacities)
        
        N = means3d.shape[0]
        _splat_bwd_kernel[(N,)](
            grad_output,
            means3d, inv_covs.reshape(N, 9), opacities,
            radii,
            grad_means, grad_inv_covs, grad_opacities,
            float(vol_range[0]), float(vol_range[1]), float(vol_range[2]), # 确保是 float
            float(voxel_size),
            N, int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        )
        
        # todo autograd.Function要求：
        # todo backward方法中，返回值的数量和顺序必须与forward方法接收的参数数量和顺序完全相同
        return grad_means, grad_inv_covs, grad_opacities, None, None, None


def test_compare_methods():
    # 固定随机种子以便复现
    torch.manual_seed(42)
    
    device = torch.device("cuda")
    
    # 1. 设置新的参数
    # N = 6 * 112 * 200  # 高斯数：134,400 个
    N = 1800
    voxel_size = 0.4   # 网格尺寸
    
    # 2. 定义新的物理边界：[-40, -40, -1, 40, 40, 5.4]
    vol_min = torch.tensor([-40.0, -40.0, -1.0], device=device)
    vol_max = torch.tensor([40.0, 40.0, 5.4], device=device)
    vol_range = torch.cat([vol_min, vol_max]) # 合并为 [x_min, y_min, z_min, x_max, y_max, z_max]

    # 3. 动态计算网格形状 (grid_shape)
    # (max - min) / voxel_size
    dim_x = int((vol_max[0] - vol_min[0]) / voxel_size)
    dim_y = int((vol_max[1] - vol_min[1]) / voxel_size)
    dim_z = int((vol_max[2] - vol_min[2]) / voxel_size)
    grid_shape = (dim_x, dim_y, dim_z) # 结果应该是 (200, 200, 16)
    
    print(f"生成的网格分辨率为: {grid_shape}")

    # 4. 生成高斯参数：确保它们分布在新的物理边界范围内
    # 我们让均值在 vol_min 和 vol_max 之间随机分布
    means3d = (torch.rand((N, 3), device=device) * (vol_max - vol_min) + vol_min).requires_grad_(True)
    
    # 协方差保持较小，防止溢出，但相对于 0.4 的 grid_size 稍微调大一点点
    L = torch.randn((N, 3, 3), device=device) * 0.1  # 控制协方差缩放的随机项
    covs = torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 0.1 # torch.eye(3, device=device) * 0.1 基准方差 在x，y，z三个主轴上，方差0.1 
    # 方差0.1 标准差约为0.316 取±3σ, 半径约为0.948, 直径约为1.9 -> 覆盖体素数 1.9/0.4 = 4.75 约为 5^3 = 125个格点
    
    inv_covs = torch.inverse(covs).detach().requires_grad_(True)
    # 透明度
    opacities = torch.rand((N,), device=device).requires_grad_(True)

    # --- 3. Triton 版测试 ---
    # 预热 (Warm-up)
    _ = GaussianSplat3D.apply(means3d, inv_covs, opacities, vol_range, voxel_size, grid_shape)
    
    torch.cuda.synchronize()
    t0 = time.time()
    density_triton = GaussianSplat3D.apply(means3d, inv_covs, opacities, vol_range, voxel_size, grid_shape)
    loss_triton = density_triton.sum()
    loss_triton.backward()
    torch.cuda.synchronize()
    t_triton = time.time() - t0
    
    grad_means_triton = means3d.grad.clone()
    # 清空梯度准备测试原方法
    means3d.grad.zero_()
    print('grad_means_triton:',grad_means_triton)

    # --- 4. 原方法 (Python 循环) 测试 ---
    # 构建坐标网格
    lin_x = torch.linspace(vol_min[0], vol_max[0] - voxel_size, dim_x, device=device)
    lin_y = torch.linspace(vol_min[1], vol_max[1] - voxel_size, dim_y, device=device)
    lin_z = torch.linspace(vol_min[2], vol_max[2] - voxel_size, dim_z, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    torch.cuda.synchronize()
    t1 = time.time()
    density_orig = splat_into_3d(grid_coords, means3d, opacities, covs, vol_range, voxel_size)
    loss_orig = density_orig.sum()
    loss_orig.backward()
    torch.cuda.synchronize()
    t_orig = time.time() - t1
    
    grad_means_orig = means3d.grad.clone()
    print('grad_means_orig:',grad_means_orig)
    # --- 5. 结果对比 ---
    print("\n" + "="*30)
    print(f"Triton 用时: {t_triton:.6f} s")
    print(f"原方法 用时: {t_orig:.6f} s")
    print(f"加速比: {t_orig / t_triton:.2f} x")
    
    

    # 数值误差
    diff_fwd = torch.abs(density_triton - density_orig)
    diff_bwd = torch.abs(grad_means_triton - grad_means_orig)
    
    print("-" * 30)
    print(f"前向最大绝对误差: {diff_fwd.max().item():.2e}")
    print(f"前向平均相对误差: {(diff_fwd.sum() / (density_orig.sum() + 1e-7)).item():.2e}")
    print(f"均值梯度最大误差: {diff_bwd.max().item():.2e}")
    print("="*30)

if __name__ == "__main__":
    test_compare_methods()
    