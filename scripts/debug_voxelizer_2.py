import torch
import triton
import triton.language as tl
import time

# ---------------------------------------------------------------------------- #
# 1. Forward Kernel: Gathering (嵌套 IF 规避编译器错误)
# ---------------------------------------------------------------------------- #

@triton.jit
def _fwd_kernel(
    means_ptr, inv_covs_ptr, opacities_ptr,
    tile_offsets_ptr, tile_point_indices_ptr,
    grid_ptr,
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size, 
    dim_x, dim_y, dim_z,
    tile_size_x, tile_size_y,
    stride_gx, stride_gy, stride_gz
):
    # todo 让每个线程明确自己负责计算3D网格中哪一个具体格点的密度
    idx_x = tl.program_id(0)
    idx_y = tl.program_id(1)
    idx_z = tl.program_id(2)

    if idx_x < dim_x:
        if idx_y < dim_y:
            if idx_z < dim_z:
                num_tiles_y = tl.cdiv(dim_y, tile_size_y) # todo 计算在y维度上，网格被切成了多少片：cdiv：向上取整除法 tl.cdiv(200,16)=13
                tile_id = (idx_x // tile_size_x) * num_tiles_y + (idx_y // tile_size_y) # todo 行数/总列数 + 列数 ：定位当前体素所属单元格

                offs = tile_id * 2
                start_idx = tl.load(tile_offsets_ptr + offs)
                end_idx = tl.load(tile_offsets_ptr + offs + 1) 

                vx = idx_x * voxel_size + vol_min_x
                vy = idx_y * voxel_size + vol_min_y
                vz = idx_z * voxel_size + vol_min_z
                
                res = 0.0
                for i in range(start_idx, end_idx): # todo 仅遍历start_idx到end_idx之间的点
                    p_idx = tl.load(tile_point_indices_ptr + i)
                    m_ptr = means_ptr + p_idx * 3
                    mx = tl.load(m_ptr)
                    my = tl.load(m_ptr + 1)
                    mz = tl.load(m_ptr + 2)
                    dx, dy, dz = vx - mx, vy - my, vz - mz
                    
                    # 距离裁剪也用嵌套 IF
                    if tl.abs(dx) < 1.25:
                        if tl.abs(dy) < 1.25:
                            if tl.abs(dz) < 1.25:
                                c_ptr = inv_covs_ptr + p_idx * 9
                                c0=tl.load(c_ptr+0); c1=tl.load(c_ptr+1); c2=tl.load(c_ptr+2)
                                c3=tl.load(c_ptr+3); c4=tl.load(c_ptr+4); c5=tl.load(c_ptr+5)
                                c6=tl.load(c_ptr+6); c7=tl.load(c_ptr+7); c8=tl.load(c_ptr+8)

                                mahal = (dx*(dx*c0 + dy*c3 + dz*c6) +
                                         dy*(dx*c1 + dy*c4 + dz*c7) +
                                         dz*(dx*c2 + dy*c5 + dz*c8))
                                res += tl.load(opacities_ptr + p_idx) * tl.exp(-0.5 * mahal)
                
                tl.store(grid_ptr + idx_x*stride_gx + idx_y*stride_gy + idx_z*stride_gz, res)

# ---------------------------------------------------------------------------- #
# 2. Backward Kernel: Splatting (修正嵌套与 InvCov 梯度)
# ---------------------------------------------------------------------------- #

@triton.jit
def _bwd_kernel(
    grad_grid_ptr, means_ptr, inv_covs_ptr, opacities_ptr,
    grad_means_ptr, grad_inv_covs_ptr, grad_opacities_ptr,
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size, dim_x, dim_y, dim_z,
    stride_gx, stride_gy, stride_gz
):
    p_idx = tl.program_id(0)
    m_ptr = means_ptr + p_idx * 3
    mx, my, mz = tl.load(m_ptr), tl.load(m_ptr+1), tl.load(m_ptr+2)
    opac = tl.load(opacities_ptr + p_idx)
    
    # 锁定 AABB
    min_ix = tl.maximum(0, tl.math.floor((mx - 1.25 - vol_min_x) / voxel_size).to(tl.int32))
    max_ix = tl.minimum(dim_x, tl.math.ceil((mx + 1.25 - vol_min_x) / voxel_size).to(tl.int32))
    min_iy = tl.maximum(0, tl.math.floor((my - 1.25 - vol_min_y) / voxel_size).to(tl.int32))
    max_iy = tl.minimum(dim_y, tl.math.ceil((my + 1.25 - vol_min_y) / voxel_size).to(tl.int32))
    min_iz = tl.maximum(0, tl.math.floor((mz - 1.25 - vol_min_z) / voxel_size).to(tl.int32))
    max_iz = tl.minimum(dim_z, tl.math.ceil((mz + 1.25 - vol_min_z) / voxel_size).to(tl.int32))

    c_ptr = inv_covs_ptr + p_idx * 9
    c0=tl.load(c_ptr); c1=tl.load(c_ptr+1); c2=tl.load(c_ptr+2)
    c3=tl.load(c_ptr+3); c4=tl.load(c_ptr+4); c5=tl.load(c_ptr+5)
    c6=tl.load(c_ptr+6); c7=tl.load(c_ptr+7); c8=tl.load(c_ptr+8)

    dmx, dmy, dmz, dopac = 0.0, 0.0, 0.0, 0.0
    # 为 inv_covs 准备梯度累加
    dc0, dc1, dc2, dc3, dc4, dc5, dc6, dc7, dc8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for ix in range(min_ix, max_ix):
        for iy in range(min_iy, max_iy):
            for iz in range(min_iz, max_iz):
                dx = ix * voxel_size + vol_min_x - mx
                dy = iy * voxel_size + vol_min_y - my
                dz = iz * voxel_size + vol_min_z - mz
                
                mahal = dx*(dx*c0 + dy*c3 + dz*c6) + dy*(dx*c1 + dy*c4 + dz*c7) + dz*(dx*c2 + dy*c5 + dz*c8)
                
                if mahal < 9.0:
                    g = tl.exp(-0.5 * mahal)
                    d_out = tl.load(grad_grid_ptr + ix*stride_gx + iy*stride_gy + iz*stride_gz)
                    common = d_out * g
                    dopac += common
                    
                    # Mean 梯度
                    fac = common * opac
                    dmx += fac * (dx*c0 + dy*c3 + dz*c6)
                    dmy += fac * (dx*c1 + dy*c4 + dz*c7)
                    dmz += fac * (dx*c2 + dy*c5 + dz*c8)
                    
                    # InvCov 梯度: dG/dInvCov = -0.5 * G * (V-M)(V-M)^T
                    inv_fac = -0.5 * fac
                    dc0 += inv_fac * dx * dx; dc1 += inv_fac * dx * dy; dc2 += inv_fac * dx * dz
                    dc3 += inv_fac * dy * dx; dc4 += inv_fac * dy * dy; dc5 += inv_fac * dy * dz
                    dc6 += inv_fac * dz * dx; dc7 += inv_fac * dz * dy; dc8 += inv_fac * dz * dz

    tl.atomic_add(grad_opacities_ptr + p_idx, dopac)
    tl.atomic_add(grad_means_ptr + p_idx*3 + 0, dmx)
    tl.atomic_add(grad_means_ptr + p_idx*3 + 1, dmy)
    tl.atomic_add(grad_means_ptr + p_idx*3 + 2, dmz)
    
    # 写回 InvCov 梯度
    ic_ptr = grad_inv_covs_ptr + p_idx * 9
    tl.atomic_add(ic_ptr + 0, dc0); tl.atomic_add(ic_ptr + 1, dc1); tl.atomic_add(ic_ptr + 2, dc2)
    tl.atomic_add(ic_ptr + 3, dc3); tl.atomic_add(ic_ptr + 4, dc4); tl.atomic_add(ic_ptr + 5, dc5)
    tl.atomic_add(ic_ptr + 6, dc6); tl.atomic_add(ic_ptr + 7, dc7); tl.atomic_add(ic_ptr + 8, dc8)

# ---------------------------------------------------------------------------- #
# 3. Autograd Wrapper & Test Script
# ---------------------------------------------------------------------------- #

class GaussianSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, inv_covs, opacities, 
                vol_range,  # todo (-40,-40,-1,40,40,5.4)
                voxel_size, # todo 0.4
                grid_shape, # todo (200,200,16) 体素网格尺寸
                tile_size): # todo 16
        device = means.device
        grid = torch.zeros(grid_shape, device=device)
        
        # todo------------------------------#
        # todo title_size(分块大小) 利用GPU的局部缓存并减少冗余计算
        # todo title_size: 定义在X和Y维度上，将3D网格切分成多个小方块
        # 预处理：生成 Tile 映射
        m_idx = (means - vol_range[:3]) / voxel_size # todo 计算映射坐标，将高斯点的位置转换为网格索引
        num_tiles_y = (grid_shape[1] + tile_size - 1) // tile_size # todo 在y轴上可划分得到多少title
        tx = (m_idx[:, 0] / tile_size).int().clamp(0)
        ty = (m_idx[:, 1] / tile_size).int().clamp(0, num_tiles_y - 1)
        tile_ids = tx * num_tiles_y + ty
        sorted_ids, sort_idx = torch.sort(tile_ids) # todo 重新排序，使得同一个Title里的点在内存中连续存放
        
        total_tiles = ((grid_shape[0]+tile_size-1)//tile_size) * num_tiles_y
        offsets = torch.zeros((total_tiles, 2), dtype=torch.int32, device=device)
        if len(sorted_ids) > 0:
            uid, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
            ends = torch.cumsum(counts, dim=0)
            offsets[uid.long(), 0] = (ends - counts).int()
            offsets[uid.long(), 1] = ends.int()
        # todo grid_shape: (200,200,16)
        _fwd_kernel[grid_shape](
            means, inv_covs, opacities, 
            offsets, sort_idx.int(), grid,
            vol_range[0].item(), vol_range[1].item(), vol_range[2].item(),
            voxel_size, 
            grid_shape[0], grid_shape[1], grid_shape[2],
            tile_size, tile_size, grid.stride(0), grid.stride(1), grid.stride(2)
        )
        ctx.save_for_backward(means, inv_covs, opacities)
        ctx.info = (vol_range, voxel_size, grid_shape)
        return grid

    @staticmethod
    def backward(ctx, grad_output):
        means, inv_covs, opacities = ctx.saved_tensors
        vol_range, voxel_size, grid_shape = ctx.info
        grad_means = torch.zeros_like(means)
        grad_inv_covs = torch.zeros_like(inv_covs)
        grad_opacities = torch.zeros_like(opacities)
        
        _bwd_kernel[(means.shape[0],)](
            grad_output.contiguous(), means, inv_covs, opacities,
            grad_means, grad_inv_covs, grad_opacities,
            vol_range[0].item(), vol_range[1].item(), vol_range[2].item(),
            voxel_size, grid_shape[0], grid_shape[1], grid_shape[2],
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2)
        )
        return grad_means, grad_inv_covs, grad_opacities, None, None, None, None

def run_test():
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    N = 6 * 112 * 200  # 高斯数：134,400 个
    voxel_size = 0.4   # 网格尺寸
    grid_shape = (200, 200, 16)
    
    # 体素空间范围
    vol_min = torch.tensor([-40.0, -40.0, -1.0], device=device)
    vol_max = torch.tensor([40.0, 40.0, 5.4], device=device)
    vol_range = torch.cat([vol_min, vol_max]) # 合并为 [x_min, y_min, z_min, x_max, y_max, z_max]

    means = (torch.rand((N, 3), device=device) * (vol_max - vol_min) + vol_min).requires_grad_(True)
    

    # 协方差保持较小，防止溢出，但相对于 0.4 的 grid_size 稍微调大一点点
    L = torch.randn((N, 3, 3), device=device) * 0.1  # 控制协方差缩放的随机项
    covs = torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 0.1 # torch.eye(3, device=device) * 0.1 基准方差 在x，y，z三个主轴上，方差0.1 
    inv_covs =  torch.inverse(covs).detach().reshape(-1, 9).contiguous().requires_grad_(True)
    opacities = torch.rand(N, device=device, requires_grad=True)

    # # 预热并测试端到端性能
    # GaussianSplatFunction.apply(means, inv_covs, opacities, vol_range, voxel_size, grid_shape, 16)
    # torch.cuda.synchronize()
    
    
    for _ in range(10):
        t0 = time.time()
        out = GaussianSplatFunction.apply(means, inv_covs, opacities, vol_range, voxel_size, grid_shape, 16)
        out.sum().backward()
        t_triton = time.time() - t0
        print(f"Triton 时间: {t_triton:.6f}s")
    # torch.cuda.synchronize()
    # print(f"13.4w 点训练步平均耗时: {(time.time()-t0)/10:.4f}s")

if __name__ == "__main__":
    run_test()