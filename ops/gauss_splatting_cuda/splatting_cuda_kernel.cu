#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


// todo c++: 参数类型写法 决定程序性能，显存访问方式，以及数据安全性：指针类型：只读指针、读写指针和数值常量
// todo const float*: 只读的显存数组，输入参数：保持固定不变
// todo float* 需要修改的显存数组 
// todo 有无*： 传值(Pass by value) 与 传址(Pass by Reference/Pointer) 的区别
// todo 带有*的(指针)：传入的数据在显存中的首地址
// todo float/int: 单个数值，不可修改
// todo __global__: CUDA函数类型限定符：告诉编译器，函数由CPU调用，但在GPU上执行，必须返回void
// Forward Kernel
__global__ void _splat_fwd_kernel_opt(
    const float* means_ptr, const float* inv_covs_ptr, const float* opacities_ptr,
    const float* radii_ptr, const float* features_ptr,
    float* grid_density_ptr, float* grid_feats_ptr,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    int N_gaussians, int n_dims, 
    int dim_x, int dim_y, int dim_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_gaussians) return; // todo if语句：逻辑变量必须放到()内，逻辑块必须放到{}里(单行可省略)

    // 均值 x,y,z
    float mx = means_ptr[idx * 3 + 0]; // todo 指针：实际地址 = 基地址 + 偏移量；访问地址对应的值 *(ptr+i): *表示取出该地址值
    float my = means_ptr[idx * 3 + 1];
    float mz = means_ptr[idx * 3 + 2];

    // 协方差逆 (3x3 矩阵展平)
    const float* c_ptr = inv_covs_ptr + idx * 9; // todo 地址
    float c0 = c_ptr[0], c1 = c_ptr[1], c2 = c_ptr[2];
    float c3 = c_ptr[3], c4 = c_ptr[4], c5 = c_ptr[5];
    float c6 = c_ptr[6], c7 = c_ptr[7], c8 = c_ptr[8];

    float opac = opacities_ptr[idx];
    float rx = radii_ptr[idx * 3 + 0];
    float ry = radii_ptr[idx * 3 + 1];
    float rz = radii_ptr[idx * 3 + 2];

    // 计算 AABB 范围 (同 Triton 逻辑)
    int ix_start = max(0, (int)((mx - rx - vol_min_x) / voxel_size)); 
    int ix_end   = min(dim_x, (int)((mx + rx - vol_min_x) / voxel_size) + 1);

    int iy_start = max(0, (int)((my - ry - vol_min_y) / voxel_size));
    int iy_end   = min(dim_y, (int)((my + ry - vol_min_y) / voxel_size) + 1);

    int iz_start = max(0, (int)((mz - rz - vol_min_z) / voxel_size));
    int iz_end   = min(dim_z, (int)((mz + rz - vol_min_z) / voxel_size) + 1);

    for (int x = ix_start; x < ix_end; x++) {
        float dx = x * voxel_size + vol_min_x - mx;
        float xc0 = dx * c0; float xc1 = dx * c1; float xc2 = dx * c2;
        for (int y = iy_start; y < iy_end; y++) {
            float dy = y * voxel_size + vol_min_y - my;
            float yc3 = dy * c3; float yc4 = dy * c4; float yc5 = dy * c5;
            for (int z = iz_start; z < iz_end; z++) {
                float dz = z * voxel_size + vol_min_z - mz;

                float mahal = (dx * (xc0 + dy * c3 + dz * c6) +
                               dy * (dx * c1 + yc4 + dz * c7) +
                               dz * (dx * c2 + dy * c5 + dz * c8));

                float density = opac * expf(-0.5f * mahal);
                long long offset = (long long)x * (dim_y * dim_z) + y * dim_z + z;

                atomicAdd(grid_density_ptr + offset, density); // todo automicAdd原子加法：用于处理多个线程同时往同一个内存地址写数据(“竞态冲突”或“写冲突”)
                // todo atomicAdd(float* address, float val)：第一个值：指针，指向要修改的内存地址；第二个值：参数，要累加进取的值

                for (int f = 0; f < n_dims; f++) {
                    float feat_val = features_ptr[idx * n_dims + f];
                    atomicAdd(grid_feats_ptr + offset * n_dims + f, density * feat_val);
                }
            }
        }
    }
}

// Backward Kernel
__global__ void _splat_bwd_kernel_opt(
    float* grad_features_ptr, float* grad_opacities_ptr, float* grad_means_ptr, float* grad_inv_covs_ptr,
    const float* grid_density_ptr, const float* grid_feats_ptr,
    const float* grad_grid_density_ptr, const float* grad_grid_feats_ptr,
    const float* means_ptr, const float* inv_covs_ptr, const float* opacities_ptr, 
    const float* radii_ptr, const float* features_ptr,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    int N_gaussians, int n_dims,
    int dim_x, int dim_y, int dim_z,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_gaussians) return;

    float mx = means_ptr[idx * 3 + 0], my = means_ptr[idx * 3 + 1], mz = means_ptr[idx * 3 + 2];
    const float* c_ptr = inv_covs_ptr + idx * 9;
    float c0 = c_ptr[0], c1 = c_ptr[1], c2 = c_ptr[2];
    float c3 = c_ptr[3], c4 = c_ptr[4], c5 = c_ptr[5];
    float c6 = c_ptr[6], c7 = c_ptr[7], c8 = c_ptr[8];

    float opac = opacities_ptr[idx];
    float rx = radii_ptr[idx * 3 + 0], ry = radii_ptr[idx * 3 + 1], rz = radii_ptr[idx * 3 + 2];

    int ix_start = max(0, (int)((mx - rx - vol_min_x) / voxel_size));
    int ix_end   = min(dim_x, (int)((mx + rx - vol_min_x) / voxel_size) + 1);
    int iy_start = max(0, (int)((my - ry - vol_min_y) / voxel_size));
    int iy_end   = min(dim_y, (int)((my + ry - vol_min_y) / voxel_size) + 1);
    int iz_start = max(0, (int)((mz - rz - vol_min_z) / voxel_size));
    int iz_end   = min(dim_z, (int)((mz + rz - vol_min_z) / voxel_size) + 1);

    float grad_opac_acc = 0.0f;
    float grad_mx_acc = 0.0f, grad_my_acc = 0.0f, grad_mz_acc = 0.0f;
    float gic0 = 0, gic1 = 0, gic2 = 0, gic3 = 0, gic4 = 0, gic5 = 0, gic6 = 0, gic7 = 0, gic8 = 0;

    for (int x = ix_start; x < ix_end; x++) {
        float dx = x * voxel_size + vol_min_x - mx;
        for (int y = iy_start; y < iy_end; y++) {
            float dy = y * voxel_size + vol_min_y - my;
            for (int z = iz_start; z < iz_end; z++) {
                float dz = z * voxel_size + vol_min_z - mz;

                float mahal = (dx * (dx * c0 + dy * c3 + dz * c6) +
                               dy * (dx * c1 + dy * c4 + dz * c7) +
                               dz * (dx * c2 + dy * c5 + dz * c8));
                
                float exp_term = expf(-0.5f * mahal);
                float density = opac * exp_term;

                long long offset = (long long)x * (dim_y * dim_z) + y * dim_z + z;
                float grid_density_val = fmaxf(grid_density_ptr[offset], eps);
                float g_density_from_loss = grad_grid_density_ptr[offset];

                float dir_x = dx * c0 + dy * c1 + dz * c2;
                float dir_y = dx * c3 + dy * c4 + dz * c5;
                float dir_z = dx * c6 + dy * c7 + dz * c8;

                float v_combined_scalar = g_density_from_loss * opac;
                grad_opac_acc += g_density_from_loss * exp_term;

                for (int f = 0; f < n_dims; f++) {
                    float g_grid = grad_grid_feats_ptr[offset * n_dims + f];
                    float F_grid = grid_feats_ptr[offset * n_dims + f];
                    float f_i = features_ptr[idx * n_dims + f];

                    float grad_gauss_feat = g_grid * density / grid_density_val;
                    atomicAdd(grad_features_ptr + idx * n_dims + f, grad_gauss_feat);

                    float feat_grad_scalar = g_grid * (f_i / grid_density_val);
                    if (grid_density_ptr[offset] > eps) {
                        feat_grad_scalar -= g_grid * (F_grid / grid_density_val);
                    }
                    grad_opac_acc += feat_grad_scalar * exp_term;
                    v_combined_scalar += feat_grad_scalar * opac;
                }

                float common = v_combined_scalar * exp_term;
                grad_mx_acc += common * dir_x;
                grad_my_acc += common * dir_y;
                grad_mz_acc += common * dir_z;

                float S = common * (-0.5f);
                gic0 += S * dx * dx; gic1 += S * dx * dy; gic2 += S * dx * dz;
                gic3 += S * dy * dx; gic4 += S * dy * dy; gic5 += S * dy * dz;
                gic6 += S * dz * dx; gic7 += S * dz * dy; gic8 += S * dz * dz;
            }
        }
    }

    atomicAdd(grad_opacities_ptr + idx, grad_opac_acc);
    atomicAdd(grad_means_ptr + idx * 3 + 0, grad_mx_acc);
    atomicAdd(grad_means_ptr + idx * 3 + 1, grad_my_acc);
    atomicAdd(grad_means_ptr + idx * 3 + 2, grad_mz_acc);

    float* out_gic = grad_inv_covs_ptr + idx * 9;
    atomicAdd(out_gic + 0, gic0); atomicAdd(out_gic + 1, gic1); atomicAdd(out_gic + 2, gic2);
    atomicAdd(out_gic + 3, gic3); atomicAdd(out_gic + 4, gic4); atomicAdd(out_gic + 5, gic5);
    atomicAdd(out_gic + 6, gic6); atomicAdd(out_gic + 7, gic7); atomicAdd(out_gic + 8, gic8);
}

void _splat_fwd_kernel_opt_launcher(
    const float* means_ptr, const float* inv_covs_ptr, const float* opacities_ptr,
    const float* radii_ptr, const float* features_ptr,
    float* grid_density_ptr, float* grid_feats_ptr,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    int N_gaussians, int n_dims, 
    int dim_x, int dim_y, int dim_z
) {
    int threads = 256;
    int blocks = (N_gaussians + threads - 1) / threads;
    
    _splat_fwd_kernel_opt<<<blocks, threads>>>(
        means_ptr, inv_covs_ptr, opacities_ptr, radii_ptr, features_ptr,
        grid_density_ptr, grid_feats_ptr, vol_min_x, vol_min_y, vol_min_z,
        voxel_size, N_gaussians, n_dims, dim_x, dim_y, dim_z
    );
}

void _splat_bwd_kernel_opt_launcher(
    float* grad_features_ptr, float* grad_opacities_ptr, float* grad_means_ptr, float* grad_inv_covs_ptr,
    const float* grid_density_ptr, const float* grid_feats_ptr,
    const float* grad_grid_density_ptr, const float* grad_grid_feats_ptr,
    const float* means_ptr, const float* inv_covs_ptr, const float* opacities_ptr, 
    const float* radii_ptr, const float* features_ptr,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    int N_gaussians, int n_dims,
    int dim_x, int dim_y, int dim_z,
    float eps
) {
    int threads = 256;
    int blocks = (N_gaussians + threads - 1) / threads;

    _splat_bwd_kernel_opt<<<blocks, threads>>>(
        grad_features_ptr, grad_opacities_ptr, grad_means_ptr, grad_inv_covs_ptr,
        grid_density_ptr, grid_feats_ptr, grad_grid_density_ptr, grad_grid_feats_ptr,
        means_ptr, inv_covs_ptr, opacities_ptr, radii_ptr, features_ptr,
        vol_min_x, vol_min_y, vol_min_z, voxel_size,
        N_gaussians, n_dims, dim_x, dim_y, dim_z, eps
    );
}