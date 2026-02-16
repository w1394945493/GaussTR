#include <torch/extension.h>
#include <vector>

// 1. 【必须】在这里声明你在 .cu 文件里定义的两个 Launcher 函数
// 这样 CPP 才知道去哪里找它们
void _splat_fwd_kernel_opt_launcher(
    const float* means_ptr, const float* inv_covs_ptr, const float* opacities_ptr,
    const float* radii_ptr, const float* features_ptr,
    float* grid_density_ptr, float* grid_feats_ptr,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    int N_gaussians, int n_dims, 
    int dim_x, int dim_y, int dim_z
);

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
);

// 2. 绑定到 Python 的函数
at::Tensor splat_forward(
    at::Tensor means, at::Tensor inv_covs, at::Tensor opacities, at::Tensor radii, at::Tensor features, 
    at::Tensor grid_density, at::Tensor grid_feats, 
    float vx, float vy, float vz, float vs
) {
    // 直接调用我们在 .cu 中写好的 launcher
    _splat_fwd_kernel_opt_launcher(
        means.data_ptr<float>(), inv_covs.data_ptr<float>(), opacities.data_ptr<float>(), 
        radii.data_ptr<float>(), features.data_ptr<float>(),
        grid_density.data_ptr<float>(), grid_feats.data_ptr<float>(), 
        vx, vy, vz, vs, 
        means.size(0), features.size(1), 
        grid_density.size(0), grid_density.size(1), grid_density.size(2)
    );
    return grid_feats;
}

void splat_backward(
    at::Tensor g_feats, at::Tensor g_opac, at::Tensor g_means, at::Tensor g_inv_covs,
    at::Tensor grid_density, at::Tensor grid_feats, at::Tensor g_grid_density, at::Tensor g_grid_feats,
    at::Tensor means, at::Tensor inv_covs, at::Tensor opacities, at::Tensor radii, at::Tensor features,
    float vx, float vy, float vz, float vs, float eps
) {
    // 同理，直接调用 bwd 的 launcher
    _splat_bwd_kernel_opt_launcher(
        g_feats.data_ptr<float>(), g_opac.data_ptr<float>(), g_means.data_ptr<float>(), g_inv_covs.data_ptr<float>(),
        grid_density.data_ptr<float>(), grid_feats.data_ptr<float>(), 
        g_grid_density.data_ptr<float>(), g_grid_feats.data_ptr<float>(),
        means.data_ptr<float>(), inv_covs.data_ptr<float>(), opacities.data_ptr<float>(), 
        radii.data_ptr<float>(), features.data_ptr<float>(),
        vx, vy, vz, vs, 
        means.size(0), features.size(1), 
        grid_density.size(0), grid_density.size(1), grid_density.size(2), 
        eps
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &splat_forward, "Splat Forward");
    m.def("backward", &splat_backward, "Splat Backward");
}