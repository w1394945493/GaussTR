import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from mmdet3d.registry import MODELS

from torch_scatter import scatter_add, scatter_max

from tqdm import tqdm

from .utils import (apply_to_items, generate_grid, get_covariance,
                    quat_to_rotmat, unbatched_forward)









@MODELS.register_module()
class GaussianVoxelizer(nn.Module):

    def __init__(self,
                 vol_range,
                 voxel_size,
                 scale_multiplier=3,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 covariance_thresh=0):
        super().__init__()
        self.voxel_size = voxel_size # todo 0.4
        vol_range = torch.tensor(vol_range) # todo [-40 -40 -1 40 40 5.4]
        self.register_buffer('vol_range', vol_range)

        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        grid_coords = generate_grid(self.grid_shape, offset=0.5) # todo 网格点
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh

        # todo (wys) 参考GaussianFormer 高斯到体素化模块
        self.empty_label = 17
        self.num_classes = 18
        self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0) # todo nn.Parameter: 可训练参数
        self.register_buffer('empty_mean', torch.tensor([0, 0, 2.2])[None, None, :]) # todo register_buffer: 跟着模型走，但不训练的状态量
        self.register_buffer('empty_scale', torch.tensor([100, 100, 8.0])[None, None, :])
        self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
        self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
        self.register_buffer('empty_opa', torch.ones(1)[None, None, :])


        pc_min = vol_range[:3].tolist()
        H,W,D,_ = grid_coords.shape
        import local_aggregate
        self.aggregator = local_aggregate.LocalAggregator(scale_multiplier=scale_multiplier, # todo scale_multiplier: 3 (±3σ范围)
                                                          H=H, W=W, D=D, # todo 体素空间 H:200 W:200 D:16
                                                          pc_min=pc_min, # todo [-40 -40 -1]
                                                          grid_size=voxel_size) # todo 体素尺寸: 0.4
        # self.aggregator.requires_grad_(False)

    @unbatched_forward
    def forward(self,
                means3d,
                opacities,
                features=None,
                scales=None,
                rotations=None,
                covariances=None,
                **kwargs):
        # if covariances is None:
        #     covariances = get_covariance(scales, quat_to_rotmat(rotations))

        gaussians = dict(
            means3d=means3d,
            opacities=opacities,
            features=features,
            scales = scales,
            rotations = rotations,
            covariances=covariances,
            **kwargs)

        '''
        import numpy as np
        # 保存为 numpy
        np.save("means3d_query.npy", means3d.cpu().numpy())
        np.save("vol_range.npy", self.vol_range.cpu().numpy())

        print("保存完毕: means3d.npy, vol_range.npy")
        '''
        # todo 对高斯点进行过滤
        if self.filter_gaussians: # todo True
            # todo 1.过滤透明度小于0.6的高斯点
            mask = opacities.squeeze(1) > self.opacity_thresh # todo opacity_thresh = 0.6
            # todo 2. 过滤在[-40, -40, -1, 40, 40, 5.4]之外的高斯点
            for i in range(3):
                mask &= (means3d[:, i] >= self.vol_range[i]) & (
                    means3d[:, i] <= self.vol_range[i + 3])
            # todo 3. 暂未用到
            if self.covariance_thresh > 0: # todo covariance_thresh=0.0 预测的高斯点的尺度都太小了
                cov_diag = torch.diagonal(covariances, dim1=1, dim2=2)
                mask &= ((cov_diag.min(1)[0] * 6) > self.covariance_thresh)
            gaussians = apply_to_items(lambda x: x[mask], gaussians)

        # todo 高斯到体素投影
        return self.splat_into_3d(
            self.grid_coords, # todo 网格坐标：(L W H 3)
            **gaussians,
            vol_range=self.vol_range,
            voxel_size=self.voxel_size)


    def splat_into_3d(self,
                    grid_coords,
                    means3d,
                    opacities,
                    features,
                    scales,
                    rotations,
                    covariances,
                    vol_range,
                    voxel_size,
                    eps=1e-6):

        # todo ----------------------------------#
        # todo 参考AnySplat中的体素化操作与GaussianFormer的LocalAggregator
        H,W,D = self.grid_shape # 200 200 16
        grid_density = torch.zeros((*grid_coords.shape[:-1], 1),device=grid_coords.device) # todo 全为0(没有用到，为了保持与GaussTR原代码格式一致)
         # todo 1. 将 3D 坐标 (means3d) 转换为体素索引。通过减去体素的最小值并除以体素大小来计算
        voxel_indices = ((means3d - vol_range[:3]) / voxel_size).round().int()
        # 限制体素索引在合理范围内，确保不会越界
        voxel_indices = torch.clamp(voxel_indices, min=torch.tensor(0, device=voxel_indices.device), max=torch.tensor([H-1, W-1, D-1], device=voxel_indices.device))
        # todo 2. 获取去重后的体素索引、原始体素索引对应去重后的位置、以及每个体素位置的高斯点的数量
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        ) # todo unique_voxels: 唯一的体素索引(去重后的结果) inverse_indices: 原始的体素索引对应到unique_voxels中的索引，返回和voxel_indices长度一样的张量，表示每个voxel_indices中每个元素在unique_voxels中的位置
        conf_flat = opacities.flatten() # todo 使用透明度作为置信度依据
        # todo 3. 根据每个体素位置的透明度进行聚合，找到每个位置的最大透明度值
        max_values, max_indices = scatter_max(conf_flat, inverse_indices, dim=0) # todo scatter_max: 按照指定的索引对张量进行聚合操作 scatter_max(src,index,dim): src: 你想要聚合的张量 index：和src相同大小的张量 dim：指定在哪个维度聚合
        sampled_xyz = rearrange(grid_coords,"H W D xyz -> (H W D) xyz").unsqueeze(0) # todo (1 200x200x16 3)
        # cov_inv = covariances[max_indices].inverse().unsqueeze(0) # todo N ->(滤除) n  (1 n 3 3) 只取密度(透明度)最大的高斯点
        # scales = torch.sqrt(covariances[max_indices].diagonal(dim1=1, dim2=2)).unsqueeze(0) # todo (1 n 3)
        # covs = covariances[max_indices].unsqueeze(0)
        # todo 4. 根据透明度筛选透明度最大的高斯点
        means = means3d[max_indices].unsqueeze(0) # todo (1 n 3)
        origi_opa = opacities[max_indices].unsqueeze(0) # todo (1 n)
        feats = features[max_indices].unsqueeze(0) # todo 语义预测 (1 n n_classes)
        gs_scales = scales[max_indices].unsqueeze(0)
        gs_rotations = rotations[max_indices].unsqueeze(0)
        # todo 5. 拼接上了这个空的高斯点
        feats = torch.cat([feats,torch.zeros_like(feats[...,:1])],dim=-1)
        means = torch.cat([means, self.empty_mean], dim=1)
        gs_scales = torch.cat([gs_scales, self.empty_scale], dim=1)
        gs_rotations = torch.cat([gs_rotations, self.empty_rot], dim=1)
        empty_sem = self.empty_sem.clone()
        empty_sem[..., self.empty_label] += self.empty_scalar
        feats = torch.cat([feats,empty_sem],dim=1)
        origi_opa = torch.cat([origi_opa,self.empty_opa],dim=1).squeeze(-1)
        # todo 6. 重新计算了一下协方差和相应的逆矩阵(没有使用相机外参)
        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = gs_scales[..., 0]
        S[..., 1, 1] = gs_scales[..., 1]
        S[..., 2, 2] = gs_scales[..., 2]
        R = quat_to_rotmat(gs_rotations)
        M = torch.matmul(S, R)
        covs = torch.matmul(M.transpose(-1, -2), M)
        # covs_inv = covs.cpu().inverse().cuda()
        covs_inv = covs.inverse()
        # todo 6. 利用Gaussformer的local aggregator进行语义信息聚合
        semantics = self.aggregator(sampled_xyz,means,origi_opa,feats,gs_scales,covs_inv) # todo 输出的semantics (200x200x16 n_classes)
        grid_feats = rearrange(semantics,"(H W D) dim -> H W D dim",H=H,W=W,D=D) # todo (200 200 16 n_classes)
        return grid_feats, grid_density


        # todo ----------------------------------#
        # todo 参照AnySplat中的体素化操作：未考虑尺度、协方差等信息了
        H,W,D = self.grid_shape # 200 200 16
        grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                            device=grid_coords.device)

        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                device=grid_coords.device)

        voxel_indices = ((means3d - vol_range[:3]) / voxel_size).round().int() # todo 将 3D 坐标 (means3d) 转换为体素索引。通过减去体素的最小值并除以体素大小来计算
        # 限制体素索引在合理范围内，确保不会越界
        voxel_indices = torch.clamp(voxel_indices, min=torch.tensor(0, device=voxel_indices.device), max=torch.tensor([H-1, W-1, D-1], device=voxel_indices.device))

        # todo 获取去重后的体素索引、原始体素索引对应去重后的位置、以及每个体素位置的高斯点的数量
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        ) # todo unique_voxels: 唯一的体素索引(去重后的结果) inverse_indices: 原始的体素索引对应到unique_voxels中的索引，返回和voxel_indices长度一样的张量，表示每个voxel_indices中每个元素在unique_voxels中的位置

        conf_flat = opacities.flatten() # todo 使用透明度作为置信度依据
        # todo 根据每个体素位置的透明度进行聚合，找到每个位置的最大透明度值
        max_values, max_indices = scatter_max(conf_flat, inverse_indices, dim=0) # todo scatter_max: 按照指定的索引对张量进行聚合操作 scatter_max(src,index,dim): src: 你想要聚合的张量 index：和src相同大小的张量 dim：制定者那个维度聚合
        # 计算每个点的透明度与该位置最大透明度之间的差值，再计算其指数值
        conf_exp = torch.exp(conf_flat - max_values[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0) # todo 按照指定索引进行求和

        # 计算每个点的权重，即透明度指数差值与该体素位置的总透明度加权和的比值
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1) # todo 作为权重 (num_gaussians,1)
        weighted_feats = features.squeeze(1) * weights # todo (num_gaussians,num_classes)
        # 按照 inverse_indices 对加权后的特征进行加和，得到每个体素位置的最终特征
        feats = scatter_add(weighted_feats, inverse_indices, dim=0)

        # todo 将加权融合后的特征放到相应的体素位置上
        flat_indices = unique_voxels[:, 0] * (W * D) + unique_voxels[:, 1] * D + unique_voxels[:, 2]  # 将 3D 坐标转换为 1D 索引，flat_indices 为每个唯一体素位置在一维数组中的索引
        grid_feats = rearrange(grid_feats,'H W D C -> (H W D) C') # 将 grid_feats 从 (H, W, D, C) 形状转换为 (H * W * D, C)
        grid_feats = scatter_add(feats, flat_indices, dim=0, out=grid_feats)  # 使用 scatter_add 根据 flat_indices 将加权后的特征放到正确的位置
        grid_feats = rearrange(grid_feats,'(H W D) C -> H W D C',H=H,W=W,D=D)

        opas_mean = scatter_add(opacities,inverse_indices,dim=0) / counts[:,None] # todo 计算各体素位置的平均透明度值
        grid_density = rearrange(grid_density,'H W D C -> (H W D) C')
        grid_density = scatter_add(opas_mean, flat_indices, dim=0, out=grid_density) # 使用 scatter_add 根据 flat_indices 将平均透明度值放到正确的位置
        grid_density = rearrange(grid_density,'(H W D) C -> H W D C',H=H,W=W,D=D)

        return grid_feats, grid_density



        # todo --------------------------------------------#
        # todo GaussianFormer中的高斯体素化操作：当高斯点数量多时，计算开销会很大
        grid_density = torch.ones((*grid_coords.shape[:-1], 1), device=grid_coords.device)

        sampled_xyz = rearrange(grid_coords,"H W D xyz -> (H W D) xyz").unsqueeze(0) # todo (1 200x200x16 3)
        cov_inv = covariances.inverse().unsqueeze(0) # todo (1 n 3 3)
        means = means3d.unsqueeze(0) # todo (1 n 3)
        opacities = opacities.unsqueeze(0).squeeze(-1) # todo (1 n)
        features = features.unsqueeze(0) # todo (1 n n_classes)
        scales = torch.sqrt(covariances.diagonal(dim1=1, dim2=2)).unsqueeze(0) # todo (1 n 3)

        semantics = self.aggregator(
            sampled_xyz,
            means,
            opacities,
            features,
            scales,
            cov_inv,
        ) # todo (200x200x16 n_classes)
        H,W,D = self.grid_shape
        grid_feats = rearrange(semantics,"(H W D) dim -> H W D dim",H=H,W=W,D=D) # todo (200 200 16 n_classes)

        return grid_feats,grid_density



        # todo----------------------------------------------------#
        # todo GaussTR中的工作：逐高斯点体素化
        grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                                device=grid_coords.device) # todo (200 200 16 1)

        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                device=grid_coords.device) # todo (200 200 16 n_classes)
        sigmas = torch.sqrt(covariances.diagonal(dim1=-2, dim2=-1))
        factors = 3 * torch.tensor([-1, 1]).to(sigmas.device).view(2, 1).expand(2, means3d.size(0)).T  # (num_gaussian, 2)
        bounds_all = means3d[:, None, :] + factors[:, :, None] * sigmas[:, None, :]
        bounds_all = bounds_all.clamp(vol_range[:3], vol_range[3:])  # (N,2,3)
        bounds_indices = ((bounds_all - vol_range[:3]) / voxel_size).long()
        covariances_inv = covariances.inverse()

        # todo 逐点循环，高斯点数量很多时会很慢(待优化): 仍然速度很慢：6x112x192个高斯点处理需几分钟以上
        for g in tqdm(range(means3d.size(0))): # todo means3d: (n,3) n: 高斯点数量
            # 计算sigma、bounding box(用1D标准差近似球/椭球支撑域)
            # sigma = torch.sqrt(torch.diag(covariances[g])) # 提取协方差的对焦元素 (3)
            # factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma) # 取 ±3σ作为截断范围(≈99.7%高斯质量)
            # bounds = means3d[g, None] + factor * sigma[None] # 得到高斯在世界坐标系下的轴向最小/最大坐标 (2,3) ±3σ

            bounds = bounds_all[g] # (2 3)
            if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                    ((bounds < vol_range[None, 3:]).max(0).values.min())):
                continue # 若高斯的3sigma区间完全落在体素体积外，则跳过该高斯

            # 找到与该高斯相交的voxels区域
            # bounds = bounds.clamp(vol_range[:3], vol_range[3:]) # 把bounds限制到体积边界范围内
            # bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist() # 转换成体素索引 idx = (x-x_min) / voxel_size

            bounds = bounds_indices[g].int().tolist() # (2 3) -> list[[x_start y_start z_start] [x_end y_end z_end]]
            slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)]) # slices: 三元组：((x_start,x_end,None),(y_start,y_end,None),(z_start,z_end,None)) None表示默认步长为1

            # 计算每个体素中心与高斯中心的马氏距离
            diff = grid_coords[slices] - means3d[g] # (x,y,z,3) -(3) = (x y z 3)
            # cov_inv = covariances[g].inverse()
            cov_inv = covariances_inv[g] # (3 3)
            maha_dist = (diff.unsqueeze(-2) @ cov_inv
                        @ diff.unsqueeze(-1)).squeeze(-1) # (x y z 1 3) @ (3 3) @ (x y z 3 1)

            density = opacities[g] * torch.exp(-0.5 * maha_dist) # todo 并使用高斯权重
            # 向3D网格累计密度，区域内所有体素都加上该高斯的权重
            grid_density[slices] += density

            if features is not None:
                grid_feats[slices] += density * features[g]

        if features is not None:
            grid_feats /= grid_density.clamp(eps)

        return grid_feats, grid_density