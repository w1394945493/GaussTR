import torch
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

import MinkowskiEngine as ME
import spconv.pytorch as spconv 
from .projection import get_world_rays,sample_image_grid


def project_features(intrinsics, extrinsics, out, depth,
                    normal=False, # 是否归一化
                    img_aug_mat=None, # 图像变换矩阵                    
                    ):
    
    device = out.device
    b, v = intrinsics.shape[:2]
    h, w = depth.shape[2:]
    _, c, _, _ = out.shape
    # todo 1.坐标变换：从2D像素到3D世界坐标：利用相机内外参和深度图，将图像上每个像素点转换到世界坐标系中
    intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")
    extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
    depths = rearrange(depth, "b v h w -> b v (h w) () ()")

    coordinates = sample_image_grid((h, w), device,normal=normal)[0]
    
    
    if img_aug_mat is not None:
        coordinates = repeat(coordinates, "h w c -> 1 v (h w) c", v=v)
        # 齐次化 (b, v, n, 3)
        coordinates = torch.cat([coordinates,  torch.ones_like(coordinates[..., :1])], dim=-1) # (b, v, n, 3)
        post_rots = img_aug_mat[..., :3, :3] # (b,v,3,3)
        post_trans = img_aug_mat[..., :3, 3] # (b,v,3)    
        # 逆平移
        coordinates = coordinates - post_trans.unsqueeze(-2)
        # 逆旋转 (b, v, n, 3)
        coordinates = (torch.inverse(post_rots).unsqueeze(2) @ coordinates.unsqueeze(-1)).squeeze(-1)
        # 去掉齐次位并对齐维度 (b, v, n, 1, 1, 2)
        coordinates = rearrange(coordinates[...,:-1], "b v n xy -> b v n () () xy")
    else:
        coordinates = repeat(coordinates, "h w c -> 1 v (h w) () () c", v=v)
    
    
    origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
    world_coords = origins + directions * depths[..., None] # todo 计算得到每个像素的3D坐标
    world_coords = world_coords.squeeze(3).squeeze(3)  # [B, V, N, 3]

    features = rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v)
    features = rearrange(features, "b v c h w -> b v h w c")
    features = rearrange(features, "b v h w c -> b v (h w) c")  # [B, V, N, C]    

    pixel_points = rearrange(world_coords, "b v n c -> b (v n) c") # [b, vn, 3]
    pixel_feats = rearrange(features, "b v n c -> b (v n) c")      # [b, vn, c]    
    return pixel_points, pixel_feats






def project_features_to_me(intrinsics, extrinsics, out, depth, voxel_resolution,
                           normal=False, # 是否归一化
                           img_aug_mat=None, # 图像变换矩阵
                           vol_range = None, # todo surroundocc中感知范围
                           pixel_flag = False, # 是否返回像素特征及3D位置
                           ):
    
    # 将像素特征反投影至3D空间
    pixel_points,pixel_feats = project_features(intrinsics, extrinsics, out, depth,
                           normal=normal, # 是否归一化
                           img_aug_mat=img_aug_mat, # 图像变换矩
                           )
    
    device = out.device
    _, c, _, _ = out.shape

    all_points = pixel_points
    feats_flat = pixel_feats    

    b_size, n_total, _ = all_points.shape
    batch_indices = torch.arange(b_size, device=device).reshape(b_size, 1, 1)
    batch_indices = batch_indices.repeat(1, n_total, 1).reshape(-1, 1) # [b * n_total, 1]
    
    
    all_points = all_points.reshape(-1, 3) # [b * n_total, 3] # 展平
    feats_flat = feats_flat.reshape(-1, c) # [b * n_total, c]
    
    # todo 2.体素化：将连续的3D点云转化为结构化的栅格特征
    with torch.no_grad():
    
        quantized_coords = torch.round(all_points / voxel_resolution).long() # 2.1量化: 通过all_points / voxel_resolution并取整 将3D坐标映射到整数索引的体素网格
        # Create coordinate matrix: batch index + quantized coordinates 
        # batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w).unsqueeze(1) # 移到外面计算
        combined_coords = torch.cat([batch_indices, quantized_coords], dim=1) # todo (n,4): 4维内容：(bs x y z)
        
        # todo 剔除不在occ感知空间范围内的点
        if vol_range is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = vol_range
            mask = (all_points[:, 0] >= x_min) & (all_points[:, 0] <= x_max) & \
                    (all_points[:, 1] >= y_min) & (all_points[:, 1] <= y_max) & \
                    (all_points[:, 2] >= z_min) & (all_points[:, 2] <= z_max)           
            combined_coords = combined_coords[mask]
        
        # 2.2唯一化：去重，找出所有被占据的唯一体素坐标，并记录每个体素包含多少个原始点，以及映射关系
        # Get unique voxel IDs and mapping indices
        unique_coords, inverse_indices, counts = torch.unique(
            combined_coords,
            dim=0,
            return_inverse=True,
            return_counts=True
        )
    
    # todo 剔除不在occ感知范围内的点
    if vol_range is not None:
        all_points = all_points[mask]
        feats_flat = feats_flat[mask]
        
    
    # todo 3. 特征聚合：将落在同一个3D体素内的像素点的特征进行聚合
    num_voxels = unique_coords.shape[0]
    # todo 3.1 将落在同一体素内的所有像素特征累加并取平均
    aggregated_feats = torch.zeros(num_voxels, c, device=device)
    aggregated_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, c), feats_flat)
    aggregated_feats = aggregated_feats / counts.view(-1, 1).float()  # Average features
    # todo 3.2 位置计算：同样方式：计算出每个体素内所有原始3D点的平均位置
    aggregated_points = torch.zeros(num_voxels, 3, device=device)
    aggregated_points.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), all_points) # todo all_points: 体素点在真实空间中位置
    aggregated_points = aggregated_points / counts.view(-1, 1).float()
    
    # todo----------------------------------------------------------------------------------------------#
    # todo 4. 构建稀疏张量：
    # todo MinkowskiEngine(ME): 主要用于处理三维点云或稀疏网格数据
    # todo 作用：在处理三维数据时，如果使用传统的卷积神经网络（CNN），需要建立一个巨大的立方体网格，例如，稠密张量（Dense）：在一个 1000x1000x1000 的网格里，即使只有几个物体，也要存储 10 亿个体素。这会迅速耗尽显存。
    # todo 稀疏张量（Sparse）：只存储有数据的体素坐标和特征。MinkowskiEngine 允许神经网络只在这些有值的点上进行计算，极大节省了空间。
    # todo ME.SparseTensor：主要接收的关键输入：坐标,即非空体素在3D空间的位置；特征：每个坐标对应的向量信息
    #?--------------------------------------------------------------------------------------------------#
    #? 学习一下MinkowskiEngine的使用：将像素特征 -> 体素空间中的特征，并使用ME进行后续处理
    # Use correct coordinate format: batch index + quantized coordinates
    
    
    
    sparse_tensor = ME.SparseTensor(
        features=aggregated_feats, # todo 非空体素在3D空间的位置
        coordinates=unique_coords.int(), # todo 每个坐标对应的向量信息 (N,4) -> [batch_index, x, y, z] bs + 体素点在体素网格中的索引
        tensor_stride=1,
        device=device
    )
    
    '''
    # debug 可视化初始的体素网格
    mask_3d = sparse_to_dense_mask(sparse_tensor, vol_range, voxel_resolution)
    import pickle
    data={
        'mask': mask_3d,
        'vol_range': vol_range,
        'voxel_resolution':voxel_resolution,
    }
    save_path = 'mask_3d.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    '''
    
    #? 使用spconv提供的库方法进行稀疏3D卷积
    # spatial_shape = unique_coords[:, 1:].max(0)[0].tolist()
    # spatial_shape = [s + 1 for s in spatial_shape]
    # sparse_tensor = spconv.SparseConvTensor(
    #     features=aggregated_feats,
    #     indices=unique_coords.int(),
    #     spatial_shape=spatial_shape, # todo 需要预先定义网格的最大尺寸
    #     batch_size=b,
    # )
    if pixel_flag:
        return sparse_tensor, aggregated_points, counts, pixel_points, pixel_feats
    
    return sparse_tensor, aggregated_points, counts

# Debug调试用：可视化初始的体素网格
def sparse_to_dense_mask(sparse_tensor, vol_range, voxel_resolution,bs=0):
    """
    将 SparseTensor 转换为 (H, W, Z) 的 0/1 数组
    """
    # 1. 计算网格的维度尺寸 (Grid Shape)
    # H = (x_max - x_min) / res, W = (y_max - y_min) / res, Z = (z_max - z_min) / res
    x_min, y_min, z_min, x_max, y_max, z_max = vol_range
    
    grid_h = int(round((x_max - x_min) / voxel_resolution))
    grid_w = int(round((y_max - y_min) / voxel_resolution))
    grid_z = int(round((z_max - z_min) / voxel_resolution))
    
    print(f"目标网格尺寸: {grid_h} x {grid_w} x {grid_z}")

    # 2. 提取第一个 Batch 的坐标 (N, 4) -> [b, x, y, z]
    coords = sparse_tensor.C
    mask = coords[:, 0] == bs
    # 这里的 batch_coords 是量化后的整数索引
    batch_coords = coords[mask, 1:].long() 

    # 3. 将量化坐标平移到从 0 开始的索引空间
    # 注意：unique_coords 是用 all_points / voxel_resolution 算的
    # 所以索引 0 对应的是物理空间 0m 处。
    # 我们需要把 [-50, 50] 映射到 [0, 250]
    offset = torch.tensor([x_min, y_min, z_min], device=coords.device) / voxel_resolution
    grid_indices = batch_coords - torch.round(offset).long()

    # 4. 剔除越界的点（防止因为浮点误差导致的索引越界）
    valid_mask = (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_h) & \
                 (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_w) & \
                 (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < grid_z)
    grid_indices = grid_indices[valid_mask]

    # 5. 填充数组
    # 使用 torch 并在最后转 numpy，速度最快
    dense_voxels = torch.zeros((grid_h, grid_w, grid_z), dtype=torch.uint8, device=coords.device)
    dense_voxels[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]] = 1

    return dense_voxels.cpu().numpy()