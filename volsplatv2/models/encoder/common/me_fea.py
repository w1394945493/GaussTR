import torch
from einops import rearrange, repeat
from ....geometry.projection import get_world_rays
from ....geometry.projection import sample_image_grid
import torch.nn.functional as F
import MinkowskiEngine as ME



def project_features_to_me(intrinsics, extrinsics, out, depth, voxel_resolution, b, v,
                           normal=False, # 是否归一化
                           img_aug_mat=None, # 图像变换矩阵
                           vol_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], # todo surroundocc中感知范围
                           ):
    device = out.device

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
    
    # 1. 展开像素点和特征（保持 batch 维度）
    pixel_points = rearrange(world_coords, "b v n c -> b (v n) c") # [b, vn, 3]
    pixel_feats = rearrange(features, "b v n c -> b (v n) c")      # [b, vn, c]    

    all_points = pixel_points
    feats_flat = pixel_feats    

    b_size, n_total, _ = all_points.shape
    batch_indices = torch.arange(b_size, device=device).reshape(b_size, 1, 1)
    batch_indices = batch_indices.repeat(1, n_total, 1).reshape(-1, 1) # [b * n_total, 1]

    all_points = all_points.reshape(-1, 3) # [b * n_total, 3] # todo 位置
    feats_flat = feats_flat.reshape(-1, c) # [b * n_total, c]
    
    # todo 2.体素化：将连续的3D点云转化为结构化的栅格特征
    with torch.no_grad():
        # todo 剔除不在occ感知范围内的点
        x_min, y_min, z_min, x_max, y_max, z_max = vol_range
        mask = (all_points[:, 0] >= x_min) & (all_points[:, 0] <= x_max) & \
                (all_points[:, 1] >= y_min) & (all_points[:, 1] <= y_max) & \
                (all_points[:, 2] >= z_min) & (all_points[:, 2] <= z_max)    
                
        
        quantized_coords = torch.round(all_points / voxel_resolution).long() # 2.1量化: 通过all_points / voxel_resolution并取整 将3D坐标映射到整数索引的体素网格
        # Create coordinate matrix: batch index + quantized coordinates 
        # batch_indices = torch.arange(b, device=device).repeat_interleave(v * h * w).unsqueeze(1) # 移到外面计算
        combined_coords = torch.cat([batch_indices, quantized_coords], dim=1) # todo (n,4): 4维内容：(bs x y z)
        
        # todo 剔除不在occ感知空间范围内的点
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

    return sparse_tensor, aggregated_points, counts