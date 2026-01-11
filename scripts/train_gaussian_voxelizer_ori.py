from setproctitle import setproctitle
setproctitle("wys")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle

from debug_gaussian_voxelizer import splat_into_3d, GaussSplatting3D



if __name__=='__main__':
    torch.manual_seed(42)
    device = torch.device("cuda")
    # 1. surroundocc配置参数
    voxel_size = 0.5
    vol_min = torch.tensor([-50.0, -50.0, -5.0], device=device)
    vol_max = torch.tensor([50.0, 50.0, 3.0], device=device)
    vol_range = torch.tensor([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], device=device)
    
    dim_x = int((vol_max[0] - vol_min[0]) / voxel_size)
    dim_y = int((vol_max[1] - vol_min[1]) / voxel_size)
    dim_z = int((vol_max[2] - vol_min[2]) / voxel_size)
    grid_shape = (dim_x, dim_y, dim_z)
    
    n_class = 18
    N = 1800 # 高斯点数量    
    # todo ---------------------------------
    # 2. 加载并处理标签
    label_file = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin.npy'
    label_np = np.load(label_file)
    target_label = np.ones(grid_shape, dtype=np.int64) * 17
    # 确保索引不越界
    valid_mask = (label_np[:, 0] < 200) & (label_np[:, 1] < 200) & (label_np[:, 2] < 16)
    label_np = label_np[valid_mask]
    target_label[label_np[:, 0], label_np[:, 1], label_np[:, 2]] = label_np[:, 3]
    target_tensor = torch.from_numpy(target_label).to(device).long() # (200, 200, 16)
    
    # todo ---------------------------------#
    # todo 随机初始化参数
    # 3. 初始化可学习的高斯参数
    # 均值：在空间内随机分布
    means3d = nn.Parameter(torch.rand((N, 3), device=device) * (vol_max - vol_min) + vol_min)
    # 协方差：通过 L (Cholesky) 构造保证正定性
    L = nn.Parameter(torch.randn((N, 3, 3), device=device) * 0.2)
    # 透明度
    opacities = nn.Parameter(torch.rand((N,), device=device))
    # 特征：初始化为类别概率的 logits
    features = nn.Parameter(torch.randn((N, n_class), device=device))

    lin_x = torch.linspace(vol_min[0], vol_max[0] - voxel_size, dim_x, device=device)
    lin_y = torch.linspace(vol_min[1], vol_max[1] - voxel_size, dim_y, device=device)
    lin_z = torch.linspace(vol_min[2], vol_max[2] - voxel_size, dim_z, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)


    optimizer = optim.Adam([means3d, L, opacities, features], lr=1e-2)
    
    weights = torch.ones(n_class, device=device)
    weights[17] = 0.05
    criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("Starting training with Triton-accelerated GaussSplatting3D...")
    for i_iter in tqdm(range(200)):
        covs = torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 1e-4
        # density, pred_feats = GaussSplatting3D.apply(
        #             means3d, covs, torch.sigmoid(opacities), features, 
        #             vol_range, voxel_size, grid_shape
        #         )   
        density, pred_feats = splat_into_3d(
                grid_coords=grid_coords,
                means3d=means3d,
                opacities=torch.sigmoid(opacities),
                covariances=covs,
                vol_range=vol_range,
                voxel_size=voxel_size,
                features=features
            )
        logits = pred_feats.permute(3, 0, 1, 2).unsqueeze(0) # (200, 200, 16, 18) -> permute -> (1, 18, 200, 200, 16)
    
        target = target_tensor.unsqueeze(0)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        if i_iter % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(pred_feats, dim=-1)
                fg_mask = (target_tensor != 17)
                fg_count = fg_mask.sum().item()
                if fg_count > 0:
                    fg_acc = (preds[fg_mask] == target_tensor[fg_mask]).float().mean()
                else:
                    fg_acc = 0.0
                    
                
                total_acc = (preds == target_tensor).float().mean()
            print(f"Iter {i_iter:03d} | Loss: {loss.item():.4f} | Total Acc: {total_acc.item():.4f} | FG Acc: {fg_acc.item():.4f} | FG Pts: {fg_count}")
    # ... 训练循环结束后 ...
    print("Saving results...")
    with torch.no_grad():
        # 1. 获取最终的预测类别 (200, 200, 16)
        final_preds = torch.argmax(pred_feats, dim=-1).cpu().numpy()
    # 设定保存路径
    save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/debug/0111/vis_results_ori'
    os.makedirs(save_dir, exist_ok=True)
    output = dict(
        occ_pred=final_preds,
        occ_gt = target_tensor.cpu().numpy(),
    )
    with open(f'{save_dir}/output.pkl', 'wb') as f:

        pickle.dump(output, f)    
    
    