from setproctitle import setproctitle
setproctitle("wys")

import time 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
from terminaltables import AsciiTable

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_gaussian_voxelizer_cuda import GaussSplatting3DCuda
from lovzsz_softmax import lovasz_softmax

def quaternion_to_rotation_matrix(q):
    """将四元数 (w, x, y, z) 转为旋转矩阵"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)
    return R

def build_covariance(s, q):
    """
    s: (N, 3) scales 已经是计算好的物理尺度
    q: (N, 4) quaternions
    """
    # 1. 构造缩放矩阵 S
    S = torch.diag_embed(s)
    # 2. 归一化四元数并构造旋转矩阵 R
    q = torch.nn.functional.normalize(q, dim=-1)
    R = quaternion_to_rotation_matrix(q)
    
    # 3. 计算 Sigma = R * S * S_T * R_T
    # M = R * S
    M = R @ S
    covs = M @ M.transpose(-1, -2)
    
    return covs

if __name__=='__main__':
    torch.manual_seed(42)
    device = torch.device("cuda")
    # 1. surroundocc配置参数
    voxel_size = 0.5
    vol_min = torch.tensor([-50.0, -50.0, -5.0], device=device)
    vol_max = torch.tensor([50.0, 50.0, 3.0], device=device)
    vol_range = torch.tensor([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], device=device)
    grid_shape = (200, 200, 16)
    n_class = 18
    N = 12800 # 高斯点数量
    
    class_indices = list(range(1, 17))
    label_str = ['barrier', 'bicycle', 'bus', 'car', 'cons.veh',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'drive.surf', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation']
    num_classes = len(class_indices)
    empty_label = 17  
    

    
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
    # 尺度    
    # scales = nn.Parameter(torch.zeros((N, 3), device=device)) # sigmoid(0) = 0.5
    
    initial_s_target = 1.0 * voxel_size
    val = (initial_s_target - (voxel_size/3)) / ((10*voxel_size) - (voxel_size/3))
    initial_scales_val = np.log(val / (1 - val))
    scales = nn.Parameter(torch.full((N, 3), initial_scales_val, device=device))
    scales.data += torch.randn_like(scales.data) * 0.1
    
    scale_min = voxel_size / 3.0 
    scale_max = 10.0 * voxel_size
    
    # 旋转 随机初始化
    rotations = nn.Parameter(torch.randn((N, 4), device=device))
    
    L = nn.Parameter(torch.randn((N, 3, 3), device=device) * 0.2)
    
    # 透明度
    opacities = nn.Parameter(torch.rand((N,), device=device))
    # 特征：初始化为类别概率的 logits
    features = nn.Parameter(torch.randn((N, n_class), device=device))
    optimizer = optim.Adam([means3d, scales, rotations, L, opacities, features], lr=1e-2)
    
    manual_class_weight=[
    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ]
    class_weights = torch.tensor(manual_class_weight).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    '''
    criterion.ignore_index
    -100
    criterion.weight
    tensor([1.0155, 1.0690, 1.3001, 1.0725, 0.9464, 1.1009, 1.2696, 1.0626, 1.1890,
            1.0622, 1.0060, 0.8571, 1.0392, 0.9087, 0.8936, 0.8549, 0.8528, 0.5000],
        device='cuda:0')    
    '''
    save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/debug/0115/train/vis_results'
    os.makedirs(save_dir, exist_ok=True)

    print("Starting training with Triton-accelerated GaussSplatting3D...")
    for i_iter in tqdm(range(200)):
        optimizer.zero_grad()
        
        s = scale_min + (scale_max - scale_min) * torch.sigmoid(scales)
        covs = build_covariance(s,rotations)
        
        # covs = torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 1e-4
    
        density, pred_feats = GaussSplatting3DCuda.apply(
                    means3d, covs, torch.sigmoid(opacities), features, 
                    vol_range, voxel_size, grid_shape
                ) 


        logits = pred_feats.permute(3, 0, 1, 2).unsqueeze(0) # (200, 200, 16, 18) -> permute -> (1, 18, 200, 200, 16)
        target = target_tensor.unsqueeze(0)
        ce_loss = 10.0 * criterion(logits, # todo (b dim 200 200 16) 
                                   target) # todo (b 200 200 16)        
        '''
        unique_values, counts = torch.unique(target, return_counts=True)

        # 打印值及其对应的出现次数
        for val, count in zip(unique_values, counts):
            print(f"值: {val.item()}, 出现次数: {count.item()}")        
        '''
        lovasz_input = torch.softmax(logits.flatten(2), dim=1)
        lovasz_loss = 1.0 * lovasz_softmax(lovasz_input.transpose(1, 2).flatten(0, 1), # todo ((b h w d) 18)
                                           target.flatten(), # todo ((b h w d))
                                           ignore=17)        
        loss = ce_loss + lovasz_loss
        
        loss.backward()
        optimizer.step()
        

        
        if (i_iter+1) % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(pred_feats, dim=-1)
                fg_mask = (target_tensor != empty_label)
                fg_count = fg_mask.sum().item()
                outputs = preds[fg_mask]
                targets = target_tensor[fg_mask]
                
                
                total_seen = torch.zeros(num_classes+1).cuda()
                total_correct = torch.zeros(num_classes+1).cuda()
                total_positive = torch.zeros(num_classes+1).cuda()
                
                for i, c in enumerate(class_indices):
                    total_seen[i] += torch.sum(targets == c).item()
                    total_correct[i] += torch.sum((targets == c)
                                                & (outputs == c)).item()    
                    total_positive[i] += torch.sum(outputs == c).item() 
                
                total_seen[-1] += torch.sum(targets != empty_label).item()
                total_correct[-1] += torch.sum((targets != empty_label)
                                                & (outputs != empty_label)).item()
                total_positive[-1] += torch.sum(outputs != empty_label).item()

                
                
                total_seen = total_seen.cpu().numpy()
                total_correct = total_correct.cpu().numpy()
                total_positive = total_positive.cpu().numpy()
                
                ious = []
                header = ['classes']
                for i in range(len(label_str)):
                    header.append(label_str[i])
                header.extend(['miou', 'iou'])
                table_columns = [['results']]

                for i in range(num_classes): # todo 只计算语义类，不包括非空类
                    if total_seen[i] == 0: # todo iou & recall
                        cur_iou = np.nan
                    else:
                        cur_iou = total_correct[i] / (total_seen[i] + total_positive[i] - total_correct[i]) # todo iou = TP / (TP + FN + FP)

                    ious.append(cur_iou)
                    table_columns.append([f'{cur_iou:.4f}'])

                miou = np.nanmean(ious)
                iou = total_correct[-1] / (total_seen[-1] + total_positive[-1] - total_correct[-1])

                table_columns.append([f'{miou:.4f}'])
                table_columns.append([f"{iou:.4f}"])

                table_data = [header]
                table_rows = list(zip(*table_columns))
                table_data += table_rows
                table = AsciiTable(table_data)
                table.inner_footing_row_border = True
                print('\n' + table.table)
                print(f"Iter {i_iter:03d} | Loss: {loss.item():.4f} | CE_loss: {ce_loss.item():.4f} | Lovasz_loss: {lovasz_loss.item():.4f} |")
            
            final_preds = torch.argmax(pred_feats, dim=-1).cpu().numpy()
            output = dict(
                occ_pred=final_preds,
                occ_gt = target_tensor.cpu().numpy(),
            )
            with open(f'{save_dir}/iter_{i_iter:03d}.pkl', 'wb') as f:

                pickle.dump(output, f)    
            
    