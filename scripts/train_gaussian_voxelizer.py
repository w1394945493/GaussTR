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
from terminaltables import AsciiTable

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_gaussian_voxelizer import splat_into_3d, GaussSplatting3D
from lovzsz_softmax import lovasz_softmax

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
    N = 3200 # 高斯点数量    
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
    optimizer = optim.Adam([means3d, L, opacities, features], lr=1e-2)
    
    manual_class_weight=[
    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ]
    class_weights = torch.tensor(manual_class_weight).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/debug/0112/vis_results'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting training with Triton-accelerated GaussSplatting3D...")
    for i_iter in tqdm(range(200)):
        covs = torch.matmul(L, L.transpose(-1, -2)) + torch.eye(3, device=device) * 1e-4
        density, pred_feats = GaussSplatting3D.apply(
                    means3d, covs, torch.sigmoid(opacities), features, 
                    vol_range, voxel_size, grid_shape
                )   
        logits = pred_feats.permute(3, 0, 1, 2).unsqueeze(0) # (200, 200, 16, 18) -> permute -> (1, 18, 200, 200, 16)
        target = target_tensor.unsqueeze(0)
        ce_loss = 10.0 * criterion(logits, target)        
        
        lovasz_input = torch.softmax(logits.flatten(2), dim=1)
        lovasz_loss = 1.0 * lovasz_softmax(lovasz_input.transpose(1, 2).flatten(0, 1), target.flatten(), ignore=17)        
        loss = ce_loss + lovasz_loss
        
        loss.backward()
        optimizer.step()
        
        class_indices = list(range(1, 17))
        label_str = ['barrier', 'bicycle', 'bus', 'car', 'cons.veh',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'drive.surf', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation']
        num_classes = len(class_indices)
        empty_label = 17
        
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
            
    