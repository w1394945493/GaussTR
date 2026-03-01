import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import pickle

from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from pyquaternion import Quaternion
from tqdm import tqdm

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"



COLORS = np.array(
    [
        [  0,   0,   0, 255],       # others
        [255, 120,  50, 255],       # barrier              orange
        [255, 192, 203, 255],       # bicycle              pink
        [255, 255,   0, 255],       # bus                  yellow
        [  0, 150, 245, 255],       # car                  blue
        [  0, 255, 255, 255],       # construction_vehicle cyan
        [255, 127,   0, 255],       # motorcycle           dark orange
        [255,   0,   0, 255],       # pedestrian           red
        [255, 240, 150, 255],       # traffic_cone         light yellow
        [135,  60,   0, 255],       # trailer              brown
        [160,  32, 240, 255],       # truck                purple
        [255,   0, 255, 255],       # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [ 75,   0,  75, 255],       # sidewalk             dard purple
        [150, 240,  80, 255],       # terrain              light green
        [230, 230, 250, 255],       # manmade              white
        [  0, 175,   0, 255],       # vegetation           green
        # [  0, 255, 127, 255],       # ego car              dark cyan
        # [255,  99,  71, 255],       # ego car
        # [  0, 191, 255, 255]        # ego car
    ]
).astype(np.float32) / 255.



@HOOKS.register_module()
class DumpResultHook(Hook):
    def __init__(self,
                 save_dir='output/vis',    
                 save_occ=True,           
                 ):
        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir    

        self.occ_path = os.path.join(save_dir, 'occ_pred')
        os.makedirs(self.occ_path,exist_ok=True)            
        self.img_path = os.path.join(save_dir, 'img_pred')
        os.makedirs(self.img_path,exist_ok=True)   
        self.depth_path = os.path.join(save_dir, 'depth_pred')
        os.makedirs(self.depth_path,exist_ok=True)   
        
        
        
        print(f"Dump results to: {self.save_dir}")        
        return
    
    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):
        
        bs, n = data_batch['cam2img'].shape[:2]   
        cols = 3 if n >= 3 else n   
        
        if 'occ_pred' in outputs[0]:
             for i in range(bs):
                # todo --------------------------------------#
                # todo 保存占用图
                occ_pred = outputs[0]['occ_pred'][i].cpu().numpy()
                occ_gt = outputs[0]['occ_gt'][i].cpu().numpy()
                output = dict(
                    occ_pred = occ_pred,
                    occ_gt = occ_gt, )                    
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}.pkl"
                save_path = os.path.join(self.occ_path, save_name)
                with open(save_path, 'wb') as f:
                    pickle.dump(output, f)  
                
                if 'gaussian' in outputs[0]:

                    means = outputs[0]['gaussian'].means[i].cpu().numpy()
                    semantics = outputs[0]['gaussian'].semantics[i]
                    probs = semantics.argmax(-1).cpu().numpy().reshape(-1, 1)
                    scales = outputs[0]['gaussian'].scales[i].cpu().numpy()
                    rotations = outputs[0]['gaussian'].rotations[i].cpu().numpy()

                    res_data =  np.concatenate([means, probs, scales, rotations], axis=1)
                    
                    save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}.npy"
                    save_path = os.path.join(self.occ_path, save_name)
                    np.save(save_path,res_data)
                 
        if 'img_pred' in outputs[0]:     
            img_pred  = outputs[0]['img_pred']
            img_gt  = data_batch['ori_img'] / 255.
            for i in range(bs):
                
                imgs = img_pred[i].clamp(min=0.,max=1.)
                grid = torchvision.utils.make_grid(
                    imgs,
                    nrow=cols,
                    padding=2
                )                
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}_img_pred.png"
                save_path = os.path.join(self.img_path, save_name)
                torchvision.utils.save_image(grid, save_path)     
            
                imgs = img_gt[i].clamp(min=0.,max=1.)
                grid = torchvision.utils.make_grid(
                    imgs,
                    nrow=cols,
                    padding=2
                )                
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}_img_gt.png"
                save_path = os.path.join(self.img_path, save_name)
                torchvision.utils.save_image(grid, save_path) 
                        
        if 'depth_pred' in outputs[0]:                       
            
            for i in range(bs):
                depth_pred = outputs[0]['depth_pred'][i]
                f_h,f_w = depth_pred.shape[-2:]
                
                depth_pred = depth_pred.unsqueeze(1)
                max_val = float(depth_pred.max())
                depth_pred /= (max_val + 1e-6)                  
                
                grid = torchvision.utils.make_grid(depth_pred, nrow=cols, padding=2)
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}_depth_pred.png"
                save_path = os.path.join(self.depth_path, save_name)
                torchvision.utils.save_image(grid, save_path)                                
        return

  