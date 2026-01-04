import os
import torchvision
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@HOOKS.register_module()
class DumpResultHook(Hook):
    def __init__(self,
                 save_dir='output/vis',
                 save_img=True,
                 save_depth=True,                 
                 ):
        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir    
            
        self.save_img = save_img
        self.save_depth = save_depth
        
        if save_img:
            self.dir_img = os.path.join(save_dir, 'img_pred')
            os.makedirs(self.dir_img,exist_ok=True)

        if save_depth:
            self.dir_depth = os.path.join(save_dir, 'depth_pred')
            os.makedirs(self.dir_depth,exist_ok=True)
            
        print(f"Dump results to: {self.save_dir}")        
        return
    
    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):
        
        bs, n = data_batch['cam2img'].shape[:2]
        if n % 2 == 0:
            cols = n // 2
        else:
            cols = n       
        if self.save_img and outputs[0]['img_pred'] is not None:        
            img_pred  = outputs[0]['img_pred']
            for i in range(bs):
                imgs = img_pred[i].clamp(min=0.,max=1.)
                # imgs = imgs[:, [2, 1, 0], :, :]
                grid = torchvision.utils.make_grid(
                    imgs,
                    nrow=cols,
                    padding=2
                )                
            save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}.png"
            save_path = os.path.join(self.dir_img, save_name)
            torchvision.utils.save_image(grid, save_path)     
            
            img_gt  = data_batch['img_gt'] / 255.
            for i in range(bs):
                imgs = img_gt[i].clamp(min=0.,max=1.)
                # imgs = imgs[:, [2, 1, 0], :, :]
                grid = torchvision.utils.make_grid(
                    imgs,
                    nrow=cols,
                    padding=2
                )                
            save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}_gt.png"
            save_path = os.path.join(self.dir_img, save_name)
            torchvision.utils.save_image(grid, save_path) 
        
        if self.save_depth and outputs[0]['depth_pred'] is not None:                       
            
            for i in range(bs):
                depth_pred = outputs[0]['depth_pred'][i]
                depth_pred = depth_pred.unsqueeze(1)
                max_val = float(depth_pred.max())
                depth_pred /= (max_val + 1e-6)                  
                
                grid = torchvision.utils.make_grid(depth_pred, nrow=cols, padding=2)
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}.png"
                save_path = os.path.join(self.dir_depth, save_name)
                torchvision.utils.save_image(grid, save_path)                
                
                
                depth_gt = data_batch['depth'][i]
                depth_gt = depth_gt.unsqueeze(1)
                max_val = float(depth_gt.max())
                depth_gt /= (max_val + 1e-6)                
                
                grid = torchvision.utils.make_grid(depth_gt, nrow=cols, padding=2)
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}_gt.png"
                save_path = os.path.join(self.dir_depth, save_name)
                torchvision.utils.save_image(grid, save_path)
        return


