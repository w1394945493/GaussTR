import os

import pickle

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS

import torch
from PIL import Image
import torchvision
import numpy as np

@HOOKS.register_module()
class DumpResultHook(Hook):

    def __init__(self,
                interval=1,
                mean = [123.675, 116.28, 103.53],
                std  = [58.395, 57.12, 57.375],
                save_dir='output/vis',
                save_vis = False,
                save_depth=False,
                save_img=False,
                ):

        self.interval = interval


        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        self.save_vis = save_vis

        self.save_depth = save_depth
        self.save_img = save_img


        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir

        if save_depth:
            self.dir_depth = os.path.join(save_dir, 'depth_pred')
            os.makedirs(self.dir_depth,exist_ok=True)
        if save_img:
            self.dir_img = os.path.join(save_dir, 'img_pred')
            os.makedirs(self.dir_img,exist_ok=True)

        print(f"Dump results to: {self.save_dir}")



    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):
        if not self.save_vis:
            return

        outputs = outputs[0]

        b = len(data_batch['data_samples'])
        n = data_batch['data_samples'][0].cam2img.shape[0]
        if n % 2 == 0:
            cols = n // 2
            rows = 2
        else:
            cols = n
            rows = 1

        # -------- 2) 保存 img_pred --------
        if self.save_img and outputs['img_pred'] is not None:

            img_pred  = outputs['img_pred']   # (b, n, 3, H, W)
            # img_pred = img_pred*self.std.view(1,1,3,1,1).to(img_pred.device) + self.mean.view(1,1,3,1,1).to(img_pred.device)

            for i in range(b):
                data_sample = data_batch['data_samples'][i]
                # imgs = img_pred[i].float() / 255.0  # 0~1 float，方便save_image
                imgs = img_pred[i].clamp(min=0.,max=1.)

                # 布局：偶数 → 3×2；奇数 → 1×n
                grid = torchvision.utils.make_grid(
                    imgs,
                    nrow=cols,
                    padding=2
                )
                save_name = f"{data_sample.scene_token}_{data_sample.token}.png"
                save_path = os.path.join(self.dir_img, save_name)

                torchvision.utils.save_image(grid, save_path)

                # ------------- 保存 img_gt ----------
                # data_sample = data_batch['data_samples'][i]
                imgs_gt = data_sample.img.clamp(0, 255) / 255.0
                grid = torchvision.utils.make_grid(
                    imgs_gt,
                    nrow=cols,
                    padding=2
                )
                save_name = f"{data_sample.scene_token}_{data_sample.token}_gt.png"
                save_path = os.path.join(self.dir_img, save_name)
                torchvision.utils.save_image(grid, save_path)

        # -------- 3) 保存 depth_pred --------
        if self.save_depth and outputs['depth_pred'] is not None:

            depth_pred = outputs['depth_pred'] # (b,n, H, W)

            depth_norm = depth_pred.clone()
            depth_norm -= depth_norm.min()
            depth_norm /= (depth_norm.max() + 1e-6)  # 归一化0~1

            for i in range(b):

                data_sample = data_batch['data_samples'][i] # todo 每个batch的数据都放在一个data_samples下

                d = depth_pred[i].unsqueeze(1)  # (n,1,H,W)
                max_val = float(d.max())
                d = d / (max_val + 1e-6)


                grid = torchvision.utils.make_grid(d, nrow=cols, padding=2)
                save_name = f"{data_sample.scene_token}_{data_sample.token}.png"
                save_path = os.path.join(self.dir_depth, save_name)

                torchvision.utils.save_image(grid, save_path)

                depth_gt = data_sample.depth.unsqueeze(1)
                max_val = float(depth_gt.max())
                depth_gt /= (max_val + 1e-6)

                grid = torchvision.utils.make_grid(depth_gt, nrow=cols, padding=2)
                save_name = f"{data_sample.scene_token}_{data_sample.token}_gt.png"
                save_path = os.path.join(self.dir_depth, save_name)
                torchvision.utils.save_image(grid, save_path)
        
        return








