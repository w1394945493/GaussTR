import os

import pickle

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS

import torch
from PIL import Image
import torchvision
import numpy as np

COLORS = np.array(
    [
        [  0,   0,   0, 255],       # 0-others               black 黑色
        [255, 120,  50, 255],       # 1-barrier              orange 橙色
        [255, 192, 203, 255],       # 2-bicycle              pink 粉色
        [255, 255,   0, 255],       # 3-bus                  yellow 黄色
        [  0, 150, 245, 255],       # 4-car                  blue 蓝色
        [  0, 255, 255, 255],       # 5-construction_vehicle cyan 青色
        [255, 127,   0, 255],       # 6-motorcycle           dark orange 深橙色
        [255,   0,   0, 255],       # 7-pedestrian           red 红色
        [255, 240, 150, 255],       # 8-traffic_cone         light yellow 浅黄色
        [135,  60,   0, 255],       # 9-trailer              brown 棕色
        [160,  32, 240, 255],       # 10-truck                purple 紫色
        [255,   0, 255, 255],       # 11-driveable_surface    dark pink 深粉色
        [10,   10,  10, 255],       # 12-other-flat            gray 黑色
        [ 75,   0,  75, 255],       # 13-sidewalk             dark purple 深紫色
        [150, 240,  80, 255],       # 14-terrain              light green 浅绿色
        [139, 137, 137, 255],       # 15-manmade              white 灰色
        [  0, 175,   0, 255],       # 16-vegetation           green 深绿色
        [180, 200, 255, 255],       # 17-sky                  green 蓝白色
    ]
)



def seg_to_color(seg, color_map):
    """
    seg: (N, H, W)  int labels 0~num_classes-1
    color_map: (num_classes, 4) RGBA
    return: (N, 3, H, W)  uint8 RGB image
    """

    assert seg.dim() == 3, "seg must be (N,H,W)"
    N, H, W = seg.shape

    # 转到 numpy
    seg_np = seg.cpu().numpy().astype(np.int32)  # (N,H,W)

    # 创建输出 (N,H,W,3)
    out = np.zeros((N, H, W, 3), dtype=np.uint8)

    # 按标签查颜色
    for cls_id in range(color_map.shape[0]):
        mask = (seg_np == cls_id)  # (N,H,W)
        out[mask] = color_map[cls_id, :3]  # RGB 3 通道

    # 转为 torch: (N,3,H,W)
    out = torch.from_numpy(out).permute(0, 3, 1, 2)  # (N,3,H,W)
    return out


@HOOKS.register_module()
class DumpResultHookV2(Hook):

    def __init__(self,
                interval=1,
                save_dir='output/vis',
                save_occ=True,
                save_depth=False,
                save_sem_seg=False,
                save_img=False,
                ):
        self.interval = interval

        mean = [123.675, 116.28, 103.53]
        std  = [58.395, 57.12, 57.375]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        self.save_occ = save_occ
        self.save_depth = save_depth
        self.save_sem_seg = save_sem_seg
        self.save_img = save_img

        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir

        if save_occ:
            self.dir_occ = os.path.join(save_dir, 'occ_pred')
            os.makedirs(self.dir_occ,exist_ok=True)
        if save_depth:
            self.dir_depth = os.path.join(save_dir, 'depth_pred')
            os.makedirs(self.dir_depth,exist_ok=True)
        if save_sem_seg:
            self.dir_seg = os.path.join(save_dir, 'seg_pred')
            os.makedirs(self.dir_seg,exist_ok=True)
        if save_img:
            self.dir_img = os.path.join(save_dir, 'img_pred')
            os.makedirs(self.dir_img,exist_ok=True)

        print(f"Dump results to: {self.save_dir}")



    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):

        outputs = outputs[0]

        b = len(data_batch['data_samples'])
        n = data_batch['data_samples'][0].cam2img.shape[0]
        if n % 2 == 0:
            cols = n // 2
            rows = 2
        else:
            cols = n
            rows = 1

        img_pred = img_pred*self.std.view(1,1,3,1,1).to(img_pred.device) + self.mean.view(1,1,3,1,1).to(img_pred.device)

        if self.save_occ:
            occ_pred  = outputs['occ_pred']   # (b, X, Y, Z)
            # ---------------------- 1) 保存占用预测 ----------------------
            for i in range(b):
                data_sample = data_batch['data_samples'][i]
                output = dict(
                    occ_pred=occ_pred[i].cpu().numpy(),  # (200,200,16)
                    occ_gt=(data_sample.gt_pts_seg.semantic_seg.squeeze().cpu().
                            numpy()), # (200,200,16)
                    mask_camera=data_sample.mask_camera,
                    img_path=data_sample.img_path
                    )

                save_path = os.path.join(self.dir_occ,
                                        f"{data_sample.scene_token}_{data_sample.token}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(output, f)


        # -------- 3) 保存 depth_pred --------
        if self.save_depth:

            depth_pred = outputs['depth_pred'] # (b,n, H, W)

            depth_norm = depth_pred.clone()
            depth_norm -= depth_norm.min()
            depth_norm /= (depth_norm.max() + 1e-6)  # 归一化0~1

            for i in range(b):

                data_sample = data_batch['data_samples'][i] # todo 每个batch的数据都放在一个data_samples下

                d = depth_norm[i].unsqueeze(1)  # (n,1,H,W)
                grid = torchvision.utils.make_grid(d, nrow=cols, padding=2)
                save_name = f"{data_sample.scene_token}_{data_sample.token}.png"
                save_path = os.path.join(self.dir_depth, save_name)

                torchvision.utils.save_image(grid, save_path)

                depth_gt = data_sample.depth.unsqueeze(1)
                depth_gt -= depth_gt.min()
                depth_gt /= (depth_gt.max() + 1e-6)
                grid = torchvision.utils.make_grid(depth_gt, nrow=cols, padding=2)
                save_name = f"{data_sample.scene_token}_{data_sample.token}_gt.png"
                save_path = os.path.join(self.dir_depth, save_name)
                torchvision.utils.save_image(grid, save_path)

        # -------- 3) 保存 sem_seg --------
        if self.save_sem_seg:

            seg_pred  = outputs['seg_pred']   # (b,n, C, H, W)

            for i in range(b):
                data_sample = data_batch['data_samples'][i] # todo 每个batch的数据都放在一个data_samples下

                sem_seg_pred = seg_pred[i].argmax(dim=1)
                seg_color = seg_to_color(sem_seg_pred, COLORS).clamp(0, 255) / 255.0
                grid = torchvision.utils.make_grid(seg_color, nrow=cols, padding=2)
                save_name = f"{data_sample.scene_token}_{data_sample.token}.png"
                save_path = os.path.join(self.dir_seg, save_name)
                torchvision.utils.save_image(grid, save_path)

                sem_seg_gt = data_sample.sem_seg
                seg_color = seg_to_color(sem_seg_gt, COLORS).clamp(0, 255) / 255.0
                grid = torchvision.utils.make_grid(seg_color, nrow=cols, padding=2)
                save_name = f"{data_sample.scene_token}_{data_sample.token}_gt.png"
                save_path = os.path.join(self.dir_seg, save_name)
                torchvision.utils.save_image(grid, save_path)


        # -------- 2) 保存 img_pred --------
        if self.save_img:

            img_pred  = outputs['img_pred']   # (b, n, 3, H, W)

            for i in range(b):
                data_sample = data_batch['data_samples'][i]
                imgs = img_pred[i].float() / 255.0  # 0~1 float，方便save_image

                # 自动布局：偶数 → 3×2；奇数 → 1×n
                grid = torchvision.utils.make_grid(
                    imgs,
                    nrow=cols,  # 自动来自你前面计算的 cols
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
                    nrow=cols,  # 自动来自你前面计算的 cols
                    padding=2
                )
                save_name = f"{data_sample.scene_token}_{data_sample.token}_gt.png"
                save_path = os.path.join(self.dir_img, save_name)
                torchvision.utils.save_image(grid, save_path)

