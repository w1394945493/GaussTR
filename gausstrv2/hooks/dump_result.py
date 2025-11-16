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
        [  0,   0,   0, 255],       # others               black 黑色
        [255, 120,  50, 255],       # barrier              orange 橙色
        [255, 192, 203, 255],       # bicycle              pink 粉色
        [255, 255,   0, 255],       # bus                  yellow 黄色
        [  0, 150, 245, 255],       # car                  blue 蓝色
        [  0, 255, 255, 255],       # construction_vehicle cyan 青色
        [255, 127,   0, 255],       # motorcycle           dark orange 深橙色
        [255,   0,   0, 255],       # pedestrian           red 红色
        [255, 240, 150, 255],       # traffic_cone         light yellow 浅黄色
        [135,  60,   0, 255],       # trailer              brown 棕色
        [160,  32, 240, 255],       # truck                purple 紫色
        [255,   0, 255, 255],       # driveable_surface    dark pink 深粉色
        [175,   0,  75, 255],     # other_flat           dark red 深红色
        [139, 137, 137, 255],       # 无特定分类            gray 灰色
        [ 75,   0,  75, 255],       # sidewalk             dark purple 深紫色
        [150, 240,  80, 255],       # terrain              light green 浅绿色
        [230, 230, 250, 255],       # manmade              white 白色
        [  0, 175,   0, 255],       # vegetation           green 绿色
    ]
)

def seg_to_color(seg_tensor):
    """
    seg_tensor: torch.Tensor (H, W) or (N, H, W) - long, each pixel is class idx
    Returns: torch.Tensor (3, H, W) or (N, 3, H, W) uint8 RGB image
    """

    # 转为 numpy 数组 (N, H, W) 或 (H, W)
    seg_np = seg_tensor.cpu().numpy()

    # 处理单张图和多张图两种情况
    if seg_np.ndim == 2:
        h, w = seg_np.shape
        color_img = np.zeros((h, w, 4), dtype=np.uint8)
        for cls_id, color in enumerate(COLORS):
            mask = (seg_np == cls_id)
            color_img[mask] = color
        color_img = color_img[..., :3]  # 去掉alpha通道
        color_img = color_img.transpose(2, 0, 1)  # (3, H, W)
        return torch.from_numpy(color_img)

    elif seg_np.ndim == 3:
        n, h, w = seg_np.shape
        color_imgs = np.zeros((n, 4, h, w), dtype=np.uint8)
        for cls_id, color in enumerate(COLORS):
            mask = (seg_np == cls_id)
            # mask: (n, h, w)
            for i in range(n):
                color_imgs[i, :, :, :][mask[i]] = color
        color_imgs = color_imgs[:, :3, :, :]  # 去掉alpha通道
        return torch.from_numpy(color_imgs)

    else:
        raise ValueError("seg_tensor must be 2D or 3D tensor")

@HOOKS.register_module()
class DumpResultHookV2(Hook):

    def __init__(self, interval=1,save_dir='output/vis'):
        self.interval = interval

        mean = [123.675, 116.28, 103.53]
        std  = [58.395, 57.12, 57.375]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir

        self.dir_occ = os.path.join(save_dir, 'occ_pred')
        self.dir_img = os.path.join(save_dir, 'img_pred')
        self.dir_depth = os.path.join(save_dir, 'depth_pred')
        self.dir_seg = os.path.join(save_dir, 'seg_pred')

        for d in [self.dir_occ, self.dir_img, self.dir_depth, self.dir_seg]:
            os.makedirs(d, exist_ok=True)

        print(f"Dump results to: {self.save_dir}")



    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):

        outputs = outputs[0]

        occ_pred  = outputs['occ_pred']   # (b, X, Y, Z)
        img_pred  = outputs['img_pred']   # (b, n, 3, H, W)
        depth_pred = outputs['depth_pred'] # (b,n, H, W)
        seg_pred  = outputs['seg_pred']   # (b,n, C, H, W)

        b = occ_pred.shape[0]
        n = data_batch['data_samples'][0].cam2img.shape[0]


        if n % 2 == 0:
            cols = n // 2
            rows = 2
        else:
            cols = n
            rows = 1

        img_pred = img_pred*self.std.view(1,1,3,1,1).to(img_pred.device) + self.mean.view(1,1,3,1,1).to(img_pred.device)

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

        # -------- 2) 保存 img_pred --------
        for i in range(b):
            imgs = img_pred[i].float() / 255.0  # 0~1 float，方便save_image

            # 自动布局：偶数 → 3×2；奇数 → 1×n
            grid = torchvision.utils.make_grid(
                imgs,
                nrow=cols,  # 自动来自你前面计算的 cols
                padding=2
            )

            save_name = f"{data_batch['data_samples'][i].scene_token}_{data_batch['data_samples'][i].token}.png"
            save_path = os.path.join(self.dir_img, save_name)

            torchvision.utils.save_image(grid, save_path)

        # -------- 3) 保存 depth_pred --------
        depth_norm = depth_pred.clone()
        depth_norm -= depth_norm.min()
        depth_norm /= (depth_norm.max() + 1e-6)  # 归一化0~1

        for i in range(b):
            d = depth_norm[i].unsqueeze(1)  # (n,1,H,W)
            grid = torchvision.utils.make_grid(d, nrow=cols, padding=2)
            save_name = f"{data_batch['data_samples'][i].scene_token}_{data_batch['data_samples'][i].token}.png"
            save_path = os.path.join(self.dir_depth, save_name)
            torchvision.utils.save_image(grid, save_path)




