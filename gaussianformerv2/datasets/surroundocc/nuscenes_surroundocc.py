import os
import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset
from pyquaternion import Quaternion


from mmdet3d.registry import DATASETS
import mmengine
from mmdet3d.registry import TRANSFORMS

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_img2global, get_lidar2global

@DATASETS.register_module()
class NuScenesSurroundOccDataset(Dataset):

    def __init__(self,
                 data_root=None,
                 imageset=None,
                 data_aug_conf=None,
                 pipeline=None,
                 vis_indices=None,
                 num_samples=0,
                 vis_scene_index=-1,
                 phase='train',
                 return_keys=[
                    'img',
                    
                    'projection_mat', # todo 'ego2img'或'lidar2img' gaussianformer中使用的'lidar2img'
                    
                    'image_wh',
                    'occ_label',
                    'occ_xyz',
                    'occ_cam_mask',
                    'ori_img',
                    
                    'cam_positions', # todo 
                    'focal_positions', # todo 

                    "cam2img", # todo (wys 12.30) 用于视图合成
                    "cam2ego",
                    "cam2lidar",
                    
                    
                    "depth", # todo (wys 12.30) 深度图
                    "img_gt",
                    "img_aug_mat",
                    
                    "scene_token",
                    "token",                      
                 ],
                 **kwargs):


        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')

        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(TRANSFORMS.build(t))

        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        self.return_keys = return_keys

        if vis_scene_index >= 0:
            frame = self.keyframes[vis_scene_index]
            num_frames = len(self.scene_infos[frame[0]])
            self.keyframes = [(frame[0], i) for i in range(num_frames)]
            print(f'Scene length: {len(self.keyframes)}')
        elif vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def __len__(self):
        return len(self.keyframes)

    def __getitem__(self, index: int) -> dict:
        scene_token, sample_idx = self.keyframes[index]
        info = deepcopy(self.scene_infos[scene_token][sample_idx])

        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()

        for t in self.pipeline:
            input_dict = t(input_dict)

        return_dict = {k: input_dict[k] for k in self.return_keys}
        return return_dict


    def get_data_info(self, info):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []
        
        cam2imgs = []
        cam2egos = []
        cam2lidars = []
        

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix # (3 3)
        lidar2ego = np.eye(4) # (4 4)单位矩阵
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego) # todo lidar2ego 和 ego2lidar

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

            img2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            img2lidar = np.linalg.inv(lidar2global) @ img2global
            
            # todo --------------------------------------------------#
            # todo (wys 12.30) 相机内参cam2img: 相对于原始尺寸(900x1600)的 
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            cam2imgs.append(viewpad)
            
            # todo --------------------------------------------------#
            # todo (wys 12.30) 相机外参cam2ego
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = Quaternion(info['data'][cam_type]['calib']['rotation']).rotation_matrix
            cam2ego[:3, 3] = np.asarray(info['data'][cam_type]['calib']['translation']).T 
            cam2egos.append(cam2ego)
            
            # todo --------------------------------------------------#
            # todo (wys 12.30) 相机外参cam2lidar      
            cam2lidar = ego2lidar @ cam2ego
            cam2lidars.append(cam2lidar)
            
            
            cam_position = img2lidar @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2lidar @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict =dict(
            scene_token = info['scene_token'],
            token = info['token'],
            occ_path=info.get("occ_path", ""),
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename']),
            
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts), # todo 
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions), # todo 
            focal_positions=np.asarray(focal_positions),
            
            cam2img = np.array(cam2imgs, dtype=np.float32), # todo 内参
            cam2ego = np.array(cam2egos, dtype=np.float32), # todo 外参
            cam2lidar = np.array(cam2lidars, dtype=np.float32), # todo 外参
            
            
            ) # todo 

        return input_dict

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"] # todo 原图大小
        fH, fW = self.data_aug_conf["final_dim"]  # todo 目标大小(网络期望输入)
        if not self.test_mode:
            # todo 训练时：确保能裁剪出fH x fW, 对尺度和裁剪位置做有约束的随机采样
            # todo 缩放
            # resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            # resize_dims = (int(W * resize), int(H * resize))
            resize = max(fH / H, fW / W) # todo 取较大的缩放比例：保证newH >= fH, newW >= fW
            resize_dims = (int(W * resize), int(H * resize)) # todo 缩放后的尺寸
                   
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # todo 随机翻转
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            # todo 旋转
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W) # todo 取较大的缩放比例：保证newH >= fH, newW >= fW
            resize_dims = (int(W * resize), int(H * resize)) # todo 缩放后的尺寸
            
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            ) # todo 上下方向偏下裁剪： 裁剪区域的下边界位置：偏向图像底部，多保留路面，车辆
            crop_w = int(max(0, newW - fW) / 2) # todo 左右居中裁剪：
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

if __name__=='__main__':
    from transform_3d import *
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
    anno_root = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/" # 标注根目录
    occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/" # occ标注根目录
    depth_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini'
    
    dataset_type = 'NuScenesSurroundOccDataset'

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )

    pipeline = [
        dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
        dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
        dict(type="ResizeCropFlipImage"),
        dict(type='LoadFeatMaps',data_root=depth_path,key='depth'), #
        dict(type="PhotoMetricDistortionMultiViewImage"), # todo
        dict(type="NormalizeMultiviewImage", **img_norm_cfg),
        dict(type="DefaultFormatBundle"),
        dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
    ]

    input_shape = (1600, 896)
    data_aug_conf = {
        # "resize_lim": (1.0, 1.0),
        "final_dim": input_shape[::-1], # todo 对图像进行缩放
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True, # todo 训练时做数据增强
    }


    dataset_dict = dict(
        type=dataset_type,
        data_root=data_root,
        # imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
        # imageset=anno_root + "nuscenes_mini_infos_train_sweeps_occ.pkl",
        imageset=anno_root + "nuscenes_mini_infos_val_sweeps_occ.pkl",
        data_aug_conf=data_aug_conf,
        pipeline=pipeline,
        phase='train',
    )

    def save_2x3_layout(data_list, filename, is_depth=False):
        
        # 创建 2行3列 画布
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i, ax in enumerate(axes.flatten()):
            if is_depth:
                # 深度图用 magma 色盘，看起来更直观
                im = ax.imshow(data_list[i], cmap='magma')
            else:
                ax.imshow(data_list[i])
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)  # 必须关闭窗口防止内存泄漏
        
    nuscenes_surroundocc = DATASETS.build(dataset_dict)
    N = 5 if len(nuscenes_surroundocc) > 5 else len(nuscenes_surroundocc)
    for i in range(N):
        data = nuscenes_surroundocc[i]
        # 1. 转换维度并搬运到 CPU
        # [6, 3, 896, 1600] -> [6, 896, 1600, 3]
        imgs_np = data['img_gt'].permute(0, 2, 3, 1).cpu().numpy()
        depths_np = data['depth'].cpu().numpy()

        # 2. 简单的去归一化，确保在 0-1 之间
        imgs_np = (imgs_np - imgs_np.min()) / (imgs_np.max() - imgs_np.min() + 1e-6)
        # 保存图片
        save_2x3_layout(imgs_np, f'vis_image_{i}.png')
        save_2x3_layout(depths_np, f'vis_depth_{i}.png', is_depth=True)