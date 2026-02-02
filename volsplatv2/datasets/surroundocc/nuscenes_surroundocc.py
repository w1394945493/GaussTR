import os
import numpy as np
from copy import deepcopy
import random

from torch.utils.data import Dataset
from pyquaternion import Quaternion

from mmdet3d.registry import DATASETS
import mmengine
from mmdet3d.registry import TRANSFORMS

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from .utils import get_img2global, get_lidar2global
# from utils import get_img2global, get_lidar2global

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
    
                    'occ_label',
                    'occ_xyz',
                    'occ_cam_mask',
                    
                    'ori_img',
                    'img',
                    "depth", # todo (wys 12.30) 深度图
                    "cam2img", # todo (wys 12.30) 用于视图合成
                    "cam2ego",
                    "cam2lidar",
                    "img_aug_mat",
                    
                    "projection_mat", # todo (wys 02.02)
                    "featmap_wh",     # todo (wys 02.02)
                    
                    "scene_token",
                    "token",                      
                 ],
                 **kwargs):


        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
        
        # todo -------------------------------#
        # todo 均匀抽取1/10 训练/评估
        # self.keyframes = self.keyframes[::10]
        
        
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
        
        scene_token, sample_idx = self.keyframes[index] # todo 关键帧数据
        info = deepcopy(self.scene_infos[scene_token][sample_idx]) # todo 以场景为单位的数据

        input_dict = self.get_data_info(info)
              
        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()

        for t in self.pipeline:
            input_dict = t(input_dict)

        return_dict = {k: input_dict[k] for k in self.return_keys}
        return return_dict

        
    
    
    def get_data_info(self, info):

        image_paths = []
        cam2imgs = []
        cam2egos = []
        cam2lidars = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix # (3 3)
        lidar2ego = np.eye(4) # (4 4)单位矩阵
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego) # todo lidar2ego 和 ego2lidar

        for cam_type in self.sensor_types:
            
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))
            
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = np.array(intrinsic)
                       
            cam2imgs.append(viewpad)
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = Quaternion(info['data'][cam_type]['calib']['rotation']).rotation_matrix
            cam2ego[:3, 3] = np.asarray(info['data'][cam_type]['calib']['translation']).T 
            cam2egos.append(cam2ego)
                 
            cam2lidar = ego2lidar @ cam2ego
            cam2lidars.append(cam2lidar)            
            
        input_dict =dict(
            scene_token = info['scene_token'],
            token = info['token'],
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename']),            
            cam2img = np.array(cam2imgs, dtype=np.float32), # todo 内参
            cam2ego = np.array(cam2egos, dtype=np.float32), # todo 外参
            cam2lidar = np.array(cam2lidars, dtype=np.float32), # todo 外参
                               
            ) # todo 

        return input_dict

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"] # todo 原图大小
        fH, fW = self.data_aug_conf["final_dim"]  # todo 网络输入尺寸
        
        featmap_h,feat_map_w = self.data_aug_conf["featmap_dim"]
        featmap_dims = (feat_map_w, featmap_h)
        
        resize = [fW/W, fH/H]
        resize_dims = (fW, fH) # todo fW, fH
        newW, newH = resize_dims        
        
        crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        
        flip = False
        rotate = 0
        
        if not self.test_mode:
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
           
        
        return resize,  resize_dims, featmap_dims, crop,  flip,  rotate


if __name__=='__main__':
    from transform_3d import *
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    # data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录 # todo v1.0-mini
    # anno_root = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/" # 标注根目录  # todo v1.0-trainval
    # occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/" # occ标注根目录(所有数据集标注根目录)
    # depth_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini'

    data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval/' # todo v1.0-trainval
    anno_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/nuscenes/' # todo v1.0-mini
    
    occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/samples/"
    depth_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/samples_dptm_small"
    
    dataset_type = 'NuScenesSurroundOccDataset'
    ann_file_name = "nuscenes_infos_train_sweeps_occ.pkl"

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )

    pipeline = [
        dict(type="BEVLoadMultiViewImageFromFiles", to_float32=True), # todo 加载图像
        dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
        dict(type="ResizeCropFlipImage"), # todo 做resize,crop,flip,rotate等预处理
        dict(type='LoadFeatMaps',data_root=depth_path, key='depth', apply_aug=True), 
        dict(type="NormalizeMultiviewImage", **img_norm_cfg), # todo BGR -> RGB mean和std
        dict(type="DefaultFormatBundle"), # todo 将图像由(h w c) -> (c h w)
        dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
    ]

    final_dim = (896,1600)
    output_dim = (112,200)
    data_aug_conf = {
        "final_dim": final_dim, # todo 网络输入
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900, 
        "W": 1600,
        "rand_flip": True, # todo 训练时做数据增强
        "output_dim": output_dim, # todo 渲染图像尺寸
    }


    dataset_dict = dict(
        type=dataset_type,
        data_root=data_root,
        imageset=os.path.join(anno_root, ann_file_name),
        # imageset=anno_root + "nuscenes_mini_infos_train_sweeps_occ.pkl",
        # imageset=anno_root + "nuscenes_mini_infos_val_sweeps_occ.pkl",
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
    # N = 5 if len(nuscenes_surroundocc) > 5 else len(nuscenes_surroundocc)
    N = len(nuscenes_surroundocc)
    for i in tqdm(range(N)):
        data = nuscenes_surroundocc[i]
        