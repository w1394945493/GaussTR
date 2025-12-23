import os
import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset
from pyquaternion import Quaternion


from mmdet3d.registry import DATASETS
import mmengine
from mmdet3d.registry import TRANSFORMS

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
                    'projection_mat',
                    'image_wh',
                    'occ_label',
                    'occ_xyz',
                    'occ_cam_mask',
                    'ori_img',
                    'cam_positions',
                    'focal_positions',
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
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
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
            lidar2img=np.asarray(lidar2img_rts),
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions))

        return input_dict

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
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
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate










if __name__=='__main__':
    from transform_3d import *
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/"

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )

    test_pipeline = [
        dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
        dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
        dict(type="ResizeCropFlipImage"),
        dict(type="NormalizeMultiviewImage", **img_norm_cfg),
        dict(type="DefaultFormatBundle"),
        dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
    ]

    dataset_type = 'NuScenesSurroundOccDataset'
    data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
    data_prefix = dict(
        CAM_FRONT='samples/CAM_FRONT',
        CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
        CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
        CAM_BACK='samples/CAM_BACK',
        CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
        CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
        LIDAR_TOP = 'samples/LIDAR_TOP', # todo
        )
    input_modality = dict(use_camera=True, use_lidar=False)

    shared_dataset_cfg = dict(
        type=dataset_type,
        data_root=data_root,
        modality=input_modality,
        data_prefix=data_prefix,
        filter_empty_gt=False)

    input_shape = (1600, 864)
    data_aug_conf = {
        "resize_lim": (1.0, 1.0),
        "final_dim": input_shape[::-1],
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }


    dataset_dict = dict(
        ann_file='nuscenes_mini_infos_val.pkl',
        data_aug_conf=data_aug_conf,
        pipeline=test_pipeline,
        **shared_dataset_cfg,
    )

    nuscenes_surroundocc = DATASETS.build(dataset_dict)
    scene_token = '325cef682f064c55a255f2625c533b75'
    token = 'b5989651183643369174912bc5641d3b'
    for i in range(len(nuscenes_surroundocc)):
        info = nuscenes_surroundocc.get_data_info(i)
        if info['token'] == token:
            data = nuscenes_surroundocc[i]
            print(i)

