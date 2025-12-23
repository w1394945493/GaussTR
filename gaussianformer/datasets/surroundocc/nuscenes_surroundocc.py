import os
import numpy as np

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS



@DATASETS.register_module()
class NuScenesSurroundOccDataset(NuScenesDataset):

    METAINFO = {
        'classes':
        ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free'),
    }

    def __init__(self,
                 data_aug_conf=None,
                 metainfo=None,
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
        if not metainfo:
            metainfo = self.METAINFO
        elif 'classes' not in metainfo:
            metainfo['classes'] = self.METAINFO['classes']
        metainfo['label2cat'] = {
            i: cat_name
            for i, cat_name in enumerate(metainfo['classes'])
        }
        super().__init__(metainfo=metainfo, **kwargs)
        self.data_aug_conf = data_aug_conf
        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.return_keys = return_keys

    def __getitem__(self, index: int) -> dict:
        input_dict = self.get_data_info(index)
        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        input_dict = self.pipeline(input_dict)
        return_dict = {k: input_dict[k] for k in self.return_keys}
        return return_dict


    def get_data_info(self, index):
        get_data_info = super(NuScenesSurroundOccDataset, self).get_data_info
        info  = get_data_info(index)
        info ['occ_path'] = os.path.join(
            self.data_root,
            f"gts/{info['scene_idx']}/{info['token']}")

        f = 0.0055

        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []

        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        ego2lidar = np.linalg.inv(lidar2ego)

        ego2global = np.array(info['ego2global'])
        lidar2global = ego2global @ lidar2ego

        for cam_type in self.sensor_types:
            image_paths.append(info['images'][cam_type]['img_path'])

            cam2img = np.eye(4)
            cam2img[:3,:3] = np.array(info['images'][cam_type]['cam2img'])
            img2cam = np.linalg.inv(cam2img)

            cam2ego = np.array(info['images']['CAM_FRONT']['cam2ego'])

            img2global = ego2global @ cam2ego @ img2cam
            lidar2img = np.linalg.inv(img2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            img2lidar = np.linalg.inv(lidar2global) @ img2global
            viewpad = cam2img
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
            pts_filename=os.path.join(self.data_root, 'samples/LIDAR_TOP',info['lidar_points']['lidar_path']),
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

