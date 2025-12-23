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
                 metainfo=None,
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

    def get_data_info(self, index):
        get_data_info = super(NuScenesSurroundOccDataset, self).get_data_info
        ori_input_dict  = get_data_info(index)
        ori_input_dict ['occ_path'] = os.path.join(
            self.data_root,
            f"gts/{ori_input_dict['scene_idx']}/{ori_input_dict['token']}")

        lidar2ego = np.array(ori_input_dict['lidar_points']['lidar2ego'])
        ego2lidar = np.linalg.inv(lidar2ego)

        ego2global = np.array(ori_input_dict['ego2global'])
        lidar2global = ego2global @ lidar2ego


        return ori_input_dict



if __name__=='__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from gaussianformer.datasets.transforms import *
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    input_size = (504, 896) # 指定输入尺寸
    test_pipeline = [
        dict(
            type='BEVLoadMultiViewImageFromFiles',to_float32=True,color_type='color',
            num_views=6),
        dict(type='LoadOccFromFile'),
        dict(type='ImageAug3D', final_dim=input_size, resize_lim=[0.56, 0.56]),
        dict(
            type='LoadFeatMaps',
            # data_root='data/nuscenes_metric3d', # todo
            data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini',
            key='depth',
            apply_aug=True),
        # todo -------------------------------#
        dict(
            type='Pack3DDetInputs',
            keys=['img', 'gt_semantic_seg'],
            meta_keys=[
                'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
                'num_views', 'img_path', 'depth', 'feats', 'mask_camera',
                'token','sample_idx','scene_token','scene_idx',
            ])
    ]

    dataset_type = 'NuScenesSurroundOccDataset' # 数据集名：nuscenes
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

    dataset_dict = dict(
        ann_file='nuscenes_mini_infos_val.pkl',
        pipeline=test_pipeline,
        **shared_dataset_cfg,
    )

    nuscenes_surroundocc = DATASETS.build(dataset_dict)
    scene_token = '325cef682f064c55a255f2625c533b75'
    token = '0bb62a68055249e381b039bf54b0ccf8'
    for i in range(len(nuscenes_surroundocc)):
        info = nuscenes_surroundocc.get_data_info(i)
        if info['token'] == token:
            data = nuscenes_surroundocc[i]
            print(i)

