import os
import random
from collections import deque
from typing import Deque, Iterable

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS



@DATASETS.register_module()
class NuScenesOccDataset(NuScenesDataset):

    METAINFO = {
        'classes':
        ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free'),
    }

    def __init__(self,
                 metainfo=None,
                 load_adj_frame=False,
                 interval=1,
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

        self.load_adj_frame = load_adj_frame
        self.interval = interval

    def get_data_info(self, index): # todo 重写 BaseDataset中的 get_data_info 方法
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        get_data_info = super(NuScenesOccDataset, self).get_data_info # todo BaseDataset中的 get_data_info 方法
        input_dict = get_data_info(index)

        def get_curr_token(seq):
            curr_index = min(len(seq) - 1, 1)
            return seq[curr_index]['scene_token']

        def fetch_prev(seq: Deque, index, interval=1):
            if index == 0:
                return None
            prev = get_data_info(index - interval)
            if prev['scene_token'] != get_curr_token(seq):
                return None
            seq.appendleft(prev)
            return prev

        def fetch_next(seq: Deque, index, interval=1):
            if index >= len(self) - interval:
                return None
            next = get_data_info(index + interval)
            if next['scene_token'] != get_curr_token(seq):
                return None
            seq.append(next)
            return next

        if self.load_adj_frame: # todo false
            input_seq = deque([input_dict], maxlen=3)
            interval = random.randint(*self.interval) if isinstance(
                self.interval, Iterable) else self.interval
            if not fetch_prev(input_seq, index, interval):
                fetch_next(input_seq, index, interval)
                index += interval
            if not fetch_next(input_seq, index, interval):
                fetch_prev(input_seq, index - interval)

            assert (len(input_seq) == 3 and input_seq[0]['scene_token'] ==
                    input_seq[1]['scene_token'] == input_seq[2]['scene_token'])
            input_dict = self.concat_adj_frames(*input_seq)
        # todo ---------------------------#
        # todo 增加了occ_gt的路径：在根目录下
        input_dict['occ_path'] = os.path.join(
            self.data_root,
            f"gts/{input_dict['scene_idx']}/{input_dict['token']}")
        return input_dict

    def concat_adj_frames(self, prev, curr, next=None):
        curr['images'] = dict(
            **curr['images'], **{
                f'PREV_{k}': v
                for k, v in prev['images'].items()
            })
        curr['ego2global'] = [curr['ego2global'], prev['ego2global']]

        if next is not None:
            curr['images'] = dict(
                **curr['images'], **{
                    f'NEXT_{k}': v
                    for k, v in next['images'].items()
                })
            curr['ego2global'].append(next['ego2global'])
        return curr

if __name__=='__main__':
    input_size = (504, 896) # 指定输入尺寸
    test_pipeline = [
        dict(
            type='BEVLoadMultiViewImageFromFiles',
            to_float32=True,
            color_type='color',
            num_views=6),
        dict(type='LoadOccFromFile'),
        dict(type='ImageAug3D', final_dim=input_size, resize_lim=[0.56, 0.56]),
        dict(
            type='LoadFeatMaps',
            # data_root='data/nuscenes_metric3d', # todo
            data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d',
            key='depth',
            apply_aug=True),
        dict(
            type='Pack3DDetInputs',
            keys=['img', 'gt_semantic_seg'],
            meta_keys=[
                'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
                'num_views', 'img_path', 'depth', 'feats', 'mask_camera',
                'token','sample_idx','scene_token','scene_idx', # ? 'token'
            ])
    ]

    dataset_type = 'NuScenesOccDataset' # 数据集名：nuscenes
    data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
    data_prefix = dict(
        CAM_FRONT='samples/CAM_FRONT',
        CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
        CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
        CAM_BACK='samples/CAM_BACK',
        CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
        CAM_BACK_LEFT='samples/CAM_BACK_LEFT')
    input_modality = dict(use_camera=True, use_lidar=False)

    shared_dataset_cfg = dict(
        type=dataset_type,
        data_root=data_root,
        modality=input_modality,
        data_prefix=data_prefix,
        filter_empty_gt=False)

    dataset_dict = dict(
        ann_file='nuscenes_mini_infos_train.pkl',
        pipeline=test_pipeline,
        **shared_dataset_cfg,
    )

    from transforms import *
    from mmengine.registry import init_default_scope

    init_default_scope('mmdet3d')
    # nuscenes_occ = NuScenesDataset(**dataset_dict)
    nuscenes_occ = DATASETS.build(dataset_dict)

    for i in range(1):
        data = nuscenes_occ[i]

    a = 1