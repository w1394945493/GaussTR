_base_ = 'mmdet3d::_base_/default_runtime.py'

import os
work_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/monosplat/ours/outputs/vis' # todo


custom_imports = dict(imports=['gausstr','monosplat'])


input_size = (112,192)
resize_lim=[0.1244, 0.12]

ori_image_shape = (900,1600)

near = 0.5
far = 100.

# mean=[123.675, 116.28, 103.53]
# std=[58.395, 57.12, 57.375]

mean=[0, 0, 0]
std=[255, 255, 255]

vit_type = 'vits'
model_url = '/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dinov2_vits14_pretrain.pth'
in_channels = 384
out_channels = [48, 96, 192, 384]

# save vis
custom_hooks = [
    dict(type='MonoSplatDumpResultHook',
         interval=1,
         save_dir = os.path.join(work_dir,'vis'),
         save_vis = True,
         mean = mean,
         std = std,
         save_img = True,
         ),
]

# model
model = dict(
    type = 'MonoSplat',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=mean,
        std=std,
        ),

    depth_head = dict(
        type='DPTHead',
        in_channels=in_channels,
        features=64,
        use_bn=False,
        out_channels=out_channels,
        # out_channels=[96, 192, 384,768],
        use_clstoken=False,
    ),
    cost_head = dict(
        type='CostHead',
        in_channels=in_channels,
        features=64,
        use_bn=False,
        out_channels=out_channels,
        # out_channels=[96, 192, 384, 768],
        use_clstoken=False,
    ),
    transformer = dict(
        type='MultiViewFeatureTransformer',
        num_layers=3,
        d_model=64,
        nhead=1,
        ffn_dim_expansion=4,
    ),
    decoder = dict(
        type='MonoSplatDecoder',
        loss_lpips=dict(
            type='LossLpips',
            weight = 0.05,
        ),
    ),
    near = near,
    far = far,
    ori_image_shape = ori_image_shape,

    vit_type = vit_type,
    model_url = model_url,

)
# Data
dataset_type = 'NuScenesOccDataset'
data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini'
data_prefix = dict(
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')
input_modality = dict(use_camera=True, use_lidar=False)
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles', # todo 读取多视角相机图像，相机参数矩阵 -> 组织数据格式 (自定义)
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(
        type='ImageAug3D', # todo 对图像数据进行缩放
        final_dim=input_size,
        resize_lim=resize_lim,
        # is_train=True # todo 训练时，先只做缩放，其他均不考虑

        ),
    dict(
        type='LoadFeatMaps', # todo 载入深度图 (自定义)
        # data_root='data/nuscenes_metric3d',
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini', # todo 深度图
        key='depth',
        apply_aug=True),
     # todo -----------------#
     # todo 参考gausstr_feature.py
    dict(
        type='LoadFeatMaps',
        # data_root='data/nuscenes_grounded_sam2', # todo 分割数据集
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_grounded_sam2/mini',
        key='sem_seg',
        apply_aug=True),
    dict(
        type='Pack3DDetInputs', # todo 把预处理的原始数据转成标注的数据集输入格式
        keys=['img'], # todo
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat',
            'sample_idx',
            'num_views', 'img_path', 'depth', 'feats',
            #--------------------------------------------#
            'sem_seg', # todo 2D分割图
            # -------------------------------------------#
            'token','scene_token','scene_idx',
        ] # todo 返回 'data_samples'
        )
]
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(type='LoadOccFromFile'),
    # dict(type='ImageAug3D', final_dim=input_size, resize_lim=[0.56, 0.56]),
    dict(type='ImageAug3D',
         final_dim=input_size,
         resize_lim=resize_lim

         ),
    dict(
        type='LoadFeatMaps',
        # data_root='data/nuscenes_metric3d', # todo
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini',
        key='depth',
        apply_aug=True),
     # todo -----------------#
     # todo 参考gausstr_feature.py
    dict(
        type='LoadFeatMaps',
        # data_root='data/nuscenes_grounded_sam2', # todo 分割数据集导入
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_grounded_sam2/mini',
        key='sem_seg',
        apply_aug=True),

    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_semantic_seg'], # todo img：2D输入图 'gt_semantic_seg': 3D occ图
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat',
            'sample_idx',
            'num_views', 'img_path', 'depth', 'feats', 'mask_camera',
            # -------------------------------------------#
            'img','sem_seg',
            # -------------------------------------------#
            'token','scene_token','scene_idx',
        ])
]

shared_dataset_cfg = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    data_prefix=data_prefix,
    # load_adj_frame = True, # TODO
    load_adj_frame = False,
    interval = 1,
    filter_empty_gt=False)

train_dataloader = dict(
    batch_size=1,
    # num_workers=4,
    # num_workers=1,
    # persistent_workers=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        # ann_file='nuscenes_infos_train.pkl',
        # ann_file='nuscenes_mini_infos_train.pkl',
        ann_file='nuscenes_mini_infos_val.pkl',
        pipeline=train_pipeline,
        **shared_dataset_cfg))
val_dataloader = dict(
    batch_size=2,
    # num_workers=4,
    # num_workers=1,
    # persistent_workers=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # ann_file='nuscenes_infos_val.pkl',
        ann_file='nuscenes_mini_infos_val.pkl', # todo ann文件：.pkl
        # ann_file='nuscenes_mini_infos_train.pkl',
        pipeline=test_pipeline, # todo 定义数据集处理流程
        **shared_dataset_cfg))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='EvalMetric',
    )
test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper', # 启用混合精度(AMP)
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]

default_hooks = dict(
     # todo
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=1)
)
