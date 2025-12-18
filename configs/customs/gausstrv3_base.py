_base_ = 'mmdet3d::_base_/default_runtime.py'

import os
work_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv3/debug' # todo

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs

custom_imports = dict(imports=['gausstrv3']) # todo

mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]

num_classes = 17



# input_size = (112,192)
# input_size = (448,800)
input_size = (504, 896)
resize_lim=[0.1244, 0.12]  #! 这个是提供了一个随机缩放比例的取值范围！(ImageAug3D中取消使用)
ori_image_shape = (900,1600)



# train_ann_file='nuscenes_mini_infos_train.pkl'
train_ann_file='nuscenes_mini_infos_val.pkl'
val_ann_file='nuscenes_mini_infos_val.pkl'



embed_dims = 256
feat_dims = 768

model = dict(
    type = 'GaussTRV3',
    num_queries=300,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor', # todo 图像数据预处理，打包为patch
        mean=mean,
        std=std),
    neck=dict(
        type='ViTDetFPN',
        in_channels=feat_dims,
        out_channels=embed_dims,
        norm_cfg=dict(type='LN2d')),
    decoder=dict(
        type='GaussTRDecoder',
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=embed_dims, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=embed_dims, num_levels=4),
            ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=2048)),
        post_norm_cfg=None),
    gauss_head=dict(
        type='GaussTRHead',
        regress_head=dict(type='MLP', input_dim=embed_dims, output_dim=3),
        opacity_head=dict(type='MLP', input_dim=embed_dims, output_dim=1, mode='sigmoid'),
        scale_head=dict(type='MLP',input_dim=embed_dims,output_dim=3,mode='sigmoid',range=(1, 16)),
        rot_head=dict(type='MLP', input_dim=embed_dims, output_dim=4,mode = 'normalize'),
        semantic_head=dict(type='MLP', input_dim=embed_dims, output_dim=num_classes,mode = 'softplus'),
        voxelizer=dict(
            type='GaussianVoxelizer',
            vol_range=[-40, -40, -1, 40, 40, 5.4],
            voxel_size=0.4,
            filter_gaussians=True,
            opacity_thresh=0.6,
            covariance_thresh=1.5e-2,
            ),
    ),
    model_url = '/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dinov2_vitb14_reg4_pretrain.pth',
    ori_image_shape = ori_image_shape,

)

# Data
dataset_type = 'NuScenesOccDataset'
# data_root = 'data/nuscenes/'
data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # todo
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
    # todo 载入occ标注
    dict(type='LoadOccFromFile'),
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
        apply_aug=True), # todo apply_aug = True: 进行了数据增强
     # todo -----------------#
     # todo 参考gausstr_feature.py
    dict(
        type='LoadFeatMaps',
        # data_root='data/nuscenes_grounded_sam2', # todo 分割数据集导入
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_grounded_sam2/mini',
        key='sem_seg',
        apply_aug=True),
    dict(
        type='Pack3DDetInputs', # todo 把预处理的原始数据转成标注的数据集输入格式
        # keys=['img'], # todo
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat',
            'sample_idx',
            'num_views', 'img_path', 'depth', 'feats',
            #--------------------------------------------#
            'img','sem_seg', # todo 2D分割图
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
         resize_lim=resize_lim,
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
        ann_file=train_ann_file,
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
        ann_file=val_ann_file, # todo ann文件：.pkl
        # ann_file='nuscenes_mini_infos_train.pkl',
        pipeline=test_pipeline, # todo 定义数据集处理流程
        **shared_dataset_cfg))
test_dataloader = val_dataloader

# todo 指标评估器
val_evaluator = dict(
    type='OccMetric',
    num_classes=18, # todo 类别： 17(Occ3D) + 1(1：天空类)
    use_lidar_mask=False,
    use_image_mask=True)
test_evaluator = val_evaluator



# Optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    type='OptimWrapper',   # 避免混合精度(AMP)
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
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

