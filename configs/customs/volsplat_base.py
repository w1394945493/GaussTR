_base_ = 'mmdet3d::_base_/default_runtime.py'
import os
work_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplat/outputs/vis2' # todo

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs

custom_imports = dict(imports=['volsplat']) # todo

mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]


sh_degree = 2
use_sh = sh_degree is not None
renderer_type = "vanilla"
# renderer_type = "gsplat"

# save_vis = False
save_vis = True
custom_hooks = [
    dict(type='DumpResultHook',
         interval=1,
         mean = mean,
         std  = std,
         save_dir = os.path.join(work_dir,'vis'),
         save_vis = save_vis,
         save_depth = True,
         save_img = True,
         ),
]  # 保存结果

input_size = (112,200)
ori_image_shape = (900,1600)

near = 0.5
far = 100.


# train_ann_file='nuscenes_mini_infos_train.pkl'
train_ann_file='nuscenes_mini_infos_val.pkl'
val_ann_file='nuscenes_mini_infos_val.pkl'

use_checkpoint = True
num_cams = 6

_dim_ = 128
num_heads = 8
num_layers = 1
patch_sizes=[8, 8, 4, 2]




model = dict(
    type = 'VolSplat',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor', # todo 图像数据：进行归一化处理，打包为patch
        mean=mean,
        std=std),
    
    ori_image_shape = ori_image_shape,
    use_checkpoint = use_checkpoint,
    
    in_embed_dim=_dim_,
    out_embed_dims=[_dim_, _dim_*2, _dim_*4, _dim_*4],
    voxel_resolution = 0.001,


    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='pretrained/dino_resnet50_pretrain.pth',
            checkpoint='/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dino_resnet50_pretrain.pth',
            prefix=None)),

    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    
    sparse_unet=dict(
        type='SparseUNetWithAttention',
        in_channels=_dim_, # 128
        out_channels=_dim_, # 128
        num_blocks=3,
        use_attention=False),    
    
    sparse_gs=dict(
        type='SparseGaussianHead',
        in_channels=_dim_, 
        out_channels=38),       
    
    gaussian_adapter=dict(
        type='GaussianAdapter_depth',
        gaussian_scale_min = 1e-10,
        gaussian_scale_max = 3.0,
        sh_degree=sh_degree,
    ),
    decoder = dict(
        type='DecoderSplatting',
        loss_lpips=dict(
            type='LossLpips',
            weight = 0.05,
        ),
        near = near,
        far = far,
        use_sh = use_sh,
        renderer_type = renderer_type,        
    )
)

# Data
dataset_type = 'NuScenesOccDataset' # todo NuScenesOCCDatasetV2
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
    dict(type='ImageAug3D',final_dim=input_size,),
    dict(
        type='LoadFeatMaps',
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini',
        key='depth',
        apply_aug=True),
     # todo -----------------#
     # todo 参考gausstr_feature.py
    dict(
        type='LoadFeatMaps',
        data_root='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_grounded_sam2/mini',
        key='sem_seg',
        apply_aug=True),

    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_semantic_seg'],
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
    batch_size=1,
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
val_evaluator = dict(type='ImgMetric')
test_evaluator = val_evaluator



# Optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    type='OptimWrapper',   # 避免混合精度(AMP)
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop') # todo
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]

default_hooks = dict(
     # todo
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=1)
)

