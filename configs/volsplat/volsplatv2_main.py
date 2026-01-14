_base_ = 'mmdet3d::_base_/default_runtime.py'

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs
# save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplatv2/outputs/vis4'
# custom_hooks = [
#     dict(type='DumpResultHook',
#          save_dir = save_dir,
#          save_img=False,
#          save_depth=False, 
#          save_occ=False,     
#         ),
# ]  #  # 保存结果

custom_imports = dict(imports=['volsplatv2']) # todo

# todo ----------------------------------#
# todo 图像预处理参数
mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]

# todo ----------------------------------#
# todo 训练：
batch_size=4
# num_workers=16
# batch_size=1
num_workers=4

train_batch_size=batch_size
train_num_workers=num_workers

val_batch_size=batch_size
val_num_workers=num_workers

# train_ann_file = "nuscenes_mini_infos_train_sweeps_occ.pkl"
# train_ann_file = "nuscenes_mini_infos_val_sweeps_occ.pkl"
train_ann_file = "nuscenes_infos_train_sweeps_occ.pkl"
# val_ann_file = "nuscenes_mini_infos_train_sweeps_occ.pkl"
# val_ann_file = "nuscenes_mini_infos_val_sweeps_occ.pkl"
val_ann_file = "nuscenes_infos_val_sweeps_occ.pkl"

# todo ----------------------------------#
# todo 视图渲染相关参数
# sh_degree = 2 # todo d_sh = (sh_degree + 1)**2
sh_degree = None
use_sh = sh_degree is not None

# renderer_type = "vanilla"
renderer_type = "gsplat"
# near = 0.5
# far = 100.
near = 0.1
far = 1000.

# todo ----------------------------------#
# todo 占用预测相关参数
vol_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size=0.5
use_embed = True # todo 是否额外增加一些可学习嵌入
num_embed=1800
num_class = 18 # 语义维度
out_channels = 11 + 3 * (sh_degree + 1)**2 if sh_degree is not None else 14
out_channels += num_class
#! 高斯尺度相关
voxel_resolution = voxel_size / 5
gaussian_scale_min = voxel_size / 3.0
gaussian_scale_max = 10 * voxel_size

use_checkpoint = True
_dim_ = 128

model = dict(
    type = 'VolSplat',

    use_checkpoint = use_checkpoint,
    
    in_embed_dim=_dim_,
    out_embed_dims=[_dim_, _dim_*2, _dim_*4, _dim_*4],
    # voxel_resolution = 0.001,
    voxel_resolution = voxel_resolution,
    vol_range=vol_range,
    
    use_embed=use_embed,
    num_embed=1800,
    
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
        use_attention=False, 
        # use_attention=True, # 是否引入一个注意力层   
        ), 
    
    sparse_gs=dict(
        type='SparseGaussianHead',
        in_channels=_dim_, 
        out_channels=out_channels),       
    
    gaussian_adapter=dict(
        type='GaussianAdapter_depth',
        # gaussian_scale_min = 1e-10,
        # gaussian_scale_max = 3.0,
        
        gaussian_scale_min = gaussian_scale_min,
        gaussian_scale_max = gaussian_scale_max,        
        
        sh_degree=sh_degree,
    ),
    decoder = dict(
        type='GaussianDecoder',
        voxelizer = dict(
            type='GaussianVoxelizer',
            vol_range=vol_range,
            voxel_size=voxel_size,
            filter_gaussians=True,
        ),
        loss_lpips=dict(
            type='LossLpips',
            weight = 0.05,
        ),
        lovasz_ignore = num_class-1,
        near = near,
        far = far,
        use_sh = use_sh,
        renderer_type = renderer_type,        
    )
)

# ----------------------------------------------------------#
# Data
# data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
# anno_root = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/" # 标注根目录
data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval/' 
anno_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/nuscenes/' 
    
# occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/" # mini surroundocc标注根目录
# depth_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini'  # mini metric 3d depth
occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/samples/" # all
depth_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/samples_dptm_small" # all
    
dataset_type = 'NuScenesSurroundOccDataset'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="BEVLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type='LoadFeatMaps',data_root=depth_path, key='depth', apply_aug=True), #
    dict(type="PhotoMetricDistortionMultiViewImage"), # todo 
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="BEVLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadFeatMaps',data_root=depth_path, key='depth', apply_aug=True), #
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]


# final_dim = (112,200)
final_dim = (448,800)
output_dim = (112,200)

data_aug_conf = {
    "final_dim": final_dim,
    
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True, # todo 训练时做数据增强
    
    "output_dim": output_dim,
}

train_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    # imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    # imageset=anno_root + "nuscenes_mini_infos_train_sweeps_occ.pkl",
    imageset=anno_root + train_ann_file,
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    # load_adj_frame = True, # todo 训练时，不引入相邻帧图像
    load_adj_frame = False,
    interval=15,
    phase='train',
    # phase='val',
)

val_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    # imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    # imageset=anno_root + "nuscenes_mini_infos_train_sweeps_occ.pkl",
    imageset=anno_root + val_ann_file,
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    load_adj_frame = False, # todo 评估时，不引入相邻帧图像
    # phase='val',
    phase='val',
)

seed = 42

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True if train_num_workers > 0 else False,
    # persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', 
                 shuffle=True, 
                 seed=seed
                 ), # todo
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=train_dataset_config)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=val_num_workers,
    persistent_workers=True if val_num_workers > 0 else False, # todo num_workers=0, persistent_workers必须为False
    # persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', 
                 shuffle=False,
                #  shuffle=True, 
                 seed=seed
                 ), # todo
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=val_dataset_config)

test_dataloader = val_dataloader


randomness = dict(
    seed=seed,
    deterministic=False,
    # deterministic=True, # todo 是否开启确定性计算，默认False
    # diff_rank_seed=False, # todo 默认False
) # todo 随机数种子设置

# todo 指标评估器
val_evaluator = dict(type='OccMetric',
    class_indices = list(range(1, 17)),
    empty_label = 17,
    label_str = ['barrier', 'bicycle', 'bus', 'car', 'cons.veh',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'drive.surf', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'],
    dataset_empty_label = 17,
    filter_minmax = False,)
test_evaluator = val_evaluator

# todo 评估间隔
train_cfg = dict(type='EpochBasedTrainLoop', 
                 max_epochs=24, 
                 val_interval=1, # todo 评估间隔
                #  val_interval=2,
                 )
val_cfg = dict(type='ValLoop') # todo
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',   # 避免混合精度(AMP)
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]

# base_lr = 2e-4
# min_lr_ratio = 0.1
# warmup_iters = 500

# Optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',   # 避免混合精度(AMP)
#     optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01,),
#     paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
#     clip_grad=dict(max_norm=35, norm_type=2))

# param_scheduler = [
#     dict(
#         type='CosineLR',
#         lr_min = base_lr * min_lr_ratio,
#         warmup_t = warmup_iters,
#         warmup_lr_init = 1e-6,
#         by_epoch=False,
#     ),
# ]



default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10,),# todo 管理打印间隔
    checkpoint=dict(type='CheckpointHook', 
                    interval=1,           # 含义：保存频率 默认单位通常是 Epoch（轮次）。
                    max_keep_ckpts=1,     # 最大保留数量（不包含“最优”权重）。
                    save_best='miou',     # 开启“最优模型”保存机制。
                    rule='greater',       # 越大越好
                    # published_keys=['miou','iou', 'psnr', 'ssim', 'lpips']
                    )
)
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True,
    mean_pattern=r'.*(time|data_time).*', # todo 对指定指标做滑动平均，其他则直接记录当前步的结果
    )    
vis_backends = [
    dict(type='LocalVisBackend'), # 本地保存（默认，生成 scalars.json）
    dict(type='TensorboardVisBackend') # 调用 TensorBoard
]
visualizer = dict(type='Visualizer',vis_backends=vis_backends,name='visualizer')

# model_wrapper_cfg = dict(
#     type='MMDistributedDataParallel',
#     find_unused_parameters=True  # 允许模型中有不参与计算的参数
# )