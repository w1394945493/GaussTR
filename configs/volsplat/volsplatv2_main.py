_base_ = 'mmdet3d::_base_/default_runtime.py'

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs
save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplatv2/outputs/vis28'

# custom_hooks = [
#     dict(type='DumpResultHook',
#          save_dir = save_dir, 
#          save_occ=True,
#          save_gaussian=True,    
#         ),]  #  # 保存结果

custom_imports = dict(imports=['volsplatv2']) # todo

# todo ----------------------------------#
# todo 图像预处理参数
mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]



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

# with_empty = False # 是否使用空高斯
with_empty = True # 是否使用空高斯

num_class = 18 # 语义维度
out_channels = 11 + 3 * (sh_degree + 1)**2 if sh_degree is not None else 14
# out_channels += num_class-1 if with_empty else num_class
out_channels += num_class

#! 高斯尺度相关
# voxel_resolution = 0.5
voxel_resolution = 0.2

# gaussian_scale_min = 0.1
# gaussian_scale_max = 0.5
gaussian_scale_min = 0.08
gaussian_scale_max = 0.64

use_checkpoint = True
_dim_ = 128
feat_dims = 768


#! depth_unet 相关
num_heads = 8
num_layers = 1
num_cams = 6
patch_sizes=[8, 8, 4, 2]


model = dict(
    type = 'VolSplat',

    use_checkpoint = use_checkpoint,
    
    # refine_voxel_resolution = refine_voxel_resolution,
    voxel_resolution = voxel_resolution,
    
    # backbone=dict(
    #     type='mmdet.ResNet',
    #     depth=50,
    #     in_channels=3,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(
    #         type='Pretrained',
    #         # checkpoint='pretrained/dino_resnet50_pretrain.pth',
    #         checkpoint='/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dino_resnet50_pretrain.pth',
    #         prefix=None)),

    # neck=dict(
    #     type='mmdet.FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=_dim_,
    #     # start_level=1,
    #     start_level = 0,
    #     add_extra_convs='on_input',
    #     num_outs=4),
    
    model_url = '/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dinov2_vitb14_reg4_pretrain.pth',
    neck=dict(
        type='ViTDetFPN',
        in_channels=feat_dims,
        out_channels=_dim_,
        norm_cfg=dict(type='LN2d')),    
    
    top_k = 25600,
    # top_k = 0,
    
    foreground_head=dict(
        type='SparseGaussianHead',
        in_channels=_dim_, 
        out_channels=out_channels,
        ),     
    
    lifter = dict(
        type='GaussianLifter',
        num_anchor=12800,
        embed_dims=_dim_,
        semantic_dim=num_class,
        pc_range=vol_range,
    ),
    
    encoder = dict(
        type = 'GaussianOccEncoder',
        num_decoder=3,
        anchor_encoder=dict(
            type='MLP',
            input_dim=out_channels, 
            output_dim=_dim_),      
        
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=_dim_ * 2,
            embed_dims=_dim_,
            feedforward_channels=_dim_ * 4,
        ),
        deformable_layer=dict(
                type='DeformableFeatureAggregation',
                embed_dims=_dim_,
                num_groups=4,
                num_levels=4,
                num_cams=num_cams,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseGaussian3DKeyPointsGenerator",
                    embed_dims=_dim_,
                    num_learnable_pts=4, # todo 可学习尺寸采样点数
                    learnable_fixed_scale=1,
                    fix_scale=None, # todo 固定尺寸采样点数：若为None，则为1
                    # fix_scale=[
                    #     [0, 0, 0],
                    #     [0.45, 0, 0],
                    #     [-0.45, 0, 0],
                    #     [0, 0.45, 0],
                    #     [0, -0.45, 0],
                    #     [0, 0, 0.45],
                    #     [0, 0, -0.45],
                    # ],
                    pc_range=vol_range,
                    scale_range=[gaussian_scale_min,gaussian_scale_max],
                ),
            ),    
        spconv_layer=dict(
            type='SparseConv3DModule', 
            in_channels=_dim_,
            embed_channels=_dim_,
            pc_range=vol_range,
            grid_size=[voxel_size, voxel_size, voxel_size],
        ),  
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=_dim_,
            output_dim = out_channels,
            semantic_dim = num_class,
            pc_range=vol_range,
            voxel_size = voxel_size,
            scale_range=[gaussian_scale_min,gaussian_scale_max],
        ), 
                    
    ),
    
    decoder = dict(
        type='GaussianDecoder',
        voxelizer = dict(
            type='GaussianVoxelizer',
            vol_range=vol_range,
            voxel_size=voxel_size,
            filter_gaussians=True,
        ),
        num_class = num_class,
        
        with_empty = with_empty,
        empty_args=dict(
            vol_range = vol_range,
            voxel_size = voxel_size
        ),
        
        near = near,
        far = far,
        use_sh = use_sh,
        renderer_type = renderer_type,

        scale_range=[gaussian_scale_min,gaussian_scale_max],        
        semantic_dim = num_class,
    )
)

# todo ----------------------------------#
# todo 训练：
batch_size=1
num_workers=4

train_batch_size=batch_size
train_num_workers=num_workers

val_batch_size=2
val_num_workers=num_workers

# data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval/' 
# anno_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/nuscenes/' # todo 全部训练
# logger_interval = 100
# train_ann_file = "nuscenes_infos_train_sweeps_occ.pkl"
# val_ann_file = "nuscenes_infos_val_sweeps_occ.pkl"



data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
anno_root = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/" # 标注根目录
logger_interval = 1
# train_ann_file = "nuscenes_mini_infos_train_sweeps_occ.pkl"
train_ann_file = "nuscenes_mini_infos_val_sweeps_occ.pkl"
# val_ann_file = "nuscenes_mini_infos_train_sweeps_occ.pkl"
val_ann_file = "nuscenes_mini_infos_val_sweeps_occ.pkl"


# occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/" # mini surroundocc标注根目录
# depth_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/mini'  # mini metric 3d depth
occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/samples/" # all
depth_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/samples_dptm_small" # 使用omni-scene提供的深度信息
    
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
]

test_pipeline = [
    dict(type="BEVLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadFeatMaps',data_root=depth_path, key='depth', apply_aug=True), #
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
]


# final_dim = (112,200)
final_dim = (448,800)
patch_size = 14
featmap_dim = [int(final_dim[0]/patch_size*4),int(final_dim[1]/patch_size*4)]
# final_dim = (896,1600)
# output_dim = (112,200)

data_aug_conf = {
    "final_dim": final_dim,
    "featmap_dim": featmap_dim,
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True, # todo 训练时做数据增强
    
    # "output_dim": output_dim,
}

train_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    imageset=anno_root + train_ann_file,
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
)

val_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    imageset=anno_root + val_ann_file,
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
)

seed = 42

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True if train_num_workers > 0 else False,
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
    persistent_workers=True if val_num_workers > 0 else False, # todo 当num_workers=0时, persistent_workers必须为False
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', 
                 shuffle=False,
                 seed=seed
                 ), # todo
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=val_dataset_config)

test_dataloader = val_dataloader


randomness = dict(
    seed=seed,
    deterministic=False,
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
    clip_grad=dict(max_norm=35, norm_type=2),
    
    # paramwise_cfg=dict(
    #         custom_keys={
    #             # 匹配背景透明度，学习率设为基础值的 10 倍 (2e-3)
    #             'decoder.empty_opa': dict(lr_mult=10.0),
    #             # 匹配背景强度标量，学习率设为基础值的 5 倍 (1e-3)
    #             'decoder.empty_scalar': dict(lr_mult=5.0),
                
                
    #             # 如果你有 backbone 且想让它收敛慢一点 (2e-5)
    #             'backbone': dict(lr_mult=0.1)
    #         }
    #     ),    
    
    )

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
    logger=dict(type='LoggerHook', 
                interval=logger_interval,
                ), # todo 管理打印间隔
    checkpoint=dict(type='CheckpointHook', 
                    interval=1,           # 含义：保存频率 默认单位通常是 Epoch（轮次）。
                    max_keep_ckpts=1,     # 最大保留数量（不包含“最优”权重）。
                    # save_best='miou',     # 开启“最优模型”保存机制。
                    # rule='greater',       # 越大越好
                    # published_keys=['miou','iou', 'psnr', 'ssim', 'lpips']
                    )
)
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True,
    mean_pattern=r'.*(time|data_time).*', # todo 对指定指标做滑动平均，其他则直接记录当前步的结果
    )   
 
# vis_backends = [
#     dict(type='LocalVisBackend'), # 本地保存（默认，生成 scalars.json）
#     dict(type='TensorboardVisBackend') # 调用 TensorBoard
# ]
# visualizer = dict(type='Visualizer',vis_backends=vis_backends,name='visualizer')

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True  # todo 多卡训练时设置，允许模型中有不参与计算的参数
)