_base_ = 'mmdet3d::_base_/default_runtime.py'

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs
save_dir = '/vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv3/outputs/vis2'

custom_hooks = [
    dict(type='DumpResultHook',
         save_dir = save_dir, 
        ),]  #  # 保存结果

custom_imports = dict(imports=['volsplatv3']) # todo

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

# with_empty = False # 是否使用空高斯进行占位
with_empty = True 

num_class = 18 # 语义维度
out_channels = 11 + 3 * (sh_degree + 1)**2 if sh_degree is not None else 14
out_channels += num_class

#! 高斯尺度相关
# voxel_resolution = 0.5
voxel_resolution = 0.5 # todo 0.001 - too small, 0.1, 0.2, 0.5

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

pc_range = vol_range
pc_xrange, pc_yrange, pc_zrange = pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]


tpv_h_ = 50
tpv_w_ = 50
tpv_z_ = 4
scale_h = 1
scale_w = 1
scale_z = 1
# gpv = 3 # todo 每个tpv体素位置的高斯数量
gpv = 1

num_points_in_pillar = [8, 16, 16]
num_points = [16, 32, 32]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0
_ffn_dim_ = _dim_ * 2


self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1),
        dict(
            type='TPVImageCrossAttention',
            pc_range=pc_range,
            num_cams=6,
            dropout=0.1,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_),
            embed_dims=_dim_,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm'))

model = dict(
    type = 'VolSplat',
    use_checkpoint = use_checkpoint,
    voxel_resolution = voxel_resolution,
    pc_range=pc_range,
    
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
            checkpoint='/c20250502/wangyushen/Weights/pretrained/dino_resnet50_pretrain.pth',
            prefix=None)),

    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        # start_level=1,
        start_level = 0,
        add_extra_convs='on_input',
        num_outs=4),

    sparse_unet=dict(
        type='SparseUNetWithAttention',
        # type='SparseUNetSpconv',
        in_channels=_dim_, # 128
        out_channels=_dim_, # 128
        num_blocks=3,
        ),   
    sparse_gs =  dict(
        type='SparseGaussianHead',
        # type='SparseGaussianHeadSpconv',
        in_channels=_dim_, 
        out_channels=out_channels,
        ),  
    gaussian_adapter=dict(
        type='GaussianAdapter_depth',
        gaussian_scale_min = gaussian_scale_min,
        gaussian_scale_max = gaussian_scale_max,        
        
        sh_degree=sh_degree,
    ),  

    volume_gs = dict(
        type="VolumeGaussian",
        use_checkpoint=use_checkpoint,

        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=pc_range,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer, self_cross_layer, self_layer
            ],
            embed_dims=_dim_,
            positional_encoding=dict(
                type='TPVFormerPositionalEncoding',
                num_feats=[48, 48, 32],
                h=tpv_h_,
                w=tpv_w_,
                z=tpv_z_)),
        
        gs_decoder = dict(
            type='VolumeGaussianDecoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            pc_range=pc_range,
            # gs_dim=14,
            gs_dim=out_channels, # 14+18=32
            
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            
            scale_h=scale_h,
            scale_w=scale_w,
            scale_z=scale_z,
            
            gpv=gpv,
            
            offset_max=[
                2 * pc_xrange / (tpv_h_*scale_h), 
                2 * pc_yrange / (tpv_w_*scale_w), 
                2 * pc_zrange / (tpv_z_*scale_z)], # 位置偏移量最大预测值
            
            # scale_max=[
            #     2 * pc_xrange / (tpv_h_*scale_h), 
            #     2 * pc_yrange / (tpv_w_*scale_w), 
            #     2 * pc_zrange / (tpv_z_*scale_z)], # 高斯尺度最大预测值
            gaussian_scale_min = gaussian_scale_min,
            gaussian_scale_max = gaussian_scale_max,         
        
        )
        
    ),

    decoder = dict(
        type='GaussianDecoder',
        
        voxelizer = dict(
            type='GaussianVoxelizer',
            vol_range=vol_range,
            voxel_size=voxel_size,
            cuda_kwargs=dict(
                scale_multiplier=3,
                H=200, W=200, D=16,
                pc_min=[-50.0, -50.0, -5.0],
                grid_size=voxel_size), #!   
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
    )
        
)

# todo ----------------------------------#
# todo 训练：
batch_size=1
num_workers=4

train_batch_size=batch_size
train_num_workers=num_workers

val_batch_size=batch_size
val_num_workers=num_workers

logger_interval = 1 # 打印间隔
val_interval=1 # 评估间隔

data_root = '/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/' 
anno_root = '/c20250502/wangyushen/Datasets/NuScenes/nuscenes_cam/v1.0-trainval/' # todo 全部训练
# train_ann_file = "nuscenes_infos_train_sweeps_occ.pkl"
train_ann_file = "nuscenes_infos_val_sweeps_occ.pkl"
val_ann_file = "nuscenes_infos_val_sweeps_occ.pkl"

occ_path = "/c20250502/wangyushen/Datasets/surroundocc/samples/" # all
depth_path = "/c20250502/wangyushen/Datasets/dataset_omniscene/samples_dptm_small" # 使用omni-scene提供的深度信息
    


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
featmap_dim = [int(final_dim[0]/4),int(final_dim[1]/4)]



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