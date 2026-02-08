_base_='mmdet3d::_base_/default_runtime.py'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


# todo ----------------------------------#
# todo 训练：
batch_size=1
num_workers=4

train_batch_size=batch_size
train_num_workers=num_workers

val_batch_size=2
val_num_workers=num_workers

data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval/' 
anno_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/nuscenes/' # todo 全部训练
logger_interval = 100
train_ann_file = "nuscenes_infos_train_sweeps_occ.pkl"
val_ann_file = "nuscenes_infos_val_sweeps_occ.pkl"

occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/samples/" # all
depth_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_metric3d/samples_dptm_small" # 使用omni-scene提供的深度信息
    
dataset_type = 'NuScenesSurroundOccDataset'

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


final_dim = (448,800)
patch_size = 14
featmap_dim = [int(final_dim[0]/patch_size*4),int(final_dim[1]/patch_size*4)]

data_aug_conf = {
    "final_dim": final_dim,
    "featmap_dim": featmap_dim,
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True, # todo 训练时做数据增强
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
    collate_fn=dict(type='custom_collate_fn_temporal'), #! 
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
    collate_fn=dict(type='custom_collate_fn_temporal'), #!
    dataset=val_dataset_config)

test_dataloader = val_dataloader

randomness = dict(
    seed=seed,
    deterministic=False,
)

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
    clip_grad=dict(max_norm=35, norm_type=2),)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]

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

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True  # todo 多卡训练时设置，允许模型中有不参与计算的参数
)