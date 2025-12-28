data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
anno_root = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/" # 标注根目录
occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/" # occ标注根目录

dataset_type = 'NuScenesSurroundOccDataset'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    # dict(type="PhotoMetricDistortionMultiViewImage"), # todo
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True, # todo 训练时做数据增强
}

train_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    # imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    # imageset=anno_root + "nuscenes_mini_infos_train_sweeps_occ.pkl",
    imageset=anno_root + "nuscenes_mini_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    # phase='train',
    phase='val',
)

val_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    # imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    # imageset=anno_root + "nuscenes_mini_infos_train_sweeps_occ.pkl",
    imageset=anno_root + "nuscenes_mini_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
    # phase='train',
)

seed = 42

# import mmengine.dataset.sampler
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    # sampler=dict(type='DefaultSampler', shuffle=True, seed=seed), # todo
    sampler=None,
    # shuffle=True, # todo
    shuffle=False,
    # worker_init_fn = None, # todo 不可以设置None
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=train_dataset_config)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    # sampler=dict(type='DefaultSampler', shuffle=False, seed=seed), # todo
    sampler=None,
    shuffle=False, # todo
    # worker_init_fn = None, # todo
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=val_dataset_config)

test_dataloader = val_dataloader

randomness = dict(
    seed=seed,
    # deterministic=False,
    # deterministic=True, # todo 是否开启确定性计算，默认False
    # diff_rank_seed=False, # todo 默认False
) # todo 随机数种子设置