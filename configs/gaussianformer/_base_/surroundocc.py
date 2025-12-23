data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini' # 数据集根目录
occ_path = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/surroundocc/mini_samples/" # occ标注根目录

train_ann_file='nuscenes_mini_infos_train.pkl'
val_ann_file='nuscenes_mini_infos_val.pkl'

dataset_type = 'NuScenesSurroundOccDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]


test_pipeline = [
    dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]



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


input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

shared_dataset_cfg = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    data_prefix=data_prefix,
    filter_empty_gt=False)


train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=dict(
        ann_file=train_ann_file,
        data_aug_conf=data_aug_conf,
        pipeline=train_pipeline,
        **shared_dataset_cfg))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=dict(
        ann_file=val_ann_file,
        data_aug_conf=data_aug_conf,
        pipeline=test_pipeline,
        **shared_dataset_cfg))

test_dataloader = val_dataloader