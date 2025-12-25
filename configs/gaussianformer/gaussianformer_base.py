_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    './_base_/model.py',
    './_base_/surroundocc.py',
]

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs

custom_imports = dict(imports=['gaussianformer']) # todo

# ========= model config ===============
embed_dims = 128
num_decoder = 4
num_single_frame_decoder = 1
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True

semantics = True
semantic_dim = 17

model = dict(
    img_backbone_out_indices=[0, 1, 2, 3],

    img_backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1, # todo 冻结：
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True, # todo BN使用eval模式(不更新均值方差)
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),),

    img_neck=dict(
        type='mmdet.FPN',
        start_level=1), # todo 主干网络

    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
    )
)

# todo 指标评估
val_evaluator = dict(
    type='OccMetric',
    class_indices = list(range(1, 17)),
    empty_label = 17,
    label_str = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
    dataset_empty_label = 17,
    filter_minmax = False,
    )

test_evaluator = val_evaluator

max_epochs = 24
base_lr = 2e-4
min_lr_ratio = 0.1
warmup_iters = 500
iters_per_epoch = 81
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
#     clip_grad=dict(max_norm=35, norm_type=2))

# param_scheduler = [
#     dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
#     dict(type='MultiStepLR', milestones=[16], gamma=0.1)
# ]

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',   # 避免混合精度(AMP)
    optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01,),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2))

# param_scheduler = [
#     dict(type='LinearLR', start_factor=1e-6 / base_lr, begin=0, end=500, by_epoch=False),
#     dict(type='MultiStepLR', milestones=[16], gamma=0.1)
# ]

# param_scheduler = [
#     dict(
#         # type='CosineLRScheduler',
#         type = 'LinearLR',
#         start_factor=1e-6 / base_lr,
#         begin=0,
#         end=warmup_iters,
#         by_epoch=False,
#     ),
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=base_lr * min_lr_ratio,
#         begin=warmup_iters,
#         # end=max_epochs * iters_per_epoch, # todo 无需定义，by_epoch = False/True，会自动补该参数为max_iters/max_epochs
#         by_epoch=False,
#     )
# ]

param_scheduler = [
    dict(
        type='CosineLR',
        lr_min = base_lr * min_lr_ratio,
        warmup_t = 500,
        warmup_lr_init = 1e-6,
        by_epoch=False,
    ),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=1)
)

custom_hooks = [
    dict(type='DumpResultHook',),
]  #