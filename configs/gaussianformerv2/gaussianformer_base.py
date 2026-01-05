_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    './_base_/model.py',
    './_base_/surroundocc.py',
]
# import mmdet3d
# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
# from mmdet3d.datasets.transforms import Pack3DDetInputs

custom_imports = dict(imports=['gaussianformerv2']) # todo

# ========= custom hooks ===============
save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformerv2/outputs/vis7'
custom_hooks = [
    dict(type='DumpResultHook',
         save_dir = save_dir,
         save_img=True,
         save_depth=True,         
        ),
]  #

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


# ----------------------------
# pixel_gauss
use_checkpoint = True
num_cams = 6

_dim_ = 128
num_heads = 8
num_layers = 1
patch_sizes=[8, 8, 4, 2]

near = 0.1
far = 1000.

model = dict(
    img_backbone_out_indices=[0, 1, 2, 3],
    loss_lpips=dict(
        type='LossLpips',
        weight = 0.05,
    ),   
    
    # img_backbone=dict(
    #     _delete_=True,
    #     type='mmdet.ResNet',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1, # todo 冻结：
    #     norm_cfg=dict(type='BN2d', requires_grad=False),
    #     norm_eval=True, # todo BN使用eval模式(不更新均值方差)
    #     style='caffe',
    #     with_cp = True,
    #     dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    #     stage_with_dcn=(False, False, True, True),),
    # img_neck=dict(
    #     type='mmdet.FPN',
    #     start_level=1), # todo 主干网络
    
    img_backbone=dict(
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
    
    

    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),    
    
    pixel_gs=dict(
        type="PixelGaussian",
        use_checkpoint=use_checkpoint,
        down_block=dict(
            type='MVDownsample2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        up_block=dict(
            type='MVUpsample2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        mid_block=dict(
            type='MVMiddle2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        patch_sizes=patch_sizes,
        in_embed_dim=_dim_,
        out_embed_dims=[_dim_, _dim_*2, _dim_*4, _dim_*4],
        num_cams=num_cams,
        near=near,
        far=far,

        ),
    


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
            # type="SparseConv3D", #! mmdet3d中已注册SparseConv3D
            type='CustomSparseConv3D', 
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
    label_str = ['barrier', 'bicycle', 'bus', 'car', 'cons.veh',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'drive.surf', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
    dataset_empty_label = 17,
    filter_minmax = False,
    )

test_evaluator = val_evaluator

# max_epochs = 20
max_epochs = 24

base_lr = 2e-4
min_lr_ratio = 0.1
warmup_iters = 500

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    type='OptimWrapper',   # 避免混合精度(AMP)
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]

# Optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',   # 避免混合精度(AMP)
#     optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01,),
#     paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
#     clip_grad=dict(max_norm=35, norm_type=2))

# param_scheduler = [
#     dict(
#         type='CosineLR',
#         lr_min = base_lr * min_lr_ratio,
#         warmup_t = 500,
#         warmup_lr_init = 1e-6,
#         by_epoch=False,
#     ),
# ]

default_hooks = dict(
    logger=dict(type='LoggerHook',
                interval=1, # todo 管理 训练 loss / metrics 的间隔
                ),
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=1)
)

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True,
    # mean_pattern=r'.*(loss|time|data_time|grad_norm).*',
    mean_pattern=r'.*(time|data_time).*', # todo 对指定指标做滑动平均
    )
# custom_hooks = [
#     dict(type='DumpResultHook',),
# ]  #

# from mmengine.visualization import Visualizer

vis_backends = [
    dict(type='LocalVisBackend'), # 本地保存（默认，生成 scalars.json）
    dict(type='TensorboardVisBackend') # 调用 TensorBoard
]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer'
)


