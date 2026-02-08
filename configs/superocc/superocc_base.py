_base_ = './_base_/surroundocc.py'

custom_imports = dict(imports=['superocc'])


point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.5, 0.5, 0.5]
scale_range = [0.01, 3.2]
u_range = [0.1, 2]
v_range = [0.1, 2]

# arch config
embed_dims = 256
num_layers = 6
num_query = 3600
memory_len = 3000
topk_proposals = 3000
num_propagated = 3000

prop_query = True
temp_fusion = True
with_ego_pos = True
num_frames = 8
num_levels = 4
num_points = 2
num_refines = [1, 1, 2, 2, 4, 4]



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

occ_names = [
     'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
     'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
     'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
     'vegetation'
]

seq_mode = True


use_ego = False
ignore_label = 0
manual_class_weight = [
    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
    1.26960524, 1.06258364, 1.189019, 1.06217292, 1.00595144, 0.85706115,
    1.03923299, 0.90867526, 0.8936431, 0.85486129, 0.8527829, 0.5]



model = dict(
    type='SuperOCC', #!
    seq_mode=seq_mode,
    data_aug=dict(
        img_color_aug=False,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)
    ),
    stop_prev_grad=0,
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint="ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth",
            prefix='backbone.'),
        type='mmdet.ResNet', #! from mmdet.models import ResNet,FPN
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch',
        # pretrained='torchvision://resnet50'
    ),
    img_neck=dict(
        type='mmdet.FPN', #!
        in_channels=[256, 512, 1024, 2048],
        out_channels=embed_dims,
        num_outs=num_levels),
    pts_bbox_head=dict(
        type='StreamOccHead', #!
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        memory_len=memory_len,
        topk_proposals=topk_proposals,
        num_propagated=num_propagated,
        prop_query=prop_query,
        temp_fusion=temp_fusion,
        with_ego_pos=with_ego_pos,
        scale_range=scale_range,
        u_range=u_range,
        v_range=v_range,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        manual_class_weight=manual_class_weight,
        ignore_label=ignore_label,
        transformer=dict(
            type='StreamOccTransformer', #!
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            pc_range=point_cloud_range,
            use_ego=use_ego
        ),
        loss_occ=dict( 
            type='CELoss', #!
            activated=True,
            loss_weight=10.0
        ),
        loss_pts=dict(type='mmdet.SmoothL1Loss', #! from mmdet.models.losses import SmoothL1Loss
                      beta=0.2, loss_weight=0.5),
    )
)



