from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmengine.model import xavier_init, constant_init
import torch, torch.nn as nn
import numpy as np
from typing import List, Optional
from ...utils.safe_ops import safe_sigmoid
from ...utils.utils import get_rotation_matrix
from .utils import linear_relu_ln
from .ops import DeformableAggregationFunction as DAF

@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        learnable_fixed_scale=1,
        fix_scale=None,
        pc_range=None,
        scale_range=None,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
    ):
        bs, num_anchor = anchor.shape[:2] # todo (b,M,28)
        fix_scale = anchor.new_tensor(self.fix_scale) # todo self.fix_scale: (7,3)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None: # todo num_learnable_pts: 2
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3))
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale * self.learnable_fixed_scale], dim=-2)

        gs_scales = anchor[..., None, 3:6] # todo 尺度
        if self.scale_act == "sigmoid":
            gs_scales = safe_sigmoid(gs_scales)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        key_points = scale * gs_scales
        rots = anchor[..., 6:10] # (b,25600,4)
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2) # todo 根据四元数计算旋转矩阵

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None] # (b 25600 1 3 3) x (b 25600 9 3 1) 旋转每个关键点
        ).squeeze(-1)

        xyz = anchor[..., :3]
        if self.xyz_act == 'sigmoid':
            xyz = safe_sigmoid(xyz)

        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        key_points = key_points + xyz.unsqueeze(2)
        return key_points # todo (b 25600 9 3) 得到一组三维anchor对应的关键点


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_deformable_func = use_deformable_func and DAF is not None
        assert self.use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        # todo ------------------------#
        # todo 生成一组三维稀疏的高斯关键点
        key_points = self.kps_generator(anchor, instance_feature) # todo anchor:(1,25600,28) feature: (1 25600 128) key_points: (1,25600,9,3)
        temp_key_points_list = (
            feature_queue
        ) = meta_queue = temp_anchor_embeds = []
        
        if self.use_deformable_func: # todo True: 输出的特征图：list: [0]: (1,6,total_num,128) [1]: (4,2)([112,200],[56,100],...) [2]:[4]:(0,22400 28000 29400)
            feature_maps = DAF.feature_maps_format(feature_maps) # todo 把多尺度特征图打包：1.拼接后的特征张量 2.每个尺度对应的空间索引列表 3.每个尺度对应的起始索引列表
        
        # todo list[(bs v 128 108 200) (bs v 128 54 100) (bs v 128 27 50) (bs v 128 14 25)] -> list[(bs v 108x200+... 128) (4 2) (4)]
        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
        ):
            weights, weight_mask = self._get_weights(
                instance_feature, temp_anchor_embed, metas # todo 嵌入了相机矩阵
            ) # todo 把查询特征整合为 (b 25600 num_cam num_level num_pt num_group) 的格式
            if self.use_deformable_func: # todo True
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor,
                        self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                ) # todo (b 25600 num_pts num_cams)
                weight_mask = (
                    weight_mask.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor,
                        self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                
                #???  ---------------------------------------------------------#
                #??? GaussianFormer的思路是：先定义一组三维空间中的高斯点
                #??? 然后将其投影到二维空间中，进行注意力交互
                points_2d, mask = self.project_points( #??? 将3DKeypoints投影到2D图像上，以及2D坐标和可见性mask
                    temp_key_points, # todo (1 25600 9 3)
                    temp_metas["projection_mat"], # todo (1 6 4 4) lidar2img
                    temp_metas.get("image_wh"), # todo (1 6 2) [1600 896]... 图像宽高
                ) # (b 25600 9 3) (b v 4 4) -> (b v 25600 9 2) (b v 25600 9) 每个点的可见性
                
                points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
                    bs, num_anchor * self.num_pts, self.num_cams, 2) # (b,25600x9 6 2)
                mask = mask.permute(0, 2, 3, 1) # (b,25600 9 6)
                mask = mask[..., None, None] & weight_mask # (b 25600 9 6 1 1) (b 25600 9 6 4 4) -> (b 25600 9 6 4 4) 4：4个特征层 4：4个采样点
                all_miss = mask.sum(dim=[2, 3, 4], keepdim=True) == 0 # 计算每个锚点在所有尺度、相机、特征层和采用点中的掩码值和，若为0，表示该点在所有维度下不可见的
                all_miss = all_miss.expand(-1, -1, self.num_pts, self.num_cams, self.num_levels, -1) # 扩展一下
                weights[~mask] = - torch.inf # 将mask为Fasle位置的权重设为负无穷
                weights[all_miss] = 0. # 进一步将完全不可见的点的权重设置为0
                weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
                    bs,
                    num_anchor * self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups) # 展平，softmax归一化
                # weights_clone = weights.detach().clone()
                # weights_clone[~all_miss.flatten(1, 2)] = 0.
                # weights = weights - weights_clone
                weights = weights * (1 - all_miss.flatten(1, 2).float())
                # todo ------------------------------------------#
                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d, weights  # points_2d (b 25600xnum_pts,num_cam,2) weights: (b 25600xnum_pts,)
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims) # 进行特征聚合 # todo 这里应该是注意力了
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            features = temp_features_next # todo (b 25600 9 128)

        features = features.sum(dim=2)  # fuse multi-point features # todo 这里采用的是融合多个采样点的特征
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add": 
            output = output + instance_feature
        elif self.residual_mode == "cat": # todo 使用的是cat
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                ) # todo : metas["projection_mat"]: (4 4)
            ) # todo (b v 128) 相机嵌入
            feature = feature[:, :, None] + camera_embed[:, None] # todo (b 25600 1 128) -> (b 25600 6 128)
        
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups) # (b 25600 6 144) -> (b 25600 216 4)
            # .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            ) # (b 25600 216 4) -> (b 25600 6 4 9 4)
        )
        if self.training and self.attn_drop > 0:
            # mask = torch.rand(
            #     bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            # )
            # mask = mask.to(device=weights.device, dtype=weights.dtype)
            # weights = ((mask > self.attn_drop) * weights) / (
            #     1 - self.attn_drop
            # )
            mask = torch.rand_like(weights)
            mask = mask > self.attn_drop
        else:
            mask = torch.ones_like(weights) > 0
        return weights, mask

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None): # todo 把3D 点投影到每个相机的2D坐标 + 可见性mask
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        ) # todo 给3D点补齐齐次坐标  [x y z 1] 4   x (4 4)
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1) # (b 6 1 1 4 4) x (b 1 25600 9 4 1)
        depth = points_2d[..., 2] # todo 深度z
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        ) # todo 除以深度得到像素坐标 x/z y/z
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        mask = (depth > 1e-5) & (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1) & \
                                (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1) # todo 可见性mask：深度是否大于0 是否落在相机范围内
        return points_2d, mask

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d, _ = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features
