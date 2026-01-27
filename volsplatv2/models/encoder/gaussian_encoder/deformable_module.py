import torch, torch.nn as nn
import numpy as np
from typing import List, Optional
from einops import rearrange,repeat

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from mmengine.model import xavier_init, constant_init



from ...utils import get_rotation_matrix, quat_to_rotmat
from ...utils import flatten_multi_scale_feats
from .utils import linear_relu_ln
from .ops import DeformableAggregationFunction as DAF

def  project_points(key_points, projection_mat, image_wh=None): # todo (1 25600 9 3) (1 6 4 4) (1 6 2)
        #? 齐次坐标变换：将3d坐标(x,y,z)末尾补1，变成(x,y,z,1)
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )  # todo (1 25600 9 3) -> (1 25600 9 4)
        #? 多视角投影：维度变化：利用广播机制，将所有anchor的各个关键点，投影到6个不同的相机视角
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)  # todo (1 6 25600 9 4)
        #? 透视除法：用投影结果的前两个维度除以第三个维度(深度)
        depth = points_2d[..., 2]
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )  # todo (1 6 25600 9 2)
        #? 归一化：将像素坐标除以特征图的宽高
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None] # todo (1 6 25600 9 2)
        #? 掩码筛选：①在相机前方；②水平坐标和垂直坐标均在[0,1]之间
        mask = (depth > 1e-5) & (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1) & \
                                (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1)
        return points_2d, mask   



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
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0) # todo 将相关全连接层进行参数初始化
    
    def forward(
        self,
        anchor, # todo (1 25600 28)
        instance_feature=None, # todo (1 25600 128) 值全为0
    ): 
        bs, num_anchor = anchor.shape[:2] # todo (1 25600 28)
        
        
        #? 据预测的anchor，通过旋转、平移和缩放，计算出一组相对于中心点的关键点
        #? 1.缩放基础构建：由两部分组成：固定尺度和可学习尺度
        fix_scale = anchor.new_tensor(self.fix_scale) # todo  self.fix_scale: (7 3) fix_scale: (7 3)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1]) # todo scale: (1 25600 7 3) .tile(): 按倍数铺平张量
        if self.num_learnable_pts > 0 and instance_feature is not None: # todo num_learnable_pts: 2
            learnable_scale = (
                torch.sigmoid(self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3))
                - 0.5 
            ) # todo (1 25600 2 3)  -0.5：让偏移量分布在[-0.5,0.5]之间
            scale = torch.cat([scale, learnable_scale * self.learnable_fixed_scale], dim=-2) # todo (1 25600 9 3)

        #? 2. 全局尺寸调整
        gs_scales = anchor[..., None, 3:6] # todo (1 25600 1 3)
        if self.scale_act == "sigmoid":
            gs_scales = torch.sigmoid(gs_scales)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales # todo scale_range: 0.08 ~ 0.64
        
        key_points = scale * gs_scales # todo (1 25600 9 3)
        rots = anchor[..., 6:10] # todo (b,25600,4)
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2) # todo (1 25600 3 3)

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None] 
        ).squeeze(-1) # todo (1 25600 9 3)

        xyz = anchor[..., :3] # todo (1 25600 3)
        # todo 最终坐标 = 旋转缩放后的偏移点 + 中心点坐标
        key_points = key_points + xyz.unsqueeze(2) 
        
        return key_points # todo (1 25600 9 3)

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
        self.kps_generator = MODELS.build(kps_generator)
        
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
        instance_feature: torch.Tensor, # todo (1 25600 128)
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor, # todo (1 25600 128)
        feature_maps: List[torch.Tensor],
        projection_mat, # todo (1 6 4 4)
        featmap_wh,
    ):  
        bs,n,_,_=projection_mat.shape
        #?--------------------------------------------#
        #? 1.生成3D采样点
        # todo 采样点
        key_points = self.kps_generator(anchor, instance_feature) # todo (1 25600 9 3)
        
        # todo 铺平后的多尺度特征图
        if self.use_deformable_func:
            feature_maps = list(flatten_multi_scale_feats(feature_maps))
            feature_maps[0] = rearrange(feature_maps[0],"(bs v) n c -> bs v n c",v=n)
        #?--------------------------------------------#
        #? 2.计算采样权重：把query的内容特征和位置特征相加，通过线性层预测每一个采样点权重       
        weights, weight_mask = self._get_weights(instance_feature, anchor_embed, projection_mat) # todo (1 25600 6 4 9 4) (1 25600 6 4 9 4)
           
        #?--------------------------------------------#
        #? 3.3D到2D的投影映射
        points_2d, mask = project_points(key_points, projection_mat, featmap_wh)
        _, _, num_anchor, num_pts, _ = points_2d.shape
        points_2d = points_2d.permute(0,2,3,1,4).reshape(bs, num_anchor * num_pts, n, 2) # todo (1 230400 6 2)
        mask = mask.permute(0, 2, 3, 1) # todo (1 25600 9 6)
        
        if self.use_deformable_func:
            weights = (weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
                bs, num_anchor,self.num_pts,self.num_cams,self.num_levels,self.num_groups,)) # todo (1 25600 9 6 4 4)        
            weight_mask = (weight_mask.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
                bs,num_anchor,self.num_pts,self.num_cams,self.num_levels,self.num_groups,)) # todo (1 25600 9 6 4 4)        
        
        #?--------------------------------------------#
        #? 4.权重归一化与非法点过滤    
        mask = mask[..., None, None] & weight_mask # todo (1 25600 9 6 4 4)
        all_miss = mask.sum(dim=[2, 3, 4], keepdim=True) == 0 # todo (1 25600 1 1 1 4)
        all_miss = all_miss.expand(-1, -1, self.num_pts, self.num_cams, self.num_levels, -1) # todo (1 25600 9 6 4 4)
        weights[~mask] = - torch.inf
        weights[all_miss] = 0.
        weights = weights.flatten(2, 4).softmax(dim=-2).reshape(bs,num_anchor * self.num_pts,self.num_cams,self.num_levels,self.num_groups) # todo (1 25600 9 6 4 4)
        weights = weights * (1 - all_miss.flatten(1, 2).float())
        features_next = DAF.apply(*feature_maps, points_2d, weights).reshape(bs, num_anchor, self.num_pts, self.embed_dims)  # todo (1 25600 9 128)  
        
        features = features_next # todo (1 25600 9 128)
        
        features = features.sum(dim=2) # todo (1 25600 128)
        output = self.proj_drop(self.output_proj(features)) # todo (1 25600 128)           
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1) # todo (1 25600 256)
        return output # todo (1 25600 128)  
    
    def _get_weights(self, instance_feature, anchor_embed, projection_mat=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(projection_mat[:, :, :3].reshape(bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (self.weights_fc(feature).reshape(bs, num_anchor, -1, self.num_groups).reshape(bs, num_anchor,self.num_cams,self.num_levels,self.num_pts,self.num_groups,))
        if self.training and self.attn_drop > 0:
            mask = torch.rand_like(weights)
            mask = mask > self.attn_drop
        else:
            mask = torch.ones_like(weights) > 0
        return weights, mask