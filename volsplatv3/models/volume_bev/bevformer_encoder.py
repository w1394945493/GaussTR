import torch
from torch import nn
from torch.nn.init import normal_
import numpy as np

from einops import rearrange
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmengine.registry import MODELS




@MODELS.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,

                 bev_only=False,
                 
                 pc_range=[-50,-50, -5, 50, 50, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 embed_dims=256,

                 num_points_in_pillar=[4],
                 
                 num_layers=5,
                 transformerlayers=None, # 必须定义！ num_layers和transformerlayers数量要一致

                 positional_encoding=None,
                 return_intermediate=False,
                 ):
        
        super().__init__(transformerlayers, num_layers) #!!! 注意力层

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.pc_range = pc_range

        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.real_z = pc_range[5] - pc_range[2]

        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
        self.bev_embedding_hw = nn.Embedding(bev_h * bev_w, embed_dims)

        if not bev_only:
            self.project_transform_hw = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1)
        
        ref_3d_hw = self.get_reference_points(bev_h, bev_w, self.real_z, num_points_in_pillar[0])
        self.register_buffer('ref_3d_hw', ref_3d_hw)

        self.positional_encoding = MODELS.build(positional_encoding) # 位置编码
        self.return_intermediate = return_intermediate
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, TPMSDeformableAttention3D) or isinstance(
        #             m, TPVCrossViewHybridAttention):
        #         m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @staticmethod
    def get_reference_points(H,
                             W,
                             Z=8,
                             num_points_in_pillar=4,
                             dim='3d',
                             bs=1,
                             device='cuda',
                             dtype=torch.float):
        """Get the reference points used in SCA and TSA.

        Args:
            H, W: spatial shape of tpv.
            Z: height of pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        zs = torch.linspace(
            0.5, Z - 0.5, num_points_in_pillar,
            dtype=dtype, device=device).view(-1, 1, 1).expand(
                num_points_in_pillar, H, W) / Z # Z方向从0.5到Z-0.5生成Z个点 (8,50 50)
        xs = torch.linspace(
            0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, -1).expand(
                num_points_in_pillar, H, W) / W # w方向从0.5到w-0.5生成W个点 归一化的值
        ys = torch.linspace(
            0.5, H - 0.5, H, dtype=dtype, device=device).view(1, -1, 1).expand(
                num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1) # (8 50 50 3)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1) # (1 8 50 50 3)
        return ref_3d

    # 计算参考点在二维图像中的位置和坐标
    def point_sampling(self, reference_points, pc_range, img_metas):
        h, w = img_metas['img_shape']
        lidar2img = img_metas['lidar2img'] # todo (1 6 4 4)
        B,N,_,_=lidar2img.shape
        reference_points = reference_points.clone() # (1 num h*w 3) 参考点归一化坐标
        
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]        

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1) # 变为齐次坐标 # (1 num h*w 4)
        
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B_ref, num_query = reference_points.size()[:3]

        num_cam = lidar2img.size(1)
        
        reference_points = reference_points.view(D, B_ref, 1, num_query, 4).repeat(
            1, B // B_ref, num_cam, 1, 1).unsqueeze(-1)
        
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4,
                                   4).repeat(D, 1, 1, num_query, 1, 1)
        
        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5        
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= w
        reference_points_cam[..., 1] /= h
        
        bev_mask = (
            bev_mask & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))

        bev_mask = torch.nan_to_num(bev_mask)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask


    def forward(self, mlvl_feats, project_feats, img_metas):
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        bev_queries_hw = self.bev_embedding_hw.weight.to(dtype)
        bev_queries_hw = bev_queries_hw.unsqueeze(0).repeat(bs, 1, 1)

        #----------------------------------------#
        # 投影特征与初始的BEV特征相加
        if project_feats[0] is not None:
            project_feats_hw = project_feats[0]
            project_feats_hw = rearrange(self.project_transform_hw(project_feats_hw), "b c h w -> b (h w) c")
            bev_queries_hw = bev_queries_hw + project_feats_hw

        bev_query = [bev_queries_hw]
        bev_pos_hw = self.positional_encoding(bs, device) # (1 2500 128)
        bev_pos = [bev_pos_hw]

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None,
                                            lvl:lvl + 1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)
        
        reference_points_cams, bev_masks = [], []
        ref_3ds = [self.ref_3d_hw]
        for ref_3d in ref_3ds:
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range,
                img_metas)              
            reference_points_cams.append(reference_points_cam)
            bev_masks.append(bev_mask)
        

        # todo --------------------------#
        # 注意力层编码
        intermediate = []
        for layer in self.layers:
            output = layer(
                bev_query,      # query
                feat_flatten,   # key
                feat_flatten,   # value
                bev_pos=bev_pos,
                bev_h=self.bev_h,
                bev_w=self.bev_w,                

                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                bev_masks=bev_masks,           
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

