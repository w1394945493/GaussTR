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

                 transformerlayers=None,
                 positional_encoding=None,
                 return_intermediate=False,
                 ):
        super().__init__(transformerlayers, num_layers)

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


    def forward(self, mlvl_feats, project_feats, img_metas):
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        bev_queries_hw = self.bev_embedding_hw.weight.to(dtype)
        bev_queries_hw = bev_queries_hw.unsqueeze(0).repeat(bs, 1, 1)

        if project_feats[0] is not None:
            project_feats_hw = project_feats
            project_feats_hw = rearrange(self.project_transform_hw(project_feats_hw), "b c h w -> b (h w) c")
            bev_queries_hw = bev_queries_hw + project_feats_hw

            bev_pos_hw = self.positional_encoding(bs, device, 'z')

        # todo -------------------------------------#
        # todo 跨视图交叉注意力
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
