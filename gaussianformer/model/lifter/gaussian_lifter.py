import torch, torch.nn as nn
from mmdet3d.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class GaussianLifter(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        semantics=False,
        semantic_dim=None, # todo 17维度，没有包含背景类
        include_opa=True,
        pts_init=False,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pts_init = pts_init
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        assert not (pts_init and anchor_grad)

        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # todo (25600 3) 生成0-1之间的随机数
        if xyz_activation == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz) # todo 将0-1反向映射回 -∞ - +∞

        scale = torch.rand_like(xyz)
        if scale_activation == "sigmoid":
            scale = safe_inverse_sigmoid(scale) # todo 将0-1反向映射回 -∞ - +∞

        rots = torch.zeros(num_anchor, 4, dtype=torch.float) # todo 初始的旋转四元数 [1 0 0 0]
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float)) # todo 初始透明度 全为0
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # todo 正态分布，大部分在0-1之间

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # todo (25600,3+3+4+1+17=28)

        self.num_anchor = num_anchor
        # todo -------------------------------------------------#
        # todo 采用可学习的初始化方式：
        # todo 论文C.初始化方式对对模型性能具有重要影响：包括：均匀分布初始化，预测伪点云初始化，真实点云初始化
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )

    def init_weights(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init).to(self.anchor.device)
        # self.anchor.data = torch.as_tensor(self.anchor_init,device=self.anchor.device,dtype=self.anchor.data.dtype)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, metas=None, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile( # todo torch.title: 复制张量
            self.instance_feature[None], (batch_size, 1, 1)
        ) # todo (b N 128) eg. N=25600

        # todo --------------------------------#
        # todo 高斯点初始化
        if self.pts_init and metas is not None: # todo False
            if self.xyz_act == "sigmoid":
                xyz = safe_inverse_sigmoid(metas['anchor_points'])
            anchor = torch.cat([
                xyz, torch.tile(self.anchor[None, :, 3:], (batch_size, 1, 1))], dim=-1)
        else:
            anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # todo self.anchor: [25600,28]

        return {
            'rep_features': instance_feature, # (b,25600,128)
            'representation': anchor, # (b, 25600,28)
            'anchor_init': self.anchor.clone() # (25600,28)
        }
