import torch
import torch.nn as nn

from mmdet3d.registry import MODELS

@MODELS.register_module()
class GaussianLifter(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        semantic_dim,
        pc_range,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) 
        