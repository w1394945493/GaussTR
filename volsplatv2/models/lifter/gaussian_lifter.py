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
        scale_range,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.scale_range = scale_range
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) 
        pc_min = torch.tensor(self.pc_range[:3], device=xyz.device)
        pc_max = torch.tensor(self.pc_range[3:], device=xyz.device)
        means = pc_min + (pc_max - pc_min) * xyz
        
        
        scales = torch.rand_like(xyz)    
        
        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1
        
        opacities = torch.rand(num_anchor,1, dtype=torch.float)
        
        colors = torch.rand_like(xyz)
        
        
        semantics = torch.randn(num_anchor, semantic_dim, dtype=torch.float) 
        anchor = torch.cat([means, scales, rots, opacities, colors,semantics], dim=-1) 
        
        self.anchor = nn.Parameter(anchor)
        self.instance_feature = nn.Parameter(torch.zeros([self.anchor.shape[0], self.embed_dims]))        
    
    def forward(self,bs):
        
        anchor = torch.tile(self.anchor,(bs,1,1))
        instance_feature = torch.tile(self.instance_feature,(bs,1,1))
        return anchor, instance_feature