import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Scale
from mmdet3d.registry import MODELS

from .utils import linear_relu_ln
from ...utils.types import Gaussians
from ..common.gaussians import build_covariance

@MODELS.register_module()
class SparseGaussian3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims,
        output_dim,
        pc_range,
        voxel_size,
        scale_range,
        semantic_dim,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        
        self.scale_range = scale_range
        self.semantic_dim = semantic_dim
                
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))
           
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        eps: float = 1e-8,
    ):  
        output = self.layers(instance_feature + anchor_embed) # todo (1 25600 32)
        offset_xyz, scales, rotations, opacities, colors, semantics = output.split([3,3,4,1,3,self.semantic_dim],dim=-1)
        offset_world = (offset_xyz.sigmoid() - 0.5) * self.voxel_size * 3 
        
        means = anchor[...,:3] + offset_world # todo (1 25600 3)
        anchor = torch.cat([means,scales,rotations,opacities,colors,semantics],dim=-1) # todo (1 25600 32)
        
        #?----------------------------------------------?
        #? 高斯参数解码
        scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * torch.sigmoid(scales) # todo (1 25600 3)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps) # todo (1 25600 4)
        opacities = opacities.sigmoid() # todo (1 25600 1)
        colors = colors.sigmoid() # todo (1 25600 3)
        covariances = build_covariance(scales, rotations) # todo (1 25600 3 3)
        semantics = F.softplus(semantics) # todo (1 25600 18)
        gaussians = Gaussians(
            means,
            scales,
            rotations,
            covariances,
            colors,
            opacities.squeeze(-1),
            semantics,)
        
        return anchor, gaussians
            
    
    