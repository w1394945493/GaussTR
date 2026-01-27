import torch
import torch.nn as nn

from mmcv.cnn import Scale
from mmdet3d.registry import MODELS

from .utils import linear_relu_ln

@MODELS.register_module()
class SparseGaussian3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        output_dim=32,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))
           
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
    ):
        output = self.layers(instance_feature + anchor_embed)
        
            
        return anchor   
            
    
    