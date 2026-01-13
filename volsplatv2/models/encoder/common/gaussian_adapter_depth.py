from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .gaussians import build_covariance
from .me_fea import project_features_to_me
from typing import Tuple, Optional

from ...utils.types import Gaussians

@MODELS.register_module()
class GaussianAdapter_depth(nn.Module):

    def __init__(self,
                 gaussian_scale_min = 1e-10,
                 gaussian_scale_max = 3.0,
                 sh_degree=2,
                 num_class = 18,
                 ):
        super().__init__()
        
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        
        self.sh_degree = sh_degree
        if self.sh_degree:
            self.d_sh = (self.sh_degree + 1) ** 2
            # Create a mask for the spherical harmonics coefficients. This ensures that at
            # initialization, the coefficients are biased towards having a large DC
            # component and small view-dependent components.
            self.register_buffer(
                "sh_mask",
                torch.ones((self.d_sh,), dtype=torch.float32), # todo d_sh: 9
                persistent=False,
            )
            for degree in range(1, self.sh_degree + 1):
                self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        else:
            self.d_sh = None
        
        self.num_class = num_class
    
    def forward(
        self,
        extrinsics: Tensor,
        opacities: Tensor,
        raw_gaussians: Tensor, #[1, 1, N, 37]
        points: Optional[Tensor] = None,
        voxel_resolution: float = 0.01,
        eps: float = 1e-8,
    ) :
    #-> Gaussians

        batch_dims = extrinsics.shape[:-2]

        b, v = batch_dims
        
        offset_xyz, scales, rotations, sh, semantics = raw_gaussians.split((3, 3, 4, 3 * self.d_sh 
                                                                 if self.d_sh is not None else 3, self.num_class), 
                                                                dim=-1) #[1, 1, N, 1, 1, c]
        
        # scales = torch.clamp(F.softplus(scales - 4.),
        #     min=self.gaussian_scale_min,
        #     max=self.gaussian_scale_max,
        #     )
        scales = self.gaussian_scale_min + (self.gaussian_scale_min - self.gaussian_scale_min) * torch.sigmoid(scales)


        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        
        if self.d_sh:
            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)    # [1, 1, 256000, 1, 1, 3, 9]
            sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        else:
            sh = torch.sigmoid(sh)
        
        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)  #[1, 1, 256000, 1, 1, 3, 3]

        xyz = points
        if xyz.ndim == 2:
            xyz = rearrange(xyz, "n c -> 1 1 n () () c")
        elif xyz.ndim == 4:
            xyz = rearrange(xyz, "b v n c -> b v n () () c")

        offset_xyz = offset_xyz.sigmoid()
        offset_world = (offset_xyz - 0.5) *voxel_resolution*3  # [1,1,N,1,1, 3]

        means = xyz + offset_world  # [1,1,N, 1,1,3]

        gaussians = Gaussians(rearrange(means,"b v r srf spp xyz -> b (v r srf spp) xyz"), # [b, 1, 256000, 1, 1, 3] -> [b, 256000, 3]
            rearrange(scales,"b v r srf spp xyz -> b (v r srf spp) xyz"), # [b, 1, 256000, 1, 1, 3] -> [b, 256000, 3]
            rearrange(rotations,"b v r srf spp d -> b (v r srf spp) d"), # [b, 1, 256000, 1, 1, 4] -> [b, 256000, 4]                             
            rearrange(covariances,"b v r srf spp i j -> b (v r srf spp) i j",), # [2, 1, 256000, 1, 1, 3, 3] -> [2, 256000, 3, 3]
            rearrange(sh,"b v r srf spp c d_sh -> b (v r srf spp) c d_sh",) \
                if self.d_sh is not None else rearrange(sh,"b v r srf spp rgb -> b (v r srf spp) rgb",),
            rearrange(opacities,   "b v r srf spp -> b (v r srf spp)"), #[2, 1, 256000, 1, 1] -> [2, 256000]
            rearrange(semantics,"b v r srf spp dim -> b (v r srf spp) dim")       
        ) 
        return gaussians        
        

