from einops import einsum, rearrange

from mmdet.structures.bbox import scale_boxes
import torch, torch.nn as nn, torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from .gaussians import build_covariance
from ..utils.types import Gaussians

@MODELS.register_module()
class VolumeGaussianDecoderBEV(BaseModule):
    def __init__(
        self, 
        bev_h, bev_w, bev_z,
        pc_range, 
        gs_dim,
        gaussian_scale_min,
        gaussian_scale_max,         
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, 
        gpv=1, 
        offset_max=None, 
        use_checkpoint=False
    ):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z

        self.pc_range = pc_range
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.gpv = gpv       

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )
        self.gs_decoder = nn.Linear(out_dims, gs_dim*gpv*bev_z)
        self.use_checkpoint = use_checkpoint

        self.pos_act = lambda x: torch.tanh(x)
        if offset_max is None:
            self.offset_max = [1.0] * 3 # meters
        else:
            self.offset_max = offset_max

        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max


        self.scale_act = lambda x: torch.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)

        gs_anchors = self.get_reference_points(bev_h * scale_h, bev_w * scale_w, bev_z * scale_z, pc_range) # (1, w, h, z, 3)
        self.register_buffer('gs_anchors', gs_anchors) 
    
    @staticmethod
    def get_reference_points(H, W, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, -1).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, -1, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3) 
        ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) 
        return ref_3d

    def forward(self, bev_list):
        
        bev_hw = bev_list[0]
        bs, _, c = bev_hw.shape
        bev_hw = bev_hw.permute(0, 2, 1).reshape(bs, c, self.bev_h, self.bev_w) # (bs c h w)
        
        if self.scale_h != 1 or self.scale_w != 1:
            bev_hw = F.interpolate(
                bev_hw,
                size=(self.bev_h*self.scale_h, self.bev_w*self.scale_w),
                mode='bilinear'
            ) 
        
        bev_hw = bev_hw.permute(0, 1, 3, 2) # (bs c h w) -> (bs c w h)
        gaussians = bev_hw
        gaussians = gaussians.permute(0, 2, 3, 1) # (bs w h c)
        bs, h, w, _ = gaussians.shape
    
        if self.use_checkpoint:
            gaussians = torch.utils.checkpoint.checkpoint(self.decoder, gaussians, use_reentrant=False)
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, gaussians, use_reentrant=False)
        else:
            gaussians = self.decoder(gaussians) # (bs w h c)
            gaussians = self.gs_decoder(gaussians) # (bs, w h z*gpv*dim)
        
        
        gaussians = gaussians.view(bs, w, h, self.bev_z, self.gpv, -1) # (bs w h z*gpv*dim) -> (bs w h z gpv dim)
        
        # ----------------------------------------------------------------------------#
        # todo：解析高斯参数:
        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0]  # bs, w, h, z, gpv, 3
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1] # bs, w, h, z, gpv, 3
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[2] # bs, w, h, z, gpv, 3
        # 偏移量 + 体素网格锚点，得到每个高斯在世界坐标系中的最终中心
        means = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + self.gs_anchors[:, :, :, :, None, :] # (b w h z gpv 3) + (1 w h z gpv 3)
        # todo 解析具体高斯属性
        rgbs = self.rgb_act(gaussians[..., 3:6])  # sigmoid 颜色
        opacities = self.opacity_act(gaussians[..., 6:7]) # sigmoid，将输出限制在0-1之间 透明度
        
        rotations = self.rot_act(gaussians[..., 7:11]) # Formalize，确保四元数满足单位长度约束 旋转
        
        # scale_x = self.scale_act(x[..., 11:12]) * self.scale_max[0] # sigmoid，高斯体在x，y，z三个方向的尺度 尺寸
        # scale_y = self.scale_act(x[..., 12:13]) * self.scale_max[1]
        # scale_z = self.scale_act(x[..., 13:14]) * self.scale_max[2]
        scales = gaussians[...,11:14]
        scales = self.gaussian_scale_min + (self.gaussian_scale_max - self.gaussian_scale_min) * torch.sigmoid(scales)
        # 协方差
        covariances = build_covariance(scales, rotations)          
        # 语义特征
        semantics = F.softplus(gaussians[...,14:])
        
        gaussians = Gaussians(
            rearrange(means,"b w h z gpv xyz -> b (w h z gpv) xyz"), # (1 50 50 4 1 3) -> (1 10000 3)
            rearrange(scales,"b w h z gpv xyz -> b (w h z gpv) xyz"),  # (1 50 50 4 1 3) -> (1 10000 3) 
            rearrange(rotations,"b w h z gpv d -> b (w h z gpv) d"),  # (1 50 50 4 1 4) -> (1 10000 4)                          
            rearrange(covariances,"b w h z gpv i j -> b (w h z gpv) i j",), # (1 50 50 4 1 3 3) -> (1 10000 3 3) 
            rearrange(rgbs,"b w h z gpv c -> b (w h z gpv) c",),  # (1 50 50 4 1 3) -> (1 10000 3) 
            rearrange(opacities,   "b w h z gpv c -> b (w h z gpv c)"), # (1 50 50 4 1 1) -> (1 10000)
            rearrange(semantics,"b w h z gpv dim -> b (w h z gpv) dim")  # (1 50 50 4 1 18) -> (1 10000 18)     
        )
        return gaussians
