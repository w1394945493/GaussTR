import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from .ops import LocalAggregator
from .utils.positional_encoding import NerfPositionalEncoder

@MODELS.register_module()
class StreamOccHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 memory_len,
                 topk_proposals=500,
                 num_propagated=500,
                 prop_query=False,
                 temp_fusion=False,
                 with_ego_pos=False,
                 transformer=None,
                 empty_label=17,
                 ignore_label=255,
                 pc_range=[],
                 voxel_size=[],
                 scale_range=[0.01, 3.2],
                 u_range=[0.1, 2],
                 v_range=[0.1, 2],
                 nusc_class_frequencies=[],
                 manual_class_weight=None,
                 score_thres=None,
                 loss_occ=None,
                 loss_pts=None,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.prop_query = prop_query
        self.temp_fusion = temp_fusion
        self.with_ego_pos = with_ego_pos
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.transformer = MODELS.build(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.score_thres = score_thres

        self.scale_range = scale_range
        self.u_range = u_range
        self.v_range = v_range
        self.ignore_label = ignore_label

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()

        self.aggregator = LocalAggregator(
            scale_multiplier=3,
            H=voxel_num[0],
            W=voxel_num[1],
            D=voxel_num[2],
            pc_min=pc_range[:3],
            grid_size=voxel_size[0],
        )
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)
        xyz = self.get_meshgrid(pc_range, voxel_num, voxel_size)
        self.register_buffer('gt_xyz', torch.tensor(xyz))

        self._init_layers()

        if manual_class_weight is not None:
            self.class_weights = torch.tensor(manual_class_weight, dtype=torch.float)
            self.cls_weights = (num_classes + 1) * F.normalize(self.class_weights, 1, -1)
        else:
            class_freqs = nusc_class_frequencies
            self.cls_weights = torch.from_numpy(1 / np.log(np.array(class_freqs[:num_classes+1]) + 0.001))

        loss_occ['class_weight'] = self.cls_weights
        loss_occ['ignore_label'] = self.ignore_label
        self.loss_occ = MODELS.build(loss_occ)
        self.loss_pts = MODELS.build(loss_pts)
        self.reset_memory()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1) # todo 随机填充为0-1之间的数：uniform_: 就地操作，直接修改数值

        # encoding ego pose
        if self.with_ego_pos:
            self.nerf_encoder = NerfPositionalEncoder(num_encoding_functions=6)
            self.ego_pose_memory = MLN(156)

    
    def init_weights(self):
        self.transformer.init_weights()
        
    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso[0] + 0.5 * reso[0] + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso[1] + 0.5 * reso[1] + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso[2] + 0.5 * reso[2] + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()  # （Dx， Dy, Dz, 3)
        return xyz
    
    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        

class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c): # todo 进行特征调制：使用外部的条件信号来动态控制主特征表现
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out