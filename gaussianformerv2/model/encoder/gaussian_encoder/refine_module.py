from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class SparseGaussian3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        restrict_xyz=False,
        unit_xyz=None,
        refine_manual=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        semantics_activation='softmax',
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0

        self.output_dim = 10 + int(include_opa) + semantic_dim
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        self.unit_xyz = unit_xyz
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            if xyz_activation == "sigmoid":
                unit_prob = [4 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_prob

        assert isinstance(refine_manual, list)
        self.refine_state = refine_manual
        assert all([self.refine_state[i] == i for i in range(len(self.refine_state))])

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
        # todo 3.3 节 细化模块：通过细化模块，对高斯属性进行细化
        # todo 使用多层感知机从高斯查询Q中解码得到中间属性
        output = self.layers(instance_feature + anchor_embed) # todo (1 25600 128) + (1 25600 128) -> (1 25600 28) 高斯属性

        if self.restrict_xyz:
            delta_xyz_sigmoid = output[..., :3]
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1 # todo 归一化到-1 - 1之间的偏移量
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
            output = torch.cat([delta_xyz, output[..., 3:]], dim=-1) # todo 将预测得到的xyz拼接到output
        # todo -----------------------#
        # todo 使用中间属性细化原属性：中间均值作为残差加到旧均值熵，其他的中间属性直接替换对应的旧属性
        # todo 通过残差连接细化高斯均值，以保存均值的一致性。直接替换其他属性：协方差和语义logit上施加sigmoid和softmax激活函数可能导致梯度消失问题
        if len(self.refine_state) > 0: # todo 将预测均值和旧均值相加
            refined_part_output = output[..., self.refine_state] + anchor[..., self.refine_state] 
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)
        
        # todo -----------------------#
        # todo 均值
        if self.xyz_act == "sigmoid": # todo xyz_act: sigmoid
            xyz = output[..., :3]
        else:
            xyz = output[..., :3].clamp(min=1e-6, max=1-1e-6)
        
        # todo -----------------------#
        # todo 尺度
        if self.scale_act == "sigmoid":
            scale = output[..., 3:6]
        else:
            scale = output[..., 3:6].clamp(min=1e-6, max=1-1e-6)
        
        # todo -----------------------#
        # todo 旋转
        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1) # todo 四元数：归一化到0-1
        
        # todo --------------------------#
        # todo output: 网络预测的原始属性值，即还没有映射到真实世界尺度的属性值 [ Δxyz | scale_raw | rot_raw | opacity_raw | semantic_raw ]
        output = torch.cat([xyz, scale, rot, output[..., 10:]], dim=-1) 

        # todo --------------------------#
        # todo 解码得到 真实世界尺度下的高斯 [ xyz_world | scale_world | rot_unit | opacity_prob | semantic_prob ]
        if self.xyz_act == 'sigmoid':
            xyz = safe_sigmoid(xyz) # todo 预测得到 0 - 1 之间的均值(位置)
        # todo 转换为体素空间下的位置
        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0] # todo -50 -> 50, -50 -> 50, -50 -> 50, -5.0 -> 3.0
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        # todo GaussianFormer中，尺度预测与MonoSplat中一致，通过sigmoid归一化
        if self.scale_act == 'sigmoid': # todo sigmoid
            gs_scales = safe_sigmoid(scale) # todo 将scale限制到-9.21 9.21 然后进行sigmoid归一化 gs_scales: (b 25600 3)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales # todo self.scale_range: 0.08，0.64

        semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)] # todo (b 25600 17)
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus': # todo softplus
            # todo 语义预测：使用的是softplus
            semantics = F.softplus(semantics) # todo log(1+e^x) semantics: (b 25600 17) softplus: 逐元素激活，将 任意实数 映射到 严格实数

        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]), # (b 25600 1) # todo include_opa: True
            semantics=semantics
        )
        return output, gaussian #, semantics


