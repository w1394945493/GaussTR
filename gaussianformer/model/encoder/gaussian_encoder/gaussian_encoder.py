from typing import List, Optional
import torch, torch.nn as nn

from mmdet3d.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder # todo 4

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order # todo ['deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm',| 'deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm',| 'deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm',| 'deformable', 'ffn', 'norm', 'refine']

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        instance_feature = rep_features # todo 查询特征
        anchor = representation # todo (b,25600,28) 初始的高斯属性，是随机化的值
        # todo 将28维的属性拆分，得到 均值、尺度、旋转、语义...并进行编码
        anchor_embed = self.anchor_encoder(anchor) # todo 锚框编码: 这里锚框定义更为广泛: 包括 高斯属性(3+3+4+1)+语义属性(OCC中是17类)

        prediction = []
        for i, op in enumerate(self.operation_order): # todo ['deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm'] x N
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                # todo 图像交叉注意力模块：用于聚合视觉信息
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor, # todo 初始的28维的锚框
                    anchor_embed, # todo 编码后的锚框: 128维
                    feature_maps, # todo 多尺度特征图
                    metas, # todo metas: 元数据：
                )
            elif "refine" in op: # todo 细化模块： 均值为旧均值+预测结果，其他属性则直接替换
                # todo 细化模块：对高斯属性进行细化：使用MLP从高斯查询中解码得到中间属性，将中间属性均值作为残差加到旧均值上，其他属性则直接替换对应的旧属性
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                ) # todo 得到预测的高斯属性

                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation": prediction}