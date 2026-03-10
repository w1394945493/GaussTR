import copy
import warnings

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (build_attention,
                                         build_feedforward_network)
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS


@MODELS.register_module()
class BEVFormerLayer(BaseModule):
    """Base `TPVFormerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """
    def __init__(self,
                 attn_cfgs=None, # 注意力模块配置,若传入列表,配置顺序必须与operation_order中的注意力顺序一致,如果是字典,所有注意力模块使用相同配置
                 ffn_cfgs=dict(
                     type='FFN',
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ), # ffn模块配置.
                 operation_order=None, # 执行流顺序,定义了这一层内部数据流转的具体步骤              
                 norm_cfg=dict(type='LN'), # 归一化层配置,默认Layer Norm
                 init_cfg=None, # 初始化配置,用于定义权重初始化的方式
                 batch_first=True, # 维度顺序.True -> (bs,n,dim) False -> (n bs dim)
                 **kwargs,
                 ):
        # 参数处理, 向下兼容
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ')
                ffn_cfgs[new_name] = kwargs[ori_name]
        
        super().__init__(init_cfg)

        self.batch_first = batch_first

        # 确保注意力机制配置与定义的执行顺序在数量上完全匹配
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'    
        
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1
        self.embed_dims = self.attentions[0].embed_dims       
        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                bev_h=None,
                bev_w=None,
                reference_points_cams=None,
                bev_masks=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        if self.operation_order[0] == 'cross_attn':
            query = torch.cat(query, dim=1)
        identity = query       
        for layer in self.operation_order:
            if layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    reference_points_cams=reference_points_cams,
                    bev_masks=bev_masks,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query                
            
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        query = torch.split(
            query, [bev_h * bev_w], dim=1)
        return query