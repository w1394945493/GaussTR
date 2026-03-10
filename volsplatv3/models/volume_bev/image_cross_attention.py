
import math
import warnings

import torch
import torch.nn as nn
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmengine.model import BaseModule, constant_init, xavier_init

from mmengine.registry import MODELS
from itertools import chain

@MODELS.register_module()
class BEVImageCrossAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=True,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 bev_h=None,
                 bev_w=None,):
        super().__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = MODELS.build(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.bev_h, self.bev_w = bev_h, bev_w
        self.init_weight()
    
    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                spatial_shapes=None,
                reference_points_cams=None,
                bev_masks=None,
                level_start_index=None):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        bs, _, _ = query.size()
        queries = torch.split(
            query, [
                self.bev_h * self.bev_w
            ],
            dim=1)
        if residual is None:
            slots = [torch.zeros_like(q) for q in queries]

        indexeses = []
        max_lens = []
        queries_rebatches = []
        reference_points_rebatches = []
        
        for bev_idx, bev_mask in enumerate(bev_masks):
            indexes = []
            for j_ in range(bs):
                indexes_per_batch = []
                for i_, mask_per_img in enumerate(bev_mask):
                    index_query_per_img = mask_per_img[j_].sum(
                        -1).nonzero().squeeze(-1) # 寻找有效query 
                    indexes_per_batch.append(index_query_per_img)
                indexes.append(indexes_per_batch)
            indexeses.append(indexes)
            
            max_len = max([len(each) for each in chain.from_iterable(indexes)]) # 计算该批次中最多的有效点数
            queries_rebatch = queries[bev_idx].new_zeros(
                [bs, self.num_cams, max_len, self.embed_dims])
            reference_points_cam = reference_points_cams[bev_idx] # (6 1 2500 8 2)
            D = reference_points_cam.size(3)
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs, self.num_cams, max_len, D, 2])

            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[j][i]
                    queries_rebatch[j, i, :len(index_query_per_img)] = queries[bev_idx][
                        j, index_query_per_img
                    ]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                        j, index_query_per_img
                    ]
            queries_rebatches.append(queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims))
            reference_points_rebatches.append(reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2))
        
        num_cams, l, bs, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).contiguous().view(self.num_cams * bs, l,
                                           self.embed_dims)
        value = value.permute(2, 0, 1, 3).contiguous().view(self.num_cams * bs, l,
                                               self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatches, # (6 605 128)
            key=key, # (6 22400 128)
            value=value,
            reference_points=reference_points_rebatches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        queries = [query.view(bs, self.num_cams, -1, self.embed_dims) for query in queries]

        for bev_idx, indexes in enumerate(indexeses):
            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes[j]):
                    slots[bev_idx][j, index_query_per_img] += queries[bev_idx][
                        j, i, :len(index_query_per_img)]
                count = bev_masks[bev_idx][:, j].sum(-1) > 0 # 6, 10000
                count = count.permute(1, 0).sum(-1) # 10000
                count = torch.clamp(count, min=1.0)
                slots[bev_idx][j] = slots[bev_idx][j] / count[:, None] # 10000, 128

        slots = torch.cat(slots, dim=1)
        slots = self.output_proj(slots) # (1 2500 128)

        return self.dropout(slots) + inp_residual

@MODELS.register_module()
class BEVMSDeformableAttention3D(BaseModule):
    """An attention module used in bevFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.

    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """
    def __init__(
        self,
        embed_dims=256,
        num_heads=8, # 注意力头数
        num_levels=4,
        num_points=[8, 64, 64],
        num_z_anchors=[4, 32, 32],
        pc_range=None,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
        floor_sampling_offset=True,
        bev_h=None,
        bev_w=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
        
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.base_num_points = num_points[0]
        self.base_z_anchors = num_z_anchors[0]
        self.points_multiplier = [
            points // self.base_z_anchors for points in num_z_anchors
        ]
        self.pc_range = pc_range
        self.bev_h, self.bev_w = bev_h, bev_w
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2)
            # for i in range(3)
            for i in range(1)
        ]) 
        
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i])
            # for i in range(3)
            for i in range(1)
        ])
        self.value_proj = nn.Linear(embed_dims, embed_dims)

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        device = next(self.parameters()).device
        # for i in range(3):
        for i in range(1):
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32,
                device=device) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                         grid_init.abs().max(-1, keepdim=True)[0]).view(
                             self.num_heads, 1, 1,
                             2).repeat(1, self.num_levels, self.num_points[i],
                                       1)
            grid_init = grid_init.reshape(self.num_heads, self.num_levels,
                                          self.num_z_anchors[i], -1, 2)
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1

            self.sampling_offsets[i].bias.data = grid_init.view(-1)
            constant_init(self.attention_weights[i], val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(
                zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels,
                                       self.points_multiplier[i], -1, 2)
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)

            attention = attn(query).reshape(bs, l, self.num_heads, -1)
            attention = attention.softmax(-1)
            attention = attention.view(bs, l, self.num_heads, self.num_levels,
                                       self.points_multiplier[i], -1)
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)

        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns


    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        for i, reference_point in enumerate(reference_points):
            bs, l, z_anchors, _ = reference_point.shape
            reference_point = reference_point.reshape(
                bs, l, self.points_multiplier[i], -1, 2)
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
        return torch.cat(reference_point_list, dim=1)

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(
            output, [
                lens[0] * self.points_multiplier[0]
            ],
            dim=1)
        outputs = [
            o.reshape(bs, -1, self.points_multiplier[i], d).sum(dim=2)
            for i, o in enumerate(outputs)
        ]
        return outputs

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query        

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)
        
        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1) # (6 22400 8 16)

        sampling_offsets, attention_weights = \
            self.get_sampling_offsets_and_attention(query) # (6 605 8 1 16 2 ) (6 605 8 1 16)

        reference_points = self.reshape_reference_points(reference_points) # (6 605 8 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = \
                sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors,
                num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, \
                xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors # 16 = 8 * 2

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy) # (6 605 8 1 16 2)

            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(
                    sampling_locations)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step) # (6 605 128)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]

        return output