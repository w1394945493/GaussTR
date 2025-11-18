from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from einops import rearrange
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gausstr.models.utils import flatten_multi_scale_feats




from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


# from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
# from mmengine.registry import MODELS as MMENGIN_MODELS
# MMENGIN_MODELS.register_module('Det3DDataPreprocessor',module=Det3DDataPreprocessor)

@MODELS.register_module()
class GaussTRV2(BaseModel):

    def __init__(self,
                 neck,
                 decoder,
                 num_queries,
                 gauss_head,
                 backbone=None,
                 projection=None,
                 encoder=None,
                 pos_embed=None,
                 attn_type=None,
                 **kwargs):
        super().__init__(**kwargs)
        if backbone is not None:
            # todo ------------------#
            if backbone.type == 'TorchHubModel':
                # self.backbone = torch.hub.load(backbone.repo_or_dir, # todo 'facebookresearch/dinov2'
                #                             backbone.model_name)  # todo dinov2_vitb14_reg
                from dinov2.models.vision_transformer import vit_base
                self.backbone = vit_base(

                    img_size = 518,
                    patch_size = 14,
                    init_values = 1.0,
                    ffn_layer = "mlp",
                    block_chunks = 0,
                    num_register_tokens=4,
                    interpolate_antialias = False,
                    interpolate_offset = 0.1,
                    )
                model_url = '/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dinov2_vitb14_reg4_pretrain.pth'
                state_dict = torch.load(model_url, map_location="cpu")
                self.backbone.load_state_dict(state_dict, strict=True)

                print(cyan(f"load checkpoint from{model_url}."))
                self.backbone.requires_grad_(False) # todo 冻结主干backbone

                self.backbone.is_init = True  # otherwise it will be re-inited by mmengine
                self.patch_size = self.backbone.patch_size

                self.vit_type = 'vitb'
                self.intermediate_layer_idx = {
                    "vits": [2, 5, 8, 11],
                    "vitb": [2, 5, 8, 11],
                    "vitl": [4, 11, 17, 23],
                }

            else:
                self.backbone = MODELS.build(backbone)
            self.frozen_backbone = all(not param.requires_grad
                                       for param in self.backbone.parameters())


            if attn_type is not None:
                assert backbone.out_indices == -2
            self.attn_type = attn_type

        if projection is not None:
            self.projection = MODELS.build(projection)
            if 'init_cfg' in projection and projection.init_cfg.type == 'Pretrained':
                self.projection.requires_grad_(False)
        self.neck = MODELS.build(neck)

        if encoder is not None:
            self.encoder = MODELS.build(encoder)
            self.pos_embed = MODELS.build(pos_embed)
            attn_cfg = encoder.layer_cfg.self_attn_cfg
            self.level_embed = nn.Parameter(
                torch.Tensor(attn_cfg.num_levels, attn_cfg.embed_dims))
        self.decoder = MODELS.build(decoder)

        self.query_embeds = nn.Embedding(
            num_queries, decoder.layer_cfg.self_attn_cfg.embed_dims)
        self.gauss_heads = ModuleList(
            [MODELS.build(gauss_head) for _ in range(decoder.num_layers)]) # todo num_layers:3

    def prepare_inputs(self, inputs_dict, data_samples):
        num_views = data_samples[0].num_views
        inputs = inputs_dict['imgs']

        cam2img = []
        cam2ego = []
        ego2global = []
        img_aug_mat = []
        depth = []
        feats = []
        sem_segs = []

        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})
            cam2img.append(data_samples[i].cam2img)
            data_samples[i].set_metainfo(
                {'cam2ego': data_samples[i].cam2ego[:num_views]})
            cam2ego.append(data_samples[i].cam2ego)
            ego2global.append(data_samples[i].ego2global)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat[:num_views]})
                img_aug_mat.append(data_samples[i].img_aug_mat)
            # todo depth
            depth.append(data_samples[i].depth)


            if hasattr(data_samples[i], 'feats'): # todo 特征图
                feats.append(data_samples[i].feats)
            if hasattr(data_samples[i], 'sem_seg'):
                sem_segs.append(data_samples[i].sem_seg) # todo 分割图

        data_samples = dict(
            depth=depth,
            cam2img=cam2img,
            cam2ego=cam2ego,
            num_views=num_views,
            ego2global=ego2global,
            img_aug_mat=img_aug_mat if img_aug_mat else None)
        if feats:
            data_samples['feats'] = feats
        if sem_segs:
            data_samples['sem_segs'] = sem_segs
        for k, v in data_samples.items():
            if isinstance(v, torch.Tensor) or not isinstance(v, Iterable):
                continue
            if isinstance(v[0], torch.Tensor):
                data_samples[k] = torch.stack(v).to(inputs)
            else:
                data_samples[k] = torch.from_numpy(np.stack(v)).to(inputs)
        return inputs, data_samples

    def forward(self, inputs, data_samples, mode='loss'):
        inputs, data_samples = self.prepare_inputs(inputs, data_samples)
        # bs, n = inputs.shape[:2]
        bs, n, _, ori_h, ori_w = inputs.shape # (b,v,3,H,W)

        # todo ---------------------------------#
        # todo 使用预训练的VFM(DINOv2)进行多视图特征提取
        if hasattr(self, 'backbone'):
            # 原代码：必须确保 input_size 为14的倍数
            # inputs = inputs.flatten(0, 1) # (b,v,3,h,w) -> ((b v) 3 h w)

            # todo: 对原图缩放一下，
            resize_h, resize_w = ori_h // self.patch_size * self.patch_size, ori_w // self.patch_size * self.patch_size
            inputs = rearrange(inputs,"b v c h w -> (b v) c h w")
            inputs = F.interpolate(inputs,(resize_h,resize_w),mode='bilinear',align_corners=True)

            if self.frozen_backbone:
                if self.backbone.training:
                    self.backbone.eval()
                with torch.no_grad(): # todo 提取单目视觉特征
                    if isinstance(self.backbone, BaseModule):
                        x = self.backbone(inputs)[0]
                        if self.attn_type is not None:
                            x = self.custom_attn(x, self.attn_type)
                    else:

                        x = self.backbone.forward_features(
                            inputs)['x_norm_patchtokens'] # todo ((b v) n 768) n = HxW / (patch_size x patch_size)

                        # 原代码：必须确保 input_size 为14的倍数
                        # x = x.mT.reshape(bs * n, -1,
                        #                  inputs.shape[-2] // self.patch_size,
                        #                  inputs.shape[-1] // self.patch_size) # 转置 -> 重新整合为特征图
                        x = x.mT.reshape(bs * n, -1,
                                         resize_h // self.patch_size,
                                         resize_w // self.patch_size) # 转置 -> 重新整合为特征图

            else:
                x = self.backbone(inputs)[0]
        else:
            x = data_samples['feats'].flatten(0, 1)

        # todo 区分 model和model.head的projection
        if hasattr(self, 'projection'):
            x = self.projection(x.permute(0, 2, 3, 1))[0]
            x = x.permute(0, 3, 1, 2)
        if hasattr(self, 'backbone') or hasattr(self, 'projection'):
            data_samples['feats'] = x.reshape(bs, n, *x.shape[1:]) # todo ((b v) c h w) -> (b v c h w)
        if n > data_samples['num_views']:
            x = x.reshape(bs, n, *x.shape[1:])
            x = x[:, :data_samples['num_views']].flatten(0, 1)

        # todo ---------------------------------#
        feats = self.neck(x) # todo ViTDetFPN: 多尺度特征图提取






        # 注：网络输入 (3 252 484) -> backbone -> (768 18 32) -> neck -> (256 72 128) (256 36 64) (256 18 32) (256 9 16)
        if hasattr(self, 'encoder'):
            encoder_inputs, decoder_inputs = self.pre_transformer(feats)
            feats = self.forward_encoder(**encoder_inputs)
        else:
            decoder_inputs = self.pre_transformer(feats) # 处理多尺度特征: 做多尺度位置编码等预处理工作，记录各层特征图的索引
            feats = flatten_multi_scale_feats(feats)[0] # 展平后的多尺度特征 (256 72 128) (256 36 64) (256 18 32) (256 9 16) -> (12240 256)

        # todo ---------------------------------#
        # todo Decoder:
        decoder_inputs.update(self.pre_decoder(feats)) # todo 准备解码器的输入：查询特征、参考点(随机生成0-1之间的张量 # 三维
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads], # todo 取 gauss_heads中的regress_head
            **decoder_inputs) # todo 解码

        query = decoder_outputs['hidden_states'] # todo 各解码层的query和参考点坐标
        reference_points = decoder_outputs['references']

        # todo ---------------------------------#
        # todo 推理
        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1], mode=mode, **data_samples)

        # todo ---------------------------------#
        # todo 训练：损失计算
        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads): # 多层
            # todo 对各层预测得到的查询进行高斯属性预测: 对query解码得到其余高斯属性,
            loss = gauss_head(
                query[i], reference_points[i], gt_imgs = inputs,mode=mode, **data_samples)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        return losses

    def custom_attn(self, x, attn_type):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).mT
        last_layer = self.backbone.layers[-1]
        qkv = last_layer.attn.qkv(last_layer.ln1(x)).reshape(
            B, N, 3, last_layer.attn.num_heads,
            last_layer.attn.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if attn_type == 'maskclip':
            v = last_layer.attn.proj(v.transpose(1, 2).flatten(2)) + x
            v = last_layer.ffn(last_layer.ln2(v), identity=v)
            if self.backbone.final_norm:
                x = self.backbone.ln1(v)
        elif attn_type == 'clearclip':
            x = last_layer.attn.scaled_dot_product_attention(q, q, v)
            x = x.transpose(1, 2).reshape(B, N, last_layer.attn.embed_dims)
            x = last_layer.attn.proj(x)
            if last_layer.attn.v_shortcut:
                x = v.squeeze(1) + x
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def pre_transformer(self, mlvl_feats):
        batch_size = mlvl_feats[0].size(0)

        mlvl_masks = []
        for feat in mlvl_feats:
            mlvl_masks.append(None)

        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(zip(mlvl_feats, mlvl_masks)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        if not hasattr(self, 'encoder'):
            return decoder_inputs_dict

        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_pos_embeds.append(self.pos_embed(None, input=feat))

        lvl_pos_embed_flatten = []
        for lvl, (feat, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        # (bs, num_feat_points, dim)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        encoder_inputs_dict = dict(
            feat=feat_flatten, # todo 展平后的特征
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten, # todo 经位置编码后的特征
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat, feat_mask, feat_pos, spatial_shapes,
                        level_start_index, valid_ratios):
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return memory

    def pre_decoder(self, memory):
        bs, _, c = memory.shape
        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1)
        reference_points = torch.rand((bs, query.size(1), 2)).to(query) # todo 生成随机的参考点 (二维)
        # reference_points = torch.rand((bs, query.size(1), 3)).to(query) # todo 生成三维参考点

        decoder_inputs_dict = dict(
            query=query, memory=memory, reference_points=reference_points)
        return decoder_inputs_dict

    def forward_decoder(self, query, memory, memory_mask, reference_points,
                        spatial_shapes, level_start_index, valid_ratios,
                        **kwargs):
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
