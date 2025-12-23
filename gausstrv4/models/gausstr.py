from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
from torch.cuda.amp import autocast
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS
from .utils import flatten_multi_scale_feats

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


# from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
# from mmengine.registry import MODELS as MMENGIN_MODELS
# MMENGIN_MODELS.register_module('Det3DDataPreprocessor',module=Det3DDataPreprocessor)

@MODELS.register_module()
class GaussTR(BaseModel):

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
                self.backbone.requires_grad_(False)
                self.backbone.is_init = True  # otherwise it will be re-inited by mmengine
                self.patch_size = self.backbone.patch_size
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
            [MODELS.build(gauss_head) for _ in range(decoder.num_layers)])

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

        occ_gts = [] # occ标注

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


            if hasattr(data_samples[i], 'feats'): # FeatUp有"feats"和"sem_seg"内容
                feats.append(data_samples[i].feats)
            if hasattr(data_samples[i], 'sem_seg'):
                sem_segs.append(data_samples[i].sem_seg)
            if hasattr(data_samples[i].gt_pts_seg, 'semantic_seg'):
                occ_gts.append(data_samples[i].gt_pts_seg.semantic_seg) # todo occ占用图

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
        if occ_gts:
            data_samples['occ_gts'] = torch.cat(occ_gts)

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
        bs, n = inputs.shape[:2]

        # todo ---------------------------------#
        # todo 使用预训练的VIT进行多视图特征提取
        if hasattr(self, 'backbone'):
            inputs = inputs.flatten(0, 1) # (b,v,3,h,w) -> ((b v) 3 h w)
            if self.frozen_backbone:
                if self.backbone.training:
                    self.backbone.eval()
                with torch.no_grad():
                    if isinstance(self.backbone, BaseModule):
                        x = self.backbone(inputs)[0]
                        if self.attn_type is not None:
                            x = self.custom_attn(x, self.attn_type)
                    else:
                        x = self.backbone.forward_features(
                            inputs)['x_norm_patchtokens']
                        x = x.mT.reshape(bs * n, -1,
                                         inputs.shape[-2] // self.patch_size,
                                         inputs.shape[-1] // self.patch_size) # 转置 -> 重新整合为特征图
            else:
                x = self.backbone(inputs)[0]
        else:
            x = data_samples['feats'].flatten(0, 1)

        if hasattr(self, 'projection'):
            x = self.projection(x.permute(0, 2, 3, 1))[0]
            x = x.permute(0, 3, 1, 2)
        if hasattr(self, 'backbone') or hasattr(self, 'projection'):
            data_samples['feats'] = x.reshape(bs, n, *x.shape[1:]) # ((b v) c h w) -> (b v c h w)
        if n > data_samples['num_views']:
            x = x.reshape(bs, n, *x.shape[1:])
            x = x[:, :data_samples['num_views']].flatten(0, 1) # todo 这里展平了!

        feats = self.neck(x) # list # todo x: ((b v) 3 36 64) 36=504/14 64=896/14 -> list ((b v) 256 144 256) (72 128) (36 64) (18 32)


        if hasattr(self, 'encoder'):
            encoder_inputs, decoder_inputs = self.pre_transformer(feats)
            feats = self.forward_encoder(**encoder_inputs)
        else:
            decoder_inputs = self.pre_transformer(feats) # todo 处理多尺度特征，整理为Transformer编解码器所需格式
            feats = flatten_multi_scale_feats(feats)[0]
        decoder_inputs.update(self.pre_decoder(feats))
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs)

        query = decoder_outputs['hidden_states'] # todo 各解码层的query和参考点坐标
        reference_points = decoder_outputs['references']

        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1], mode=mode, **data_samples)

        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads): # 多层
            loss = gauss_head(
                query[i], reference_points[i], mode=mode, **data_samples)
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
        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1) # todo 可学习的特征嵌入
        # todo 初始参考点： 归一化到
        reference_points = (torch.rand((bs, query.size(1), 2))*0.5 + 0.25).to(query) # todo 随机的0-1之间的参考点
        # reference_points = torch.rand((bs, query.size(1), 2)).to(query)


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






    # def extract_img_feat(self, img, status="train"):
    #     """Extract features of images."""
    #     B, N, C, H, W = img.size()
    #     img = img.view(B * N, C, H, W)
    #     # if self.use_checkpoint and status != "test":
    #     #     img_feats = torch.utils.checkpoint.checkpoint(
    #     #                     self.backbone, img, use_reentrant=False)
    #     # else:
    #     #     img_feats = self.backbone(img)
    #     img_feats = self.backbone(img)
    #     img_feats = self.neck(img_feats) # BV, C, H, W
    #     img_feats_reshaped = []
    #     for img_feat in img_feats:
    #         _, C, H, W = img_feat.size()
    #         img_feats_reshaped.append(img_feat.view(B, N, C, H, W)) # todo 这里由(bv c h w) -> (b v c h w)


    #     return img_feats_reshaped

