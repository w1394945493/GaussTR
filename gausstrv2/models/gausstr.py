from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange,repeat
import torch.nn.functional as F

from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from .utils import flatten_multi_scale_feats

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"



@MODELS.register_module()
class GaussTR(BaseModel):

    def __init__(self,
                 neck,
                 decoder,
                 gauss_head,
                 num_queries,
                 model_url,
                 ori_image_shape,
                 patch_size = 14,
                 **kwargs):
        super().__init__(**kwargs)

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
        state_dict = torch.load(model_url, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        print(cyan(f"load checkpoint from{model_url}."))
        self.backbone.requires_grad_(False)
        self.backbone.is_init = True

        self.neck = MODELS.build(neck)
        self.decoder = MODELS.build(decoder)
        self.query_embeds = nn.Embedding(
            num_queries, decoder.layer_cfg.self_attn_cfg.embed_dims)
        self.gauss_heads = ModuleList(
            [MODELS.build(gauss_head) for _ in range(decoder.num_layers)])



        self.patch_size = patch_size
        self.ori_image_shape = ori_image_shape

        print(cyan(f'successfully init Model!'))




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

        rgb_gts = [] # 真实图像
        occ_gts = [] # occ标注

        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})
            cam2img.append(data_samples[i].cam2img)

            # # normalize the standred format into intrinsics
            # ori_h, ori_w = self.ori_image_shape # (900, 1600)
            # intrinsics = data_samples[i].cam2img
            # intrinsics[:, 0, 0] /= ori_w
            # intrinsics[:, 1, 1] /= ori_h
            # intrinsics[:, 0, 2] /= ori_w
            # intrinsics[:, 1, 2] /= ori_h
            # cam2img.append(intrinsics)
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
            # todo rgb_gts
            rgb_gts.append(data_samples[i].img)
            if hasattr(data_samples[i], 'feats'): # todo 特征图
                feats.append(data_samples[i].feats)
            if hasattr(data_samples[i], 'sem_seg'):
                sem_segs.append(data_samples[i].sem_seg) # todo 分割图
            if hasattr(data_samples[i].gt_pts_seg, 'semantic_seg'):
                occ_gts.append(data_samples[i].gt_pts_seg.semantic_seg) # todo occ占用图

        data_samples = dict(
            rgb_gts = rgb_gts,
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
        bs, n, _, h, w = inputs.shape # (b,v,3,H,W)
        # device = inputs.device

        # 将图像缩放为14的倍数
        concat = rearrange(inputs,"b v c h w -> (b v) c h w")
        resize_h, resize_w = h // self.patch_size * self.patch_size, w // self.patch_size * self.patch_size
        concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)

        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone.forward_features(
                concat)['x_norm_patchtokens']
            x = x.mT.reshape(bs * n, -1,
                                concat.shape[-2] // self.patch_size,
                                concat.shape[-1] // self.patch_size)
        feats = self.neck(x) # (36, 64)

        decoder_inputs = self.pre_transformer(feats) # todo (bv 256 144, 256) (bv 256 72 128) (bv 256 36 64) (bv 256 18 32) 
        feats = flatten_multi_scale_feats(feats)[0]
        decoder_inputs.update(self.pre_decoder(feats))

        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs)
        query = decoder_outputs['hidden_states'] # (num_layers, b 300 256)
        reference_points = decoder_outputs['references']




        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1],
                image_shape=(h,w),
                mode=mode, **data_samples)

        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads): # 多层
            loss = gauss_head(
                query[i], reference_points[i],
                image_shape=(h,w),
                mode=mode, **data_samples)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        return losses


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

    def pre_decoder(self, memory):
        bs, _, c = memory.shape
        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1)
        reference_points = torch.rand((bs, query.size(1), 2)).to(query)

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