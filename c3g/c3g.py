from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange,repeat
import torch.nn.functional as F

from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from vggt.models.vggt import VGGT
from vggt.heads.dpt_head import DPTHead

from .model.encoder.common.gmae import Transformer
from .model.encoder.common.gaussian_adapter import GaussianAdapter,UnifiedGaussianAdapter,Gaussians
from .geometry.projection import sample_image_grid

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@MODELS.register_module()
class C3G(BaseModel):

    def __init__(self,
                 vggt_path,
                 ori_image_shape,
                 num_gaussians = 2048,
                 pose_free = True,
                 patch_size = 14,
                 **kwargs):
        super().__init__(**kwargs)

        vggt_model = VGGT.from_pretrained(vggt_path)
        for param in vggt_model.parameters():
            param.requires_grad = False

        self.aggregator = vggt_model.aggregator

        self.dpt_head = DPTHead(2048)
        for n, param in self.dpt_head.named_parameters():
            param.requires_grad = False

        transformer_dim = 2048
        self.gaussian_tokens = nn.Parameter(torch.randn(num_gaussians, transformer_dim))
        self.anchor_positions = nn.Parameter(torch.tensor([[0.,0.,1.]]).repeat(num_gaussians,1), requires_grad=False)

        self.gmae_decoder = Transformer(
            dim=transformer_dim,
            depth=2,
            heads=16,
            dim_head=transformer_dim//16,
            mlp_dim=transformer_dim * 2,

        )

        self.pose_free = pose_free
        self.gaussian_adapter = UnifiedGaussianAdapter() if pose_free else GaussianAdapter()

        self.raw_gs_dim = 3 + 1 + self.gaussian_adapter.d_in
        gaussians_per_token = 1
        self.gaussians_per_token = gaussians_per_token


        self.gmae_to_gaussians = nn.Linear(transformer_dim, self.raw_gs_dim * gaussians_per_token)


        self.patch_size = patch_size
        self.ori_image_shape = ori_image_shape
        self.num_surfaces = 1
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

        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})

            # normalize the standred format into intrinsics
            ori_h, ori_w = self.ori_image_shape # (900, 1600)
            intrinsics = data_samples[i].cam2img
            intrinsics[:, 0, 0] /= ori_w
            intrinsics[:, 1, 1] /= ori_h
            intrinsics[:, 0, 2] /= ori_w
            intrinsics[:, 1, 2] /= ori_h
            cam2img.append(intrinsics)

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
        device = inputs.device

        concat = rearrange(inputs,"b v c h w -> (b v) c h w")
        resize_h, resize_w = h // self.patch_size * self.patch_size, w // self.patch_size * self.patch_size
        concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)
        resize  = rearrange(concat,"(b v) c h w -> b v c h w",b=bs)

        aggregated_tokens_list, patch_start_idx = self.aggregator(resize) # len -> 24 [[b v h*w/(14*14)+5 2048]] patch_start_idx: 5
        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                res = self.dpt_head(aggregated_tokens_list, resize, patch_start_idx)
                vis_depth = res[0][..., -1]

        dec_feat = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        dec_feat = rearrange(dec_feat, "b v n d -> b (v n) d")
        all_decoder_tokens = torch.cat((dec_feat, self.gaussian_tokens.unsqueeze(0).expand(bs, -1, -1),), dim=1)

        decoded_tokens = self.gmae_decoder(all_decoder_tokens, mask=None)
        gaussian_params = self.gmae_to_gaussians(decoded_tokens[:, -self.gaussian_tokens.shape[0]:])  # b n d(3+1+d') # todo (b 2048 14)
        gaussian_params = rearrange(gaussian_params, "b n (gpt c) -> b (n gpt) c", gpt=self.gaussians_per_token, c=self.raw_gs_dim)

        pts_all = gaussian_params[:, :, :3].unsqueeze(-2) + self.anchor_positions.unsqueeze(dim=0).repeat(bs,self.gaussians_per_token,1).unsqueeze(dim=2) # b n 3
        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = gaussian_params[:,:,3:]
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        gaussian_feature = None

        if self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                # self.map_pdf_to_opacity(densities, global_step), #
                densities,
                rearrange(gaussians[..., 1:], "b n srf c -> b n srf () c"),
                features = gaussian_feature,
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(bs, n, -1, -1, -1)

            intrinsics = data_samples['cam2img'][...,:3,:3]
            extrinsics = data_samples['cam2ego']

            gaussians = self.gaussian_adapter.forward(
                rearrange(extrinsics, "b v i j -> b v () () () i j"),
                rearrange(intrinsics, "b v i j -> b v () () () i j"),

                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                # self.map_pdf_to_opacity(densities, global_step),
                densities,
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        final_gaussians = Gaussians(
                    rearrange(
                        gaussians.means,
                        "b n srf spp xyz -> b (n srf spp) xyz",
                    ),
                    rearrange(
                        gaussians.covariances,
                        "b n srf spp i j -> b (n srf spp) i j",
                    ),
                    rearrange(
                        gaussians.harmonics,
                        "b n srf spp c d_sh -> b (n srf spp) c d_sh",
                    ),
                    rearrange(
                        gaussians.opacities,
                        "b n srf spp -> b (n srf spp)",
                    ),
                    gaussians.features if gaussians.features is not None else None
                )
        return