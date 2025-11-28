from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from einops import rearrange,repeat
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monosplat.models.encoder import mv_feature_add_position,prepare_feat_proj_data_lists
from monosplat.models.encoder import warp_with_pose_depth_candidates
from monosplat.models.encoder import UNetModel
from monosplat.geometry import sample_image_grid,get_world_rays
from monosplat.models.encoder.common.gaussians import build_covariance
from monosplat.misc.sh_rotation import rotate_sh
from monosplat.models.types import Gaussians





@MODELS.register_module()
class MonoSplat(BaseModel):
    def __init__(self,
                depth_head,
                cost_head,
                transformer,
                decoder,

                near = 0.5,
                far = 100.,

                ori_image_shape = (900,1600),

                scale_min = 0.5,
                scale_max = 15.0,
                num_surfaces = 1,
                opacity_multiplier = 1,

                num_depth_candidates = 128,
                feature_channels = 64,
                costvolume_unet_attn_res = (4,),
                costvolume_unet_channel_mult = (1,1,1),
                costvolume_unet_feat_dim = 128,
                num_views = 6,# num cams
                depth_unet_feat_dim = 32,
                depth_unet_attn_res = [16],
                depth_unet_channel_mult = [1, 1, 1, 1, 1],

                gaussian_raw_channels = 84,
                d_sh = 25,

                gaussians_per_pixel = 1,
                patch_size = 14,

                vit_type = 'vitb',
                model_url = '',


                **kwargs,):
        super().__init__(**kwargs)


        self.vit_type = vit_type

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        if vit_type == 'vits':
            from dinov2.models.vision_transformer import vit_small
            self.backbone = vit_small(

                img_size = 518,
                patch_size = 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
                # num_register_tokens=4, # vits14_reg4
                num_register_tokens=0, # vits14
                interpolate_antialias = False,
                interpolate_offset = 0.1,
                )
        elif vit_type == 'vitb':
            from dinov2.models.vision_transformer import vit_base
            self.backbone = vit_base(
                img_size = 518,
                patch_size = 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
                num_register_tokens=4, # vits14_reg4
                interpolate_antialias = False,
                interpolate_offset = 0.1,
                )
        else:
            assert ValueError(f'not support vit_type: {vit_type}!')

        state_dict = torch.load(model_url, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        print(f"load checkpoint from{model_url}.")
        self.backbone.requires_grad_(False)

        self.depth_head = MODELS.build(depth_head)
        # for param in self.depth_head.parameters():
        #     param.requires_grad = False

        self.cost_head = MODELS.build(cost_head)
        self.transformer = MODELS.build(transformer)

        self.near = near
        self.far = far

        self.num_depth_candidates = num_depth_candidates
        self.gaussians_per_pixel = gaussians_per_pixel

        self.scale_min = scale_min
        self.scale_max = scale_max

        self.ori_image_shape = ori_image_shape

        self.num_surfaces = num_surfaces
        self.opacity_multiplier = opacity_multiplier

        self.d_sh = d_sh
        self.patch_size = patch_size


        self.regressor_feat_dim=costvolume_unet_feat_dim
        input_channels = num_depth_candidates + feature_channels * 2
        channels = self.regressor_feat_dim

        self.corr_refine_net = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=costvolume_unet_attn_res,
                channel_mult=costvolume_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1))

        self.regressor_residual = nn.Conv2d(input_channels, num_depth_candidates, 1, 1, 0)

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # # CNN-based feature upsampler
        self.proj_feature_mv = nn.Conv2d(feature_channels, depth_unet_feat_dim, 1, 1)
        self.proj_feature_mono = nn.Conv2d(feature_channels, depth_unet_feat_dim, 1, 1)

        # Depth refinement: 2D U-Net
        input_channels = depth_unet_feat_dim*2 + 3 + 1 + 1 + 1
        channels = depth_unet_feat_dim
        self.refine_unet = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )

        # Gaussians prediction: covariance, color
        gau_in = 3 + depth_unet_feat_dim + 2 * depth_unet_feat_dim
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        in_channels = 1 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        self.to_disparity = nn.Sequential(
            nn.Conv2d(in_channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        )

        self.decoder = MODELS.build(decoder)

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

            # normalize:
            ori_h, ori_w = self.ori_image_shape # (900, 1600)
            intrinsics = data_samples[i].cam2img
            intrinsics[:, 0, 0] /= ori_w
            intrinsics[:, 1, 1] /= ori_h
            intrinsics[:, 0, 2] /= ori_w
            intrinsics[:, 1, 2] /= ori_h
            cam2img.append(intrinsics)

            data_samples[i].set_metainfo(
                {'cam2ego': data_samples[i].cam2ego[:num_views]})

            extrinsics = data_samples[i].cam2ego
            cam2ego.append(extrinsics)

            ego2global.append(data_samples[i].ego2global)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat[:num_views]})
                img_aug_mat.append(data_samples[i].img_aug_mat)
            depth.append(data_samples[i].depth)
            if hasattr(data_samples[i], 'sem_seg'):
                sem_segs.append(data_samples[i].sem_seg)

            if hasattr(data_samples[i], 'img'):
                rgb_gts.append(data_samples[i].img) # 真实图像

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
        if rgb_gts:
            data_samples['rgb_gts'] = rgb_gts # 真实图像

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


        bs, n, _, h, w = inputs.shape
        device = inputs.device

        inputs_resize = F.interpolate(rearrange(inputs,"b v c h w -> (b v) c h w"),size=(h,w), mode='bilinear', align_corners=False)
        inputs_resize = rearrange(inputs_resize,"(b v) c h w -> b v c h w",b=bs)
        concat = rearrange(inputs_resize,"b v c h w -> (b v) c h w")
        resize_h, resize_w = h // self.patch_size * self.patch_size, w // self.patch_size * self.patch_size
        concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)

        near = self.near
        near = torch.full((bs,n),near).to(device)
        far = self.far
        far = torch.full((bs,n),far).to(device)

        num_depth_candidates = self.num_depth_candidates
        gaussians_per_pixel = self.gaussians_per_pixel
        num_surfaces = self.num_surfaces
        opacity_multiplier = self.opacity_multiplier
        gpp = gaussians_per_pixel

        d_sh = self.d_sh
        scale_min = self.scale_min
        scale_max = self.scale_max

        num_reference_views = 1
        inter_w, inter_h = 64,64

        intrinsics = data_samples['cam2img'][...,:3,:3]
        extrinsics = data_samples['cam2ego']


        cam_origins = extrinsics[:,:,:3,-1]
        distance_matrix = torch.cdist(cam_origins, cam_origins, p=2)
        _, idx = torch.topk(distance_matrix, num_reference_views + 1, largest=False, dim=2)
        features = self.backbone.get_intermediate_layers(concat,
                                                        self.intermediate_layer_idx[self.vit_type],
                                                        return_class_token=True) # 4 ([bv n h_dim]) n = 252 / 14 x 448  / 14 = 18 x 32 = 576
        features_mono, disps_rel = self.depth_head(features,
                                                patch_h=resize_h // self.patch_size,
                                                patch_w=resize_w // self.patch_size)  # (bv c h1 w1) (bv 1 h1 w1) h1 = 18 x 8 = 144  w1 = 32 x 8 = 256

        features_mv = self.cost_head(features,
                                    patch_h=resize_h // self.patch_size,
                                    patch_w=resize_w // self.patch_size) # (bv c h1 w1) h1 = 18 x 8 = 144  w1 = 32 x 8 = 256


        features_mv = F.interpolate(features_mv, (inter_w, inter_h), mode="bilinear", align_corners=True) #

        features_mv = mv_feature_add_position(features_mv, 2, feature_channels=64)
        features_mv_list = list(torch.unbind(rearrange(features_mv, "(b v) c h w -> b v c h w", b=bs, v=n), dim=1))
        features_mv_list = self.transformer(
            features_mv_list,
            attn_num_splits=2,
            nn_matrix=idx,
        )
        features_mv = rearrange(torch.stack(features_mv_list, dim=1), "b v c h w -> (b v) c h w")

        features_mv_warped, intr_warped, poses_warped = (
            prepare_feat_proj_data_lists(
                rearrange(features_mv, "(b v) c h w -> b v c h w", v=n, b=bs),
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)
                num_reference_views=num_reference_views,
                idx=idx) # idx
        )

        min_disp = rearrange(1.0 / far.clone().detach(), "b v -> (b v) ()") # far: 100.0
        max_disp = rearrange(1.0 / near.clone().detach(), "b v -> (b v) ()") # near: 0.5
        disp_range_norm = torch.linspace(0.0, 1.0, num_depth_candidates).to(min_disp.device) # num_depth_candidates: 128
        disp_candi_curr = (min_disp + disp_range_norm.unsqueeze(0) * (max_disp - min_disp)).type_as(features_mv)
        disp_candi_curr = repeat(disp_candi_curr, "bv d -> bv d fh fw", fh=features_mv.shape[-2], fw=features_mv.shape[-1])

        raw_correlation_in = []
        for i in range(num_reference_views):
            features_mv_warped_i = warp_with_pose_depth_candidates(
                features_mv_warped[:, i, :, :, :],
                intr_warped[:, i, :, :],
                poses_warped[:, i, :, :],
                1 / disp_candi_curr, # disp：候选视差 -> 1 / disp: 候选深度
                warp_padding_mode="zeros"
            )
            raw_correlation_in_i = (features_mv.unsqueeze(2) * features_mv_warped_i).sum(1) / (features_mv.shape[1]**0.5)
            raw_correlation_in.append(raw_correlation_in_i)

        raw_correlation_in = torch.mean(torch.stack(raw_correlation_in, dim=1), dim=1)

        # refine cost volume and get depths
        features_mono_tmp = F.interpolate(features_mono, (inter_w, inter_h), mode="bilinear", align_corners=True)
        raw_correlation_in = torch.cat((raw_correlation_in, features_mv, features_mono_tmp), dim=1)
        raw_correlation = self.corr_refine_net(raw_correlation_in) # ((b v) c h w)
        raw_correlation = raw_correlation + self.regressor_residual(raw_correlation_in)
        pdf = F.softmax(self.depth_head_lowres(raw_correlation), dim=1)
        disps_metric = (disp_candi_curr * pdf).sum(dim=1, keepdim=True)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]
        pdf_max = F.interpolate(pdf_max, (h, w), mode="bilinear", align_corners=True)
        disps_metric_fullres = F.interpolate(disps_metric, (h, w), mode="bilinear", align_corners=True)

        # feature refinement
        features_mv_in_fullres = F.interpolate(features_mv, (h, w), mode="bilinear", align_corners=True)
        features_mv_in_fullres = self.proj_feature_mv(features_mv_in_fullres)
        features_mono_in_fullres = F.interpolate(features_mono, (h, w), mode="bilinear", align_corners=True) # todo 通过双线性插值对单目特征进行上采样
        features_mono_in_fullres = self.proj_feature_mono(features_mono_in_fullres)
        disps_rel_fullres = F.interpolate(disps_rel, (h, w), mode="bilinear", align_corners=True)

        images_reorder = rearrange(inputs_resize, "b v c h w -> (b v) c h w")
        refine_out = self.refine_unet(
            torch.cat((features_mv_in_fullres, features_mono_in_fullres, images_reorder, \
                disps_metric_fullres, disps_rel_fullres, pdf_max),
                    dim=1)
            )

        # gaussians head
        raw_gaussians_in = [refine_out, features_mv_in_fullres, features_mono_in_fullres, images_reorder]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)

        # delta fine depth and density
        disparity_in = [refine_out, disps_metric_fullres, disps_rel_fullres, pdf_max]
        disparity_in = torch.cat(disparity_in, dim=1)
        delta_disps_density = self.to_disparity(disparity_in)
        delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)

        fine_disps = (disps_metric_fullres + delta_disps).clamp(
            1.0 / rearrange(far, "b v -> (b v) () () ()"),
            1.0 / rearrange(near, "b v -> (b v) () () ()"),
        )

        depths = 1.0 / fine_disps # 将视差转换成深度
        depths = repeat(
            depths,
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=bs,
            v=n,
            srf=1,
        ) # (b v (h w) 1 1)
        densities = repeat(
            F.sigmoid(raw_densities),
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=bs,
            v=n,
            srf=1,
        )
        raw_gaussians = rearrange(raw_gaussians, "(b v) c h w -> b v (h w) c", v=n, b=bs) # (b v (h w) 84)

        xy_ray, _ = sample_image_grid((h, w), device) # [0.1]网格点
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=num_surfaces,
        )

        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size # 归一化的图像坐标

        extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
        intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j") # 归一化的内参

        coordinates = rearrange(xy_ray, "b v r srf xy -> b v r srf () xy") # 归一化的像素坐标
        gaussians = rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c")

        opacities = densities / gpp

        scales, rotations, sh = gaussians.split((3, 4, 3 * d_sh if d_sh is not None else 3), dim=-1)

        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        rotations = rotations.broadcast_to((*scales.shape[:-1],4)) # rotations

        c2w_rotations = extrinsics[..., :3, :3]

        harmonics = sh
        if d_sh is not None:
            # Apply sigmoid to get valid colors.
            sh_mask = torch.ones((d_sh,), dtype=torch.float32).to(device) # d_sh: 25
            sh_degree = 4
            for degree in range(1, sh_degree + 1):
                sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            sh = sh.broadcast_to((*opacities.shape, 3, d_sh)) * sh_mask
            harmonics = rotate_sh(sh, c2w_rotations[..., None, :, :]) # 谐函数

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # 自车坐标系下的协方差矩阵

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None] # 初始位置 + 方向 x 预测深度

        pixel_gaussians = Gaussians(
            rearrange(means,"b v r srf spp xyz -> b (v r srf spp) xyz",),
            rearrange(covariances,"b v r srf spp i j -> b (v r srf spp) i j",),
            rearrange(harmonics,"b v r srf spp c d_sh -> b (v r srf spp) c d_sh",) \
                if d_sh is not None else rearrange(harmonics,"b v r srf spp rgb -> b (v r srf spp) rgb",),
            rearrange(scales,"b v r srf spp c -> b (v r srf spp) c",),
            rearrange(rotations,"b v r srf spp c -> b (v r srf spp) c",),
            rearrange(opacity_multiplier * opacities,"b v r srf spp -> b (v r srf spp)",),
        )

        return  self.decoder(pixel_gaussians, image_shape=(h,w),near=near, far=far, mode=mode,**data_samples)

