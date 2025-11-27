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
from gausstr.models.utils import flatten_multi_scale_feats

from gausstrv2.models.encoder import mv_feature_add_position,prepare_feat_proj_data_lists
from gausstrv2.models.encoder import warp_with_pose_depth_candidates
from gausstrv2.models.encoder import UNetModel
from gausstrv2.geometry import sample_image_grid,get_world_rays
from gausstrv2.models.encoder.common.gaussians import build_covariance
from gausstrv2.misc.sh_rotation import rotate_sh
from gausstrv2.models.types import Gaussians

# from gausstr.models.utils import cam2world

torch.autograd.set_detect_anomaly(True)


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

                # todo MonoSplat
                 depth_head=None,
                 cost_head=None,
                 transformer=None,
                 near = 0.5,
                 far = 51.2,
                # todo AnySplat

                 encoder=None,
                 pos_embed=None,
                 attn_type=None,
                 **kwargs):
        super().__init__(**kwargs)

        # todo backbone
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

        # todo -----------------------------------#
        # todo MonoSplat:
        use_monosplat = False
        if use_monosplat:
            if depth_head is not None:
                self.depth_head = MODELS.build(depth_head)
                for param in self.depth_head.parameters(): #! 不冻结似乎会导致报错
                    param.requires_grad = False


            if cost_head is not None:
                self.cost_head = MODELS.build(cost_head)
            if transformer is not None:
                self.transformer = MODELS.build(transformer)

            self.near = near
            self.far = far

            num_depth_candidates = 128
            feature_channels = 64
            costvolume_unet_attn_res = (4,)
            costvolume_unet_channel_mult = (1,1,1)
            costvolume_unet_feat_dim = 128
            num_views = 6 # num cams
            depth_unet_feat_dim = 32

            depth_unet_attn_res = [16]
            depth_unet_channel_mult = [1, 1, 1, 1, 1]

            # gaussian_raw_channels = 84 # 2+3+4+3x25
            gaussian_raw_channels = 12 # 2+3+4+3
            #??? 增加了一个delats的预测量
            # gaussian_raw_channels = 13 # 2+1+3+4+3=13

            gaussians_per_pixel = 1


            self.regressor_feat_dim=costvolume_unet_feat_dim
            # Cost volume refinement
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

            num_classes = 18
            self.to_gaussians_semantic = nn.Sequential(
                nn.Conv2d(gau_in, num_classes   * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(
                    num_classes * 2, num_classes, 3, 1, 1
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
            self.vit_type = 'vitb'
            self.intermediate_layer_idx = {
                "vits": [2, 5, 8, 11],
                "vitb": [2, 5, 8, 11],
                "vitl": [4, 11, 17, 23],
            }


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
        # todo GaussTR
        use_gausstr = False
        if use_gausstr:
            if hasattr(self, 'backbone'):
                # 原代码：必须确保 input_size 为14的倍数
                # inputs = inputs.flatten(0, 1) # (b,v,3,h,w) -> ((b v) 3 h w)
                # todo: 对原图缩放一下，
                resize_h, resize_w = ori_h // self.patch_size * self.patch_size, ori_w // self.patch_size * self.patch_size
                concat = rearrange(inputs,"b v c h w -> (b v) c h w")
                concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)

                if self.frozen_backbone:
                    if self.backbone.training:
                        self.backbone.eval()
                    with torch.no_grad(): # todo 提取单目视觉特征
                        if isinstance(self.backbone, BaseModule):
                            x = self.backbone(concat)[0]
                            if self.attn_type is not None:
                                x = self.custom_attn(x, self.attn_type)
                        else:

                            x = self.backbone.forward_features(
                                concat)['x_norm_patchtokens'] # todo ((b v) n 768) n = HxW / (patch_size x patch_size)

                            # 原代码：必须确保 input_size 为14的倍数
                            # x = x.mT.reshape(bs * n, -1,
                            #                  concat.shape[-2] // self.patch_size,
                            #                  concat.shape[-1] // self.patch_size) # 转置 -> 重新整合为特征图
                            x = x.mT.reshape(bs * n, -1,
                                            resize_h // self.patch_size,
                                            resize_w // self.patch_size) # 转置 -> 重新整合为特征图

                else:
                    x = self.backbone(concat)[0]
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
            # todo 查询高斯预测
            # todo Neck
            feats = self.neck(x) # todo ViTDetFPN: 多尺度特征图提取

            # 注：网络输入 (3 252 484) -> backbone -> (768 18 32) -> neck -> (256 72 128) (256 36 64) (256 18 32) (256 9 16)
            if hasattr(self, 'encoder'):
                encoder_inputs, decoder_inputs = self.pre_transformer(feats)
                feats = self.forward_encoder(**encoder_inputs)
            else:
                decoder_inputs = self.pre_transformer(feats) # 处理多尺度特征: 做多尺度位置编码等预处理工作，记录各层特征图的索引
                feats = flatten_multi_scale_feats(feats)[0] # 展平后的多尺度特征 (256 72 128) (256 36 64) (256 18 32) (256 9 16) -> (12240 256)




            # todo ---------------------------------#
            # todo 解码器
            decoder_inputs.update(self.pre_decoder(feats)) # todo 准备解码器的输入：查询特征、参考点(随机生成0-1之间的张量 # 二维/三维
            decoder_outputs = self.forward_decoder(
                reg_branches=[h.regress_head for h in self.gauss_heads], # todo 取 gauss_heads中的regress_head
                **decoder_inputs) # todo 解码

            query = decoder_outputs['hidden_states'] # todo 各解码层的query和参考点坐标
            reference_points = decoder_outputs['references']
        else:
            query = []
            reference_points = []
            for i in range(len(self.gauss_heads)):
                query.append(None)
                reference_points.append(None)


        # todo ---------------------------------#
        # todo MonoSplat start 像素高斯预测
        use_monosplat = False
        if use_monosplat:
            device = inputs.device

            # 将输入再缩放到更小的尺寸,以减少计算开销
            h, w = 112,192 # 图像尺寸
            inputs_resize = F.interpolate(rearrange(inputs,"b v c h w -> (b v) c h w"),size=(h,w), mode='bilinear', align_corners=False)
            inputs_resize = rearrange(inputs_resize,"(b v) c h w -> b v c h w",b=bs)
            concat = rearrange(inputs_resize,"b v c h w -> (b v) c h w")
            resize_h, resize_w = h // self.patch_size * self.patch_size, w // self.patch_size * self.patch_size
            concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)

            inter_h, inter_w = 64,64

            near = self.near
            near = torch.full((bs,n),near).to(x.device)
            far = self.far
            far = torch.full((bs,n),far).to(x.device)


            num_depth_candidates = 128
            gaussians_per_pixel = 1
            num_surfaces = 1
            opacity_multiplier = 1
            gpp = gaussians_per_pixel
            # d_sh = 25
            scale_min = 0.5
            scale_max = 15.0

            cam2img = data_samples['cam2img'] # (b v 4 4) -> (b v 3 3)
            cam2ego = data_samples['cam2ego']
            img_aug_mat = data_samples['img_aug_mat']

            extrinsics = data_samples['cam2ego']

            # 900 -> 252 1600 -> 448
            s_x, s_y = 1/1600, 1/900


            intrinsics = cam2img[...,:3,:3].clone() # TODO 相机内参
            intrinsics[:, :, 0, 0] *= s_x
            intrinsics[:, :, 1, 1] *= s_y
            intrinsics[:, :, 0, 2] *= s_x
            intrinsics[:, :, 1, 2] *= s_y

            num_reference_views = 1

            cam_origins = extrinsics[:,:,:3,-1]
            distance_matrix = torch.cdist(cam_origins, cam_origins, p=2)
            _, idx = torch.topk(distance_matrix, num_reference_views + 1, largest=False, dim=2)

            # 输出中间层特征
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

            # todo intr_warped, poses_warped
            features_mv_warped, intr_warped, poses_warped = (
                prepare_feat_proj_data_lists(
                    rearrange(features_mv, "(b v) c h w -> b v c h w", v=n, b=bs),
                    intrinsics, # (b v 3 3)
                    extrinsics, # (b v 4 4)
                    num_reference_views=num_reference_views,
                    idx=idx) # idx
            )

            # todo 将深度范围[near, far]映射到视差范围[1/far, 1/near], 确保在视差范围内均匀采样
            min_disp = rearrange(1.0 / far.clone().detach(), "b v -> (b v) ()") # far: 100.0
            max_disp = rearrange(1.0 / near.clone().detach(), "b v -> (b v) ()") # near: 0.5
            disp_range_norm = torch.linspace(0.0, 1.0, num_depth_candidates).to(min_disp.device) # num_depth_candidates: 128
            disp_candi_curr = (min_disp + disp_range_norm.unsqueeze(0) * (max_disp - min_disp)).type_as(features_mv)
            disp_candi_curr = repeat(disp_candi_curr, "bv d -> bv d fh fw", fh=features_mv.shape[-2], fw=features_mv.shape[-1])

            # todo 构建多视角代价体
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
            features_mono_tmp = F.interpolate(features_mono, (64, 64), mode="bilinear", align_corners=True)
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

            gaussians_semantic = self.to_gaussians_semantic(raw_gaussians_in)

            # delta fine depth and density
            disparity_in = [refine_out, disps_metric_fullres, disps_rel_fullres, pdf_max]
            disparity_in = torch.cat(disparity_in, dim=1)
            delta_disps_density = self.to_disparity(disparity_in)
            delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)

            # outputs
            # todo 深度
            fine_disps = (disps_metric_fullres + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (b v) () () ()"),
                1.0 / rearrange(near, "b v -> (b v) () () ()"),
            ) # 原始视差+细化的偏移量，将预测视差限制在此[1/far,1/near]范围内
            depths = 1.0 / fine_disps # 将视差转换成深度
            depths = repeat(
                depths,
                "(b v) dpt h w -> b v (h w) srf dpt",
                b=bs,
                v=n,
                srf=1,
            ) # (b v (h w) 1 1)
            # todo 透明度
            densities = repeat(
                F.sigmoid(raw_densities),
                "(b v) dpt h w -> b v (h w) srf dpt",
                b=bs,
                v=n,
                srf=1,
            ) # 密度/透明度 (b v (h w) 1 1)

            raw_gaussians = rearrange(raw_gaussians, "(b v) c h w -> b v (h w) c", v=n, b=bs) # (b v (h w) 84)

            # Convert the features and depths into Gaussians.
            xy_ray, _ = sample_image_grid((h, w), device) # todo [0,1]之间采样网格！ (h w xy)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=num_surfaces,
            ) # c = 12 = 2(offset_xy) + 3(scales) + 4(rotations) + 3(rgbs)
            offset_xy = gaussians[..., :2].sigmoid()

            #??----------------------------------------?
            #??????? delats：一个细化深度的参数
            # delats = gaussians[...,3]

            # todo 加上偏移量
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size # 归一化的图像坐标


            # 解析得到最后的gaussian属性
            # todo intrinsics extrinsics
            extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
            intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j") # 归一化的内参

            coordinates = rearrange(xy_ray, "b v r srf xy -> b v r srf () xy") # 归一化的像素坐标

            #?? 增加了一个delats 的预测量
            gaussians = rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c")
            # gaussians = rearrange(gaussians[..., 3:], "b v r srf c -> b v r srf () c")


            opacities = densities / gpp

            # scales, rotations, sh = gaussians.split((3, 4, 3 * d_sh), dim=-1)
            scales, rotations, rgbs = gaussians.split((3, 4, 3), dim=-1)
            scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
            rotations = rotations.broadcast_to((*scales.shape[:-1],4)) # rotations

            # Apply sigmoid to get valid colors.
            # sh_mask = torch.ones((d_sh,), dtype=torch.float32).to(device) # d_sh: 25
            # sh_degree = 4
            # for degree in range(1, sh_degree + 1):
            #     sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

            # todo c2w_rotations
            c2w_rotations = extrinsics[..., :3, :3]

            # sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            # sh = sh.broadcast_to((*opacities.shape, 3, d_sh)) * sh_mask
            # harmonics = rotate_sh(sh, c2w_rotations[..., None, :, :]) # 谐函数

            # Create world-space covariance matrices.
            covariances = build_covariance(scales, rotations)
            covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # 自车坐标系下的协方差矩阵

            # todo 计算高斯点的均值/位置:
            # Compute Gaussian means.
            # todo 注释的原代码：...
            origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
            means = origins + directions * depths[..., None] # 初始位置 + 方向 x 预测深度

            # # ??---------------------------------------------------?
            # # ?????? 参照GaussTR，根据xy_ray，采样得到深度信息
            # depth = data_samples['depth']
            # depth = depth.clamp(max=self.gauss_heads[0].depth_limit)
            # depth = rearrange(depth,"bs v h w -> (bs v) 1 h w")
            # ref_pts = rearrange(xy_ray,"bs v n 1 k -> (bs v) n 1 k") # k=2 (x,y)

            # sample_depth = F.grid_sample(depth,ref_pts*2-1) # todo F.grid_sample: 对图像进行采样 采样网格：[-1,1]
            # sample_depth = rearrange(sample_depth,"(bs v) c n 1 -> bs v (n 1) c",bs=bs)
            # ref_pts = rearrange(ref_pts,"(bs v) n 1 k -> bs v (n 1) k",bs=bs)
            # points = torch.cat([
            #     ref_pts * torch.tensor(self.gauss_heads[0].image_shape[::-1]).to(ref_pts), # image_shape: (h,w) -> [::-1] -> (w,h)
            #     sample_depth * (1 + delats)
            # ],-1)

            # means = cam2world(points,cam2img,cam2ego,img_aug_mat)[...,None,None,:] # (b v r xyz) -> (b v rsrf spp xyz)

            '''
            # todo debug：直接使用 像素坐标+深度值 作为高斯点在像素坐标系下的坐标，投影回三维空间中，来查看大部分是否落在OCC感知空间内
            xy_ray_gt, _ = sample_image_grid((h,w),device) # todo sample_image_grid: 生成一个给定形状的网格坐标，包括：归一化的浮点坐标(范围0-1)；整数像素索引：表示真实像素的行列号
            xy_ray_gt = rearrange(xy_ray_gt, "h w xy -> (h w) () xy")[None,None,...]
            xy_ray_gt = xy_ray_gt.expand(bs,n,-1,-1,-1)
            depth = data_samples['depth']
            depth = rearrange(depth,"bs v h w -> (bs v) 1 h w")
            ref_pts_gt = rearrange(xy_ray_gt,"bs v n 1 k -> (bs v) n 1 k") # k=2 (x,y)
            sample_depth = F.grid_sample(depth,ref_pts_gt*2-1)
            sample_depth = rearrange(sample_depth,"(bs v) c n 1 -> bs v (n 1) c",bs=bs)
            ref_pts_gt = rearrange(ref_pts_gt,"(bs v) n 1 k -> bs v (n 1) k",bs=bs)
            points_gt = torch.cat([
                ref_pts_gt * torch.tensor(self.gauss_heads[0].image_shape[::-1]).to(ref_pts_gt), # image_shape: (h,w) -> [::-1] -> (w,h)
                sample_depth
            ],-1)
            means_gt = cam2world(points_gt,cam2img,cam2ego,img_aug_mat)
            means_gt = rearrange(means_gt,"bs v r xyz -> bs (v r) xyz")[0] # 取一个batch
            np.save("means3d_gt.npy", means_gt.cpu().numpy())

            means_pred = rearrange(means,"b v r srf spp xyz -> b (v r srf spp) xyz")[0] # 取一个batch
            means_pred = means_pred.detach()
            np.save("means3d_pred.npy", means_pred.cpu().numpy())
            '''


            pixel_gaussians = Gaussians(
                rearrange(means,"b v r srf spp xyz -> b (v r srf spp) xyz",),
                rearrange(covariances,"b v r srf spp i j -> b (v r srf spp) i j",),
                rearrange(scales,"b v r srf spp c -> b (v r srf spp) c",),
                rearrange(rotations,"b v r srf spp c -> b (v r srf spp) c",),
                rearrange(rgbs,"b v r srf spp rgb -> b (v r srf spp) rgb",),
                rearrange(gaussians_semantic, '(b v) c h w -> b (v h w) c',b=bs),
                # rearrange(harmonics,"b v r srf spp c d_sh -> b (v r srf spp) c d_sh",),
                rearrange(opacity_multiplier * opacities,"b v r srf spp -> b (v r srf spp)",),
            )
        else:
            pixel_gaussians = None
        # todo MonoSplat end
        # todo --------------------------------------#





        # todo ---------------------------------#
        # todo 推理
        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1],
                gt_imgs = inputs,
                pixel_gaussians = pixel_gaussians,
                mode=mode,
                **data_samples)

        # todo ---------------------------------#
        # todo 训练：损失计算
        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads): # 多层
            # todo 对各层预测得到的查询进行高斯属性预测: 对query解码得到其余高斯属性,
            loss = gauss_head(
                query[i], reference_points[i],
                gt_imgs = inputs,
                pixel_gaussians = pixel_gaussians,
                mode=mode,
                **data_samples)
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
