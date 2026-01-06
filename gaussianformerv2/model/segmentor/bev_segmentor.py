from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange,repeat

from mmdet3d.registry import MODELS

from .base_segmentor import CustomBaseSegmentor
from ...loss import CE_ssc_loss,lovasz_softmax
from ...geometry import sample_image_grid,get_world_rays
from ..decoder import rasterize_gaussians

from colorama import Fore


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])

@MODELS.register_module()
class BEVSegmentor(CustomBaseSegmentor):
    def __init__(
        self,
        
        pixel_gs = None,
        loss_lpips = None,
        
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],

        lovasz_ignore = 17,
        num_classes = 18,
        balance_cls_weight = True,
        manual_class_weight=[
        1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
        1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
        1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pixel_gs = MODELS.build(pixel_gs)
        self.loss_lpips = MODELS.build(loss_lpips)
        
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True

        self.lovasz_ignore = lovasz_ignore
        if balance_cls_weight:
            if manual_class_weight is not None:
                self.class_weights = torch.tensor(manual_class_weight)
            else:
                class_freqs = nusc_class_frequencies
                self.class_weights = torch.from_numpy(1 / np.log(class_freqs[:num_classes] + 0.001))
            self.class_weights = num_classes * F.normalize(self.class_weights, 1, -1)
            print(self.__class__, self.class_weights)
        else:
            self.class_weights = torch.ones(num_classes)
        print(cyan(f'successfully init Model!'))

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        
        
        img_feats = self.img_neck(img_feats) # todo FPN
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped}) # todo 4层多尺度特征图
        return result


    def _run_forward(self, data, mode): # todo 重写base_model的_run_forward()
        imgs = data.pop('img')
        metas = data
        results = self(imgs, metas, mode=mode)
        return results
    
    def plucker_embedder(
        self,
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def forward(self,
                imgs,
                metas,
                points = None,
                mode = 'loss',
                **kwargs,):
        
        results = {
            'imgs': imgs, # (b v 3 h w)
            'metas': metas,
            'points': points
        }
        
        
        results.update(kwargs)
        outs = self.extract_img_feat(**results) # todo 提取多尺度图像特征图outs:{'ms_img_feats'}: (b v c h w)  1/8 1/16 1/32 1/64 4个尺度的特征图        
        results.update(outs)

        # todo --------------------------#
        # todo 使用第1个特征图进行视图渲染        
        img_feats = outs['ms_img_feats'][0]
        input_h, input_w = imgs.shape[-2:] # todo 网络输入尺寸
        f_h, f_w = img_feats.shape[-2:] # todo 特征图尺寸
        
        img_aug_mat = metas['img_aug_mat'] # (b v 4 4) # todo 变换增强矩阵
        resize = torch.diag(torch.tensor([f_w/input_w, f_h/input_h],
                                         dtype=img_aug_mat.dtype,device = img_aug_mat.device))
        mat = torch.eye(4).to(img_aug_mat.device)
        mat[:2,:2] = resize
        mat = repeat(mat,"i j -> () () i j")
        img_aug_mat = mat @ img_aug_mat
        # todo 内参(相对于原图900x1600)
        intrinsics =  metas['cam2img'][...,:3,:3] # (b v 3 3)      
        # todo 外参
        extrinsics = metas['cam2lidar']
              
        depth = metas['depth']
        depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False)
        
        pixel_gaussians = self.pixel_gs(
                img_feats=rearrange(img_feats, "b v c h w -> (b v) c h w"),                 
                depths_in=depth, # (b v h w)
                intrinsics = intrinsics,
                extrinsics = extrinsics,
                img_aug_mat = img_aug_mat,
                )  

        means3d = pixel_gaussians.means
        harmonics = pixel_gaussians.harmonics # (b n c d_sh) | (b n c), c=rgb
        opacities = pixel_gaussians.opacities
        scales = pixel_gaussians.scales
        rotations = pixel_gaussians.rotations
        covariances = pixel_gaussians.covariances
        # todo --------------------------------------#       
        # todo 内参
        output_imgs = metas['output_img']
        rgb_gt = output_imgs
        output_h,output_w = output_imgs.shape[-2:]
        output_img_aug_mat = metas['output_img_aug_mat'] # (b v 4 4) # todo 变换增强矩阵(注意：仅缩放)
        output_intrinsics =  (output_img_aug_mat @ metas['output_cam2img'])[...,:3,:3] # (b v 3 3) # todo 相对于原图的内外参        
        
        # todo 外参
        output_extrinsics = metas['output_cam2lidar']

        colors, rendered_depth = rasterize_gaussians(
            extrinsics=output_extrinsics,
            intrinsics=output_intrinsics,
            image_shape = (output_h,output_w),
            means3d=means3d,
            rotations=rotations,
            scales=scales,
            covariances=covariances,
            opacities=opacities,
            colors=harmonics, # todo (b n c d_sh)
            use_sh=False,
            is_normalize = False, # todo 
            near_plane=0.1,
            far_plane=1000.,
        
            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32)               
        
        # todo --------------------------#
        # todo occ 占用预测
        outs = self.lifter(**results)
        results.update(outs)
        outs = self.encoder(**results)
        results.update(outs)
        outs = self.head(**results)
        results.update(outs)
        
        if mode == 'predict':
            outputs = [{
                    'occ_pred': results['final_occ'], # (b (h w d))
                    'occ_gt': results['sampled_label'], # (b (h w d))
                    'occ_mask': results['occ_mask'].flatten(1), # (b (h w d))
                    
                    'img_pred': colors,
                    'depth_pred': rendered_depth,
                }]
            return outputs

        # todo loss ----------------------------------------------#
        losses = {}
        
        
        occ_mask = results['occ_mask'].flatten(1) # (b 640000)
        semantics = results['pred_occ'][0] # (b 18 640000)
        sampled_label = results['sampled_label']

        sampled_label = sampled_label[occ_mask][None]

        semantics = semantics.transpose(1, 2)[occ_mask][None].transpose(1, 2)
        losses['loss_voxel_ce'] = 10.0 * \
            CE_ssc_loss(semantics, sampled_label, self.class_weights.type_as(semantics), ignore_index=255)

        lovasz_input = torch.softmax(semantics, dim=1) # todo (b num_classes g)
        losses['loss_voxel_lovasz'] = 1.0 * lovasz_softmax(
            lovasz_input.transpose(1, 2).flatten(0, 1), sampled_label.flatten(), ignore=self.lovasz_ignore)

        rendered_depth = rendered_depth.flatten(0,1)
        depth = depth.flatten(0,1)
        losses['loss_depth'] = 0.05 * self.depth_loss(rendered_depth, depth,criterion='l1')
        
        rgb = colors.flatten(0,1)
        rgb_gt = rgb_gt.flatten(0,1) / 255.
        reg_loss = (rgb - rgb_gt) ** 2
        losses['loss_l2'] = reg_loss.mean()
        losses['loss_lpips'] = self.loss_lpips(rgb_gt, rgb)
        
        return losses
    
    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion: # todo 这个没有用到
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion: # todo 只是用了l1 loss
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss