import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from einops import rearrange


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gausstr.models.gsplat_rasterization import rasterize_gaussians
from gausstr.models.utils import (OCC3D_CATEGORIES, cam2world, flatten_bsn_forward,
                    get_covariance, rotmat_to_quat)
from gausstr.models.gausstr_head import prompt_denoising,merge_probs



@MODELS.register_module()
class GaussTRV2Head(BaseModule):

    def __init__(self,
                 opacity_head,
                 feature_head,
                 scale_head,
                 regress_head,
                 reduce_dims,
                 image_shape,
                 patch_size,
                 voxelizer,

                 segment_head=None,
                 rgb_head=None,
                 loss_lpips=None,
                 loss_depth=None,

                 near = 0.5,
                 far = 51.2,

                 depth_limit=51.2,
                 projection=None,
                 text_protos=None,
                 prompt_denoising=True):
        super().__init__()
        self.opacity_head = MODELS.build(opacity_head)
        self.feature_head = MODELS.build(feature_head)
        self.scale_head = MODELS.build(scale_head)
        self.regress_head = MODELS.build(regress_head)

        # todo 增加的模块
        self.segment_head = MODELS.build(
            segment_head) if segment_head else None
        self.rgb_head = MODELS.build(
            rgb_head) if rgb_head else None
        self.loss_lpips = MODELS.build(
            loss_lpips) if loss_lpips else None
        self.loss_depth = MODELS.build(
            loss_depth) if loss_depth else None

        self.near = near
        self.far = far

        self.reduce_dims = reduce_dims
        self.image_shape = image_shape # todo 网络输入尺寸
        self.patch_size = patch_size
        self.depth_limit = depth_limit

        self.prompt_denoising = prompt_denoising # todo True

        if projection is not None:
            self.projection = MODELS.build(projection) # todo 将 768 -> 128
            # if 'init_cfg' in projection and projection.init_cfg.type == 'Pretrained':
            #     self.projection.requires_grad_(False)
        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu')) # todo CLIP类别嵌入 (h_dim,n_class)

        self.voxelizer = MODELS.build(voxelizer)
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg')) # todo mmseg

    def forward(self,
                x, # todo 查询特征
                ref_pts, # todo 参考点
                depth, # todo 真实深度图 (b v h w)
                cam2img,
                cam2ego,
                pixel_gaussians=None,
                feats=None,
                img_aug_mat=None, # todo (b v 4 4)
                gt_imgs=None, # todo inputs: 网络输入(rgb) ((b v) c h w)
                sem_segs=None, # todo cam2img, cam2ego, feats, img_aug_mat, sem_segs: 标注和真值 (b v h w)
                mode='tensor',
                **kwargs):

        bs, n = cam2img.shape[:2] # todo: n: 视角数
        # depth = depth.clamp(max=self.depth_limit) # depth_limit: 51.2

        if x is not None:
            assert ref_pts is not None
            x = x.reshape(bs, n, *x.shape[1:]) # (b,v,300,256)
            # ----------------------------------------------------#
            # 偏移量计算
            deltas = self.regress_head(x) # (b,v,300,3) 计算偏移量：表示每个参考点的位置调整

            if ref_pts.shape[-1] == 3:
                # todo ------------------------------------------------------#
                # todo 参考点：三维 x y h + 预测的偏移量
                ref_pts = (deltas + inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid() # todo 增加了batch维度

                points = torch.cat([
                    ref_pts[...,:2] * torch.tensor(self.image_shape[::-1]).to(x),
                    ref_pts[...,2,None] * self.depth_limit, # [...,2:] [...,2,None] 和[...,2:3] 效果一致 [...,2]会降维
                ],-1)

                sample_depth = ref_pts[...,2,None] * self.depth_limit
            else:
                assert ref_pts.shape[-1] == 2 # todo 二维

                ref_pts = (
                    deltas[..., :2] +
                    inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid() # 参考点位置更新，参考点与x，y偏移量相加，得到新的参考点(在图像上的二维位置)

                # -----------------------------------------------------#
                #  todo 根据二维参考点的位置，直接从metric 3D预测的深度图中 采样深度信息
                sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None], # flatten_bsn_forward: 用于处理(bs n ...)的张量数据
                                                ref_pts[...,:2].unsqueeze(2) * 2 - 1) # 根据参考点对深度图进行采样，得到每个参考点的信息
                sample_depth = sample_depth[:, :, 0, 0, :, None]
                # 预测点 像素坐标系下
                points = torch.cat([
                    ref_pts * torch.tensor(self.image_shape[::-1]).to(x), # todo self.image_shape: 网络输入尺寸
                    sample_depth * (1 + deltas[..., 2:3])
                ], -1) # 计算3D点 (b,v,300,3)



            # todo ------------------------------------#
            # todo： 位置计算：cam2img cam2ego img_aug_mat
            means3d = cam2world(points, cam2img, cam2ego, img_aug_mat) # 将2D图像坐标转换为3D世界坐标
            # 从高斯查询中，预测高斯属性：透明度、特征向量(代替SH)、缩放因子、旋转四元数
            opacities = self.opacity_head(x).float() # 不透明度、特征和尺度计算
            # todo ------------------------------------#
            # features = self.feature_head(x).float() # (b,v,300,768)

            # todo ------------------------------------#
            rgb_features = self.rgb_head(x).float()
            seg_features = self.segment_head(x).float()


            # todo ------------------------------------#
            # todo： 协方差计算：
            scales = self.scale_head(x) * self.scale_transform(
                sample_depth, cam2img[..., 0, 0]).clamp(1e-6)
            covariances = flatten_bsn_forward(get_covariance, scales,
                                            cam2ego[..., None, :3, :3])

            # todo ------------------------------------#
            # todo： 旋转矩阵计算
            rotations = flatten_bsn_forward(rotmat_to_quat, cam2ego[..., :3, :3])
            rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1) # 协方差和旋转矩阵

            features = torch.cat([rgb_features,seg_features],dim=-1)


        if (pixel_gaussians is not None) and (x is None):
            means3d = pixel_gaussians.means
            features = torch.cat([
                            pixel_gaussians.rgbs,
                            pixel_gaussians.semantic,
                        ],dim = -1)
            opacities = pixel_gaussians.opacities.unsqueeze(-1)
            scales = pixel_gaussians.scales
            rotations = pixel_gaussians.rotations
            covariances = pixel_gaussians.covariances
            semantices = pixel_gaussians.semantic.softmax(-1)

        elif (pixel_gaussians is not None) and (x is not None):
            means3d = torch.cat([
                means3d.flatten(1, 2),
                pixel_gaussians.means,
            ],dim = 1)
            features = torch.cat([
                features.flatten(1, 2),
                torch.cat(
                    [
                        pixel_gaussians.rgbs,
                        pixel_gaussians.semantic,
                    ],dim = -1),
                ],dim=1)
            opacities = torch.cat([
                opacities.flatten(1, 2),
                pixel_gaussians.opacities.unsqueeze(-1),
            ],dim = 1)
            scales = torch.cat([
                scales.flatten(1, 2),
                pixel_gaussians.scales,
            ],dim = 1)
            rotations = torch.cat([
                rotations.flatten(1, 2),
                pixel_gaussians.rotations,
            ],dim = 1)
            covariances = torch.cat([
                covariances.flatten(1, 2),
                pixel_gaussians.covariances,
            ],dim =1)
            semantices = torch.cat([
                seg_features.flatten(1,2).softmax(-1),
                pixel_gaussians.semantic.softmax(-1),
            ],dim = 1)

        elif (pixel_gaussians is None) and (x is not None):
            means3d = means3d.flatten(1, 2)
            features = features.flatten(1, 2)
            opacities = opacities.flatten(1, 2)

            scales = scales.flatten(1, 2)
            rotations = rotations.flatten(1, 2)
            covariances = covariances.flatten(1, 2)
            semantices = seg_features.flatten(1,2).softmax(-1)
        else:
            raise ValueError('pixel_gaussians and query both None!')


        rendered = rasterize_gaussians(
            # means3d.flatten(1, 2), # (b,vx300,3)
            # features.flatten(1, 2), # (b,vx300,h_dim) 颜色/特征
            # opacities.squeeze(-1).flatten(1, 2),
            # scales.flatten(1, 2),
            # rotations.flatten(1, 2),
            means3d,
            features,
            opacities.squeeze(-1),
            scales,
            rotations,
            cam2img,
            cam2ego,
            img_aug_mats=img_aug_mat,
            image_size=(900, 1600), # 原图像尺寸
            near_plane=0.1,
            far_plane=100,
            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32).flatten(0, 1) # (b v c h w) ((b v) c h w)

        rendered_depth = rendered[:, -1] # todo ((b v) h w) 深度图
        rendered_rgb = rendered[:, :3] #  ((b v) c h w) -> ((b v) c-1 h w)
        rendered_seg = rendered[:,3:-1]

        # todo ---------------------------------------#
        # todo 推理
        if mode == 'predict':
            # todo 体素化
            density, grid_feats = self.voxelizer(
                # means3d=means3d.flatten(1, 2),  # (b,v,300,3) -> flatten: (b,vx300,3)
                # opacities=opacities.flatten(1, 2),
                # # features=feat_probs.flatten(1, 2).softmax(-1),
                # features = seg_features.flatten(1,2).softmax(-1), # (b,v,300,3) -> (b,vx300,3)
                # covariances=covariances.flatten(1, 2)

                means3d=means3d,
                opacities=opacities,
                # features=feat_probs.flatten(1, 2).softmax(-1),
                features = semantices, # (b,v,300,3) -> (b,vx300,3)
                covariances=covariances

                ) # 将离散的3D高斯分布转换为occ占据预测网格图

            if self.prompt_denoising:
                probs = prompt_denoising(grid_feats)
            else:
                probs = grid_feats.softmax(-1)

            preds = probs.argmax(-1) # (bs,h,w,16)
            preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17) # 密度过小，则将其类别设置为17（sky/empty类)

            rendered_seg = rearrange(rendered_seg,'bsv c h w -> bsv h w c').softmax(-1).argmax(-1)
            seg_pred = rearrange(rendered_seg,'(bs v) h w -> bs v h w',bs=bs) # (b v h w)

            depth_pred = rearrange(rendered_depth,'(bs v) h w -> bs v h w',bs=bs) # (b v h w)
            img_pred = rearrange(rendered_rgb,'(bs v) c h w -> bs v c h w',bs=bs) # (b v 3 h w )
            outputs = [{
                'occ_pred': preds, # (b,200,200,16)
                'depth_pred': depth_pred, # (b v h w)
                # 'seg_pred': rendered_seg.reshape(bs, n, *rendered_seg.shape[1:]), # (bsv h w)
                # 'seg_pred': None,
                'seg_pred':seg_pred,
                # 'img_pred': rendered_img.reshape(bs, n, *rendered_img.shape[1:]), # (bsv,3 h w)
                # 'img_pred': None,
                'img_pred': img_pred,
            }]
            # return preds
            return outputs

        # todo ---------------------------------------#
        # todo 训练：损失计算

        losses = {}
        # GaussTR原代码：把depth中大于等于depth_limit的深度值全部替换成1e-3
        # depth = torch.where(depth < self.depth_limit, depth,
        #                     1e-3).flatten(0, 1) # todo depth: 视觉基础模型提取的深度图

        depth = depth.clamp(max=self.depth_limit)
        depth = depth.flatten(0,1)
        # todo 深度估计损失 在MonoSplat工作中，没有对深度估计的结果做监督
        losses['loss_silog_l1_depth'] = self.depth_loss(rendered_depth, depth) # Silog损失
        # losses['mae_depth'] = self.depth_loss(
        #     rendered_depth, depth, criterion='l1') # todo 11.24 感觉这个损失的计算重复了

        # todo wys 11.24 尝试引入一下MonoSplat设计的depth预测损失
        if self.loss_depth:
            near = self.near
            near = torch.full((bs,n),near).to(depth.device)
            far = self.far
            far = torch.full((bs,n),far).to(depth.device)

            losses['loss_depth'] = self.loss_depth.forward(
                rearrange(rendered_depth,'(bs v) h w -> bs v h w',bs=bs),
                gt_imgs,
                near=near,
                far=far,)

        # todo MSE损失计算
        rgb_target = gt_imgs.flatten(0,1)
        reg_loss = (rendered_rgb - rgb_target) ** 2
        losses['loss_mae'] = reg_loss.mean() # todo mae损失
        # todo LPIPS损失计算: wys 11.24
        if self.loss_lpips:
            losses['loss_lpips'] = self.loss_lpips(rgb_target,rendered_rgb)


        # todo 语义预测损失计算
        probs = rendered_seg.flatten(2).mT
        target = sem_segs.flatten(0, 1).flatten(1).long()
        target = torch.where(target == 12, torch.tensor(0, device=target.device), target)
        losses['loss_ce'] = F.cross_entropy(
            probs.mT,
            target, # 分割图: min：0 max：17
            ignore_index=0) # 忽略第0和12类

        return losses

    def photometric_error(self, src_imgs, rec_imgs):
        return (0.85 * self.ssim(src_imgs, rec_imgs) +
                0.15 * F.l1_loss(src_imgs, rec_imgs))

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss

    def scale_transform(self, depth, focal, multiplier=7.5):
        return depth * multiplier / focal.reshape(*depth.shape[:2], 1, 1)

    def compute_ref_params(self, cam2img, cam2ego, ego2global, img_aug_mat):
        ego2keyego = torch.inverse(ego2global[:, 0:1]) @ ego2global[:, 1:]
        cam2keyego = ego2keyego.unsqueeze(2) @ cam2ego.unsqueeze(1)
        cam2keyego = torch.cat([cam2ego.unsqueeze(1), cam2keyego],
                               dim=1).flatten(1, 2)
        cam2img = cam2img.unsqueeze(1).expand(-1, 3, -1, -1, -1).flatten(1, 2)
        img_aug_mat = img_aug_mat.unsqueeze(1).expand(-1, 3, -1, -1,
                                                      -1).flatten(1, 2)
        return dict(
            cam2imgs=cam2img, cam2egos=cam2keyego, img_aug_mats=img_aug_mat)

    def visualize_rendered_results(self,
                                   results,
                                   arrangement='vertical',
                                   save_dir='vis'):
        # (bs, t*n, 3/1, h, w)
        assert arrangement in ('vertical', 'tiled')
        if not isinstance(results, (list, tuple)):
            results = [results]
        vis = []
        for res in results:
            res = res[0]
            if res.dim() == 3:
                res = res.reshape(
                    res.size(0), 1, -1, vis[0].size(1) // self.downsample)
                res = res.unsqueeze(0).expand(3, *([-1] * 4)).flatten(0, 1)
                res = F.interpolate(res, scale_factor=self.downsample)

            img = res.permute(0, 2, 3, 1)  # (t * n, h, w, 3/1)
            if arrangement == 'vertical':
                img = img.flatten(0, 1)
            else:
                img = torch.cat((
                    torch.cat((img[2], img[4]), dim=0),
                    torch.cat((img[0], img[3]), dim=0),
                    torch.cat((img[1], img[5]), dim=0),
                ),
                                dim=1)
            img = img.detach().cpu().numpy()
            if img.shape[-1] == 1:
                from matplotlib import colormaps as cm
                cmap = cm.get_cmap('Spectral_r')
                img = cmap(img / (img.max() + 1e-5))[..., 0, :3]
            img -= img.min()
            img /= img.max()
            vis.append(img)
        vis = np.concatenate(vis, axis=-2)

        if not hasattr(self, 'save_cnt'):
            self.save_cnt = 0
        else:
            self.save_cnt += 1
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        plt.imsave(osp.join(save_dir, f'{self.save_cnt}.png'), vis)
