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
                 depth_limit=51.2,
                 projection=None,
                 text_protos=None,
                 prompt_denoising=True):
        super().__init__()
        self.opacity_head = MODELS.build(opacity_head)
        self.feature_head = MODELS.build(feature_head)
        self.scale_head = MODELS.build(scale_head)
        self.regress_head = MODELS.build(regress_head)

        self.segment_head = MODELS.build(
            segment_head) if segment_head else None
        self.rgb_head = MODELS.build(
            rgb_head) if rgb_head else None


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
                feats=None,
                img_aug_mat=None, # todo (b v 4 4)
                gt_imgs=None, # todo inputs: 网络输入(rgb) ((b v) c h w)
                sem_segs=None, # todo cam2img, cam2ego, feats, img_aug_mat, sem_segs: 标注和真值 (b v h w)
                mode='tensor',
                **kwargs):
        bs, n = cam2img.shape[:2] # todo: n: 视角数
        x = x.reshape(bs, n, *x.shape[1:]) # (b,v,300,256)

        depth = depth.clamp(max=self.depth_limit) # depth_limit: 51.2

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


        # if mode == 'predict':
        #     # todo 使用线性层进行预测
        #     # features = self.projection(features)


        #     # todo ------------------------#
        #     # todo
        #     rendered = rasterize_gaussians(
        #         means3d.flatten(1, 2), # (b,vx300,3)
        #         features.flatten(1, 2), # (b,vx300,dim) 颜色/特征
        #         opacities.squeeze(-1).flatten(1, 2),
        #         scales.flatten(1, 2),
        #         rotations.flatten(1, 2),
        #         cam2img,
        #         cam2ego,
        #         img_aug_mats=img_aug_mat,
        #         image_size=(900, 1600),
        #         near_plane=0.1,
        #         far_plane=100,
        #         render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
        #         channel_chunk=32).flatten(0,1) # (b v c h w) ((b v) c h w)

        #     rendered_depth = rendered[:, -1]
        #     rendered = rendered[:, :-1]

        #     rendered = rearrange(rendered,'bsv c h w -> bsv h w c')
        #     # rendered_img = self.img_head(rendered)
        #     # rendered_img = rearrange(rendered_img,'bsv h w c -> bsv c h w')

        #     # rendered_seg = self.segment_head(rendered)
        #     # rendered_seg = rearrange(rendered_seg,'bsv h w c -> bsv c h w')
        #     # rendered_seg = rearrange(rendered_seg,'bsv h w c -> bsv c h w')

        #     rendered_seg = rendered @ self.text_proto_embeds
        #     rendered_seg = merge_probs(rendered_seg, OCC3D_CATEGORIES)
        #     rendered_seg = rendered_seg.softmax(-1).argmax(-1)
        #     rendered_seg += (rendered_seg > 10) * 1 + 1

        #     # todo --------------------------------#
        #     # todo 使用线性层进行预测
        #     # feat_probs = self.segment_head(features)

        #     # todo 使用CLIP文本嵌入进行分类预测
        #     # feat_probs = features @ self.text_proto_embeds # 文本嵌入：21 ： 14 + 2 + 5 有两大类包含多个小类

        #     # todo 体素化
        #     density, grid_feats = self.voxelizer(
        #         means3d=means3d.flatten(1, 2),
        #         opacities=opacities.flatten(1, 2),
        #         # features=feat_probs.flatten(1, 2).softmax(-1),
        #         features = seg_features.flatten(1,2).softmax(-1)
        #         covariances=covariances.flatten(1, 2)) # 将离散的3D高斯分布转换为occ占据预测网格图

        #     if self.prompt_denoising:
        #         probs = prompt_denoising(grid_feats)
        #     else:
        #         probs = grid_feats.softmax(-1)

        #     # todo 已定义分类头数为18，无需再额外后处理
        #     #? 使用线性层进行预测
        #     preds = probs.argmax(-1) # (bs,h,w,16)
        #     preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17) # 密度过小，将其类别设置为17（天空类)

        #     #? 使用CLIP生成的文本嵌入进行预测
        #     # probs = merge_probs(probs, OCC3D_CATEGORIES) # (bs,x,y,z,n_cls)
        #     # preds = probs.argmax(-1) # (bs,h,w,16)
        #     # preds += (preds > 10) * 1 + 1  # skip two classes of "others" others:0 other flat: 12 跳过这两类目标
        #     # preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17) # 密度过小，将其类别设置为17（天空类)

        #     outputs = [{
        #         'occ_pred': preds, # (b,200,200,16)
        #         # 'img_pred': rendered_img.reshape(bs, n, *rendered_img.shape[1:]), # (bsv,3 h w)
        #         'img_pred': None,
        #         'depth_pred': rendered_depth.reshape(bs, n, *rendered_depth.shape[1:]), # (bsv h w)
        #         'seg_pred': rendered_seg.reshape(bs, n, *rendered_seg.shape[1:]), # (bsv h w)
        #     }]

        #     # return preds
        #     return outputs



        # todo feats: 视觉基础模型提取的特征图
        # tgt_feats = feats.flatten(-2).mT # todo .mT: 对矩阵转置，即交换最后两个维度
        # # if hasattr(self, 'projection'):
        # #     tgt_feats = self.projection(tgt_feats)[0]
        # # 执行PCA(主成分分析)降维操作，对输入张量进行低秩近似，减少特征维度
        # u, s, v = torch.pca_lowrank(
        #     tgt_feats.flatten(0, 2).double(), q=self.reduce_dims, niter=4) # (b,v,(h w),c) -> v: (c,q) q：降维后的维度
        # tgt_feats = tgt_feats @ v.to(tgt_feats)
        # # todo features: 查询特征
        # features = features @ v.to(features) # 768 -> 128
        # features = features.float()

        # 768 -> 3 + 128 (rgb + seg feat)
        # features = self.projection(features)

        #---------------------------------------#
        # gsplat 进行渲染 推理occ占据预测：无需光栅化
        features = torch.cat([rgb_features,seg_features],dim=-1)


        rendered = rasterize_gaussians(
            means3d.flatten(1, 2), # (b,vx300,3)
            features.flatten(1, 2), # (b,vx300,128) 颜色/特征
            opacities.squeeze(-1).flatten(1, 2),
            scales.flatten(1, 2),
            rotations.flatten(1, 2),
            cam2img,
            cam2ego,
            img_aug_mats=img_aug_mat,
            image_size=(900, 1600),
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
                means3d=means3d.flatten(1, 2),  # (b,v,300,3) -> flatten: (b,vx300,3)
                opacities=opacities.flatten(1, 2),
                # features=feat_probs.flatten(1, 2).softmax(-1),
                features = seg_features.flatten(1,2).softmax(-1), # (b,v,300,3) -> (b,vx300,3)
                covariances=covariances.flatten(1, 2)) # 将离散的3D高斯分布转换为occ占据预测网格图

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
        depth = torch.where(depth < self.depth_limit, depth,
                            1e-3).flatten(0, 1) # todo depth: 视觉基础模型提取的深度图

        # todo 深度估计损失
        losses['loss_depth'] = self.depth_loss(rendered_depth, depth) # todo 深度图计算损失
        losses['mae_depth'] = self.depth_loss(
            rendered_depth, depth, criterion='l1')

        # todo 特征损失估计
        # Interpolating to high resolution for supervision can improve mIoU by 0.7
        # compared to average pooling to low resolution.
        # bsn, c, h, w = rendered.shape
        # tgt_feats = feats.flatten(0,1).reshape(bsn, c, h // self.patch_size,
        #                                  w // self.patch_size)
        # tgt_feats = F.interpolate(
        #     tgt_feats, scale_factor=self.patch_size, mode='bilinear')
        # rendered_mT = rendered.flatten(2).mT
        # tgt_feats = tgt_feats.flatten(2).mT.flatten(0, 1)

        # # # todo 通过高斯点积进行渲染监督：
        # losses['loss_cosine'] = F.cosine_embedding_loss(
        #     rendered_mT.flatten(0, 1), tgt_feats, torch.ones_like(
        #         tgt_feats[:, 0])) * 5

        # rendered_mT = rendered.flatten(2).mT
        # todo img损失 参考omni-scene工作
        # rendered_img = self.img_head(rendered)
        rgb_target = gt_imgs.flatten(0,1)
        reg_loss = (rendered_rgb - rgb_target) ** 2
        losses['loss_img'] = reg_loss.mean() # todo mae损失
        # ? LPIPS损失计算: 待做


        # todo 分割图损失
        # ? (1) 使用线性层进行预测
        # rendered_seg = self.segment_head(rendered)
        # losses['loss_ce'] = F.cross_entropy(
        #     rendered_seg.mT,
        #     sem_segs.flatten(0, 1).flatten(1).long(), # 分割图: min：0 max：17
        #     ignore_index=0)

        # ? (2) 使用CLIP生成的文本嵌入进行预测
        # rendered_seg = rendered_mT @ self.text_proto_embeds
        # probs = merge_probs(rendered_seg, OCC3D_CATEGORIES)
        # b_p, n_p, c_p = probs.shape
        # zeros = torch.zeros(b_p, n_p, 1, device=probs.device, dtype=probs.dtype)
        # probs = torch.cat([
        #                     zeros,               # 0  忽略 others
        #                     probs[:, :, :11],    # 1-11
        #                     zeros,               # 12 忽略 others flat
        #                     probs[:, :, 11:],    # 13-17
        #                 ], dim=2)

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
