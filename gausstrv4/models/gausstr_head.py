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

from einops import rearrange,repeat
from gsplat import rasterization



from .gsplat_rasterization import rasterize_gaussians
from .utils import (OCC3D_CATEGORIES, cam2world, flatten_bsn_forward,
                    get_covariance, rotmat_to_quat)
from ..loss import lovasz_softmax_flat,CE_ssc_loss



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
class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 num_layers=2,
                 activation='relu',
                 mode=None,
                 range=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4 # todo 若左边是None或False，则使用右边默认值
        output_dim = output_dim or input_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation
        self.range = range
        self.mode = mode

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = getattr(F, self.activation)(
                layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.mode is not None:
            if self.mode == 'sigmoid':
                x = F.sigmoid(x)
            if self.range is not None:
                x = self.range[0] + (self.range[1] - self.range[0]) * x
        return x


def prompt_denoising(logits, logit_scale=100, pd_threshold=0.1):
    probs = logits.softmax(-1)
    probs_ = F.softmax(logits * logit_scale, -1)
    max_cls_conf = probs_.flatten(1, 3).max(1).values
    selected_cls = (max_cls_conf < pd_threshold)[:, None, None,
                                                 None].expand(*probs.shape)
    probs[selected_cls] = 0
    return probs

# todo -----------------------------#
# todo 合并类别概率
def merge_probs(probs, categories):
    merged_probs = []
    i = 0
    for cats in categories:
        p = probs[..., i:i + len(cats)]
        i += len(cats)
        if len(cats) > 1:
            p = p.max(-1, keepdim=True).values # todo 如果某个类别包含多个子类，则选择概率最大的子类来表示该类别概率
        merged_probs.append(p)
    return torch.cat(merged_probs, dim=-1)


@MODELS.register_module()
class GaussTRHead(BaseModule):

    def __init__(self,
                 opacity_head,
                 feature_head,
                 scale_head,
                 regress_head,
                 reduce_dims,
                 image_shape,
                 patch_size,
                 voxelizer,
                 num_classes,
                 segment_head=None,
                 depth_limit=51.2,
                 projection=None,
                 text_protos=None,
                 prompt_denoising=True,
                 balance_cls_weight = True,
                 manual_class_weight=[
                    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],

                 ):
        super().__init__()
        self.opacity_head = MODELS.build(opacity_head) # todo 透明度预测
        self.feature_head = MODELS.build(feature_head)
        self.scale_head = MODELS.build(scale_head)
        self.regress_head = MODELS.build(regress_head)
        self.segment_head = MODELS.build(
            segment_head) if segment_head else None

        self.reduce_dims = reduce_dims
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.depth_limit = depth_limit
        self.prompt_denoising = prompt_denoising

        if projection is not None:
            self.projection = MODELS.build(projection)
            if 'init_cfg' in projection and projection.init_cfg.type == 'Pretrained':
                self.projection.requires_grad_(False)
        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu')) # todo CLIP类别嵌入 (h_dim,n_class)

        self.voxelizer = MODELS.build(voxelizer)
        # self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg')) # todo mmseg

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
        self.num_classes = num_classes

    def forward(self,
                x,
                ref_pts,
                depth, # todo x, ref_pts, depth: 网络输出结果
                cam2img,
                cam2ego,
                feats=None,
                img_aug_mat=None,
                sem_segs=None, # todo cam2img, cam2ego, feats, img_aug_mat, sem_segs: 标注和真值
                occ_gts=None,
                mode='tensor',
                **kwargs):

        bs, n = cam2img.shape[:2] # todo
        x = x.reshape(bs, n, *x.shape[1:]) # (b,v,300,256)

        deltas = self.regress_head(x) # (b,v,300,3) 计算偏移量：表示每个参考点的位置调整: x,y,
        ref_pts = (
            deltas[..., :2] +
            inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid() # 参考点位置更新，参考点与x，y偏移量相加，得到新的参考点

        depth = depth.clamp(max=self.depth_limit)
        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None], # depth: (b v 1 h w)
                                           ref_pts.unsqueeze(2) * 2 - 1) # (b v 1 n 2) 根据参考点对深度图进行采样，得到每个参考点的信息
        sample_depth = sample_depth[:, :, 0, 0, :, None] # (b v 1 1 n) -> (b v n 1)
        points = torch.cat([
            ref_pts * torch.tensor(self.image_shape[::-1]).to(x), # todo 变换到网络输入坐标上
            sample_depth * (1 + deltas[..., 2:3])
        ], -1) # 计算3D点 (b,v,300,3)

        # todo ------------------------------------#
        # todo： 位置计算：cam2img cam2ego
        means3d = cam2world(points, cam2img, cam2ego, img_aug_mat) # 将2D图像坐标转换为3D世界坐标
        # 从高斯查询中，预测高斯属性：透明度、特征向量(代替SH)、缩放因子、旋转四元数
        opacities = self.opacity_head(x).float() # 不透明度、特征和尺度计算

        features = self.feature_head(x).float() # (b,v,300,dim)

        # todo ------------------------------------#
        # todo： 协方差计算：
        # todo 尺度：
        scales = self.scale_head(x) * self.scale_transform(
            sample_depth, cam2img[..., 0, 0]).clamp(1e-6)
        covariances = flatten_bsn_forward(get_covariance, scales,
                                          cam2ego[..., None, :3, :3]) # RS(RS)^T：协方差计算

        # todo ------------------------------------#
        # todo： 旋转计算
        rotations = flatten_bsn_forward(rotmat_to_quat, cam2ego[..., :3, :3])
        rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1) # 协方差和旋转矩阵


        means3d = rearrange(means3d,'b v g c -> b (v g) c') # (b (v g) 3)
        opacities = rearrange(opacities,'b v g c -> b (v g) c') # (b (v g) 1)
        features = rearrange(features,'b v g c -> b (v g) c') # (b (v g) n_cls)
        scales = rearrange(scales,'b v g c -> b (v g) c') # (b (v g) 3)
        rotations = rearrange(rotations,'b v g c -> b (v g) c') # (b (v g) 4)
        covariances = rearrange(covariances,'b v g i j -> b (v g) i j') # (b (v g) 3 3)

        # todo -------------------------------------#
        # K = torch.tensor([
        #     [2.5, 0.0, 100.0],
        #     [0.0, 2.5, 100.0],
        #     [0.0, 0.0, 1.0],
        # ])

        # R = torch.tensor([
        #     [1.0, 0.0,  0.0],
        #     [0.0, 1.0,  0.0],
        #     [0.0, 0.0, -1.0],
        # ])  # [3, 3]

        # z_centers = -1.0 + (torch.arange(16) + 0.5) * 0.4  # [16]

        # T = torch.eye(4).repeat(16, 1, 1)  # [16, 4, 4]
        # T[:, :3, :3] = R
        # T[:, 2, 3] = z_centers

        # T = T.to(means3d.device)
        # K = K.unsqueeze(0).repeat(16,1,1).to(means3d.device)
        # H, W = 200,200
        # near = -1.0
        # far  = 5.4

        # # 通过rasterization操作得到occ占用
        # grid_feats = []
        # for i in range(means3d.size(0)):

        #     means = means3d[i]
        #     rot =rotations[i]
        #     scale = scales[i]
        #     opa = opacities[i].squeeze(-1)
        #     feat = features[i]

        #     rendered_image = rasterization(
        #         means, # (n,3)
        #         rot, # (n,4)
        #         scale, # (n,3)
        #         opa, # (n)
        #         feat, # (n,c)
        #         T, # (v 4 4)
        #         K, # (v 3 3)
        #         width=H, # 192
        #         height=W, # 112
        #         near_plane=near,
        #         far_plane=far,
        #         render_mode='RGB'
        #         )[0]

        #     rendered_image = rearrange(rendered_image,'D H W C -> H W D C')
        #     grid_feats.append(rendered_image)
        # grid_feats = torch.stack(grid_feats, dim=0)


        # density, grid_feats = self.voxelizer(
        #     means3d=means3d.flatten(1, 2),
        #     opacities=opacities.flatten(1, 2),
        #     features=features.flatten(1, 2).softmax(-1), # (b n n_class)
        #     scales = scales.flatten(1, 2), # todo 增加的
        #     covariances=covariances.flatten(1, 2)) # 将离散的3D高斯分布转换为occ占据预测网格图



        density, grid_feats = self.voxelizer(
            means3d=means3d,
            opacities=opacities,
            features=features, # (b n n_class)
            scales = scales, # todo 增加的
            covariances=covariances) # 将离散的3D高斯分布转换为occ占据预测网格图

        if mode == 'predict':
            # # todo -----------------------------------#
            # # todo 占据预测 3.3 开放词汇占据预测
            # features = features @ self.text_proto_embeds # 查询特征与文本嵌入结合，帮助模型理解类别信息 (b,v,300,768) @ (768 21)
            # density, grid_feats = self.voxelizer(
            #     means3d=means3d.flatten(1, 2),
            #     opacities=opacities.flatten(1, 2),
            #     features=features.flatten(1, 2).softmax(-1), # (b n n_class)
            #     scales = scales.flatten(1, 2), # todo 增加的
            #     covariances=covariances.flatten(1, 2)) # 将离散的3D高斯分布转换为occ占据预测网格图

            # if self.prompt_denoising:
            #     probs = prompt_denoising(grid_feats)
            # else:
            #     probs = grid_feats.softmax(-1)
            # # todo -----------------------------------#
            # # todo merge_probs: 合并类别的概率：每个类别可能包含多个子类
            # probs = merge_probs(probs, OCC3D_CATEGORIES) # (b,x,y,z,n_txt_cls) (bs,x,y,z,n_cls)
            # preds = probs.argmax(-1) # (bs,h,w,16)
            # preds += (preds > 10) * 1 + 1  # skip two classes of "others"
            # # preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17) # 密度过小，将其类别设置为17

            probs = grid_feats.softmax(-1)
            preds = probs.argmax(-1)

            return preds


        # # todo feats: 视觉基础模型提取的特征图
        # tgt_feats = feats.flatten(-2).mT # todo .mT: 对矩阵转置，即交换最后两个维度
        # if hasattr(self, 'projection'):
        #     tgt_feats = self.projection(tgt_feats)[0]
        # # 执行PCA(主成分分析)降维操作，对输入张量进行低秩近似，减少特征维度
        # u, s, v = torch.pca_lowrank(
        #     tgt_feats.flatten(0, 2).double(), q=self.reduce_dims, niter=4) # (b,v,(h w),c) -> v: (c,q) q：降维后的维度
        # tgt_feats = tgt_feats @ v.to(tgt_feats)
        # features = features @ v.to(features)
        # features = features.float()
        # # todo ---------------------------#
        # # gsplat 光栅化

        # rendered = rasterize_gaussians(
        #     means3d.flatten(1, 2), # (b,vx300,3)
        #     features.flatten(1, 2), # (b,vx300,128)
        #     opacities.squeeze(-1).flatten(1, 2),
        #     scales.flatten(1, 2),
        #     rotations.flatten(1, 2),
        #     cam2img,
        #     cam2ego,
        #     img_aug_mats=img_aug_mat,
        #     image_size=(900, 1600),
        #     near_plane=0.1,
        #     far_plane=100,
        #     render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
        #     channel_chunk=32).flatten(0, 1) # ((b v) c h w)


        # rendered_depth = rendered[:, -1] # todo ((b v) h w) 深度图
        # rendered = rendered[:, :-1] #  ((b v) c h w) -> ((b v) c-1 h w)


        # # todo ------------------------------#
        # # todo 损失计算 3.2 VFM对齐的自监督学习
        losses = {}
        # depth = torch.where(depth < self.depth_limit, depth,
        #                     1e-3).flatten(0, 1) # todo depth: 视觉基础模型提取的深度图
        # # todo 深度预测监督：结合尺度不变对数和L1损失
        # losses['loss_depth'] = self.depth_loss(rendered_depth, depth) # todo 深度图计算损失
        # losses['mae_depth'] = self.depth_loss(
        #     rendered_depth, depth, criterion='l1')

        # # Interpolating to high resolution for supervision can improve mIoU by 0.7
        # # compared to average pooling to low resolution.
        # bsn, c, h, w = rendered.shape
        # tgt_feats = tgt_feats.mT.reshape(bsn, c, h // self.patch_size,
        #                                  w // self.patch_size)
        # tgt_feats = F.interpolate(
        #     tgt_feats, scale_factor=self.patch_size, mode='bilinear')
        # rendered = rendered.flatten(2).mT
        # tgt_feats = tgt_feats.flatten(2).mT.flatten(0, 1)
        # # todo 通过高斯点积进行渲染监督：
        # losses['loss_cosine'] = F.cosine_embedding_loss(
        #     rendered.flatten(0, 1), tgt_feats, torch.ones_like(
        #         tgt_feats[:, 0])) * 5

        # # --------------------#
        # if self.segment_head: # (optional) 分割损失
        #     losses['loss_ce'] = F.cross_entropy(
        #         self.segment_head(rendered).mT,
        #         sem_segs.flatten(0, 1).flatten(1).long(),
        #         ignore_index=0)

        probs = rearrange(grid_feats,"b H W D C -> b C (H W D)")
        target = rearrange(occ_gts,"b H W D ->b (H W D)").long()
        losses['loss_ce'] = 10.0 * CE_ssc_loss(probs, target, self.class_weights.type_as(probs), ignore_index=255)

        inputs = torch.softmax(probs, dim=1).transpose(1,2).flatten(0,1)
        target = occ_gts.flatten()
        ignore = self.num_classes - 1
        valid = (target != ignore)
        probas = inputs[valid]
        labels = target[valid] # todo 前景点的数量是很稀疏的
        losses['loss_lovasz'] = 1.0 * lovasz_softmax_flat(probas,labels)

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
