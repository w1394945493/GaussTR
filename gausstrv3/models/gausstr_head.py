import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange,repeat

from mmdet3d.registry import MODELS
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from .utils import flatten_bsn_forward
from ..geometry import get_world_rays
from .encoder.common.gaussians import build_covariance
from ..loss import lovasz_softmax_flat,CE_ssc_loss
from .decoder import rasterize_gaussians,render_cuda

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
        hidden_dim = hidden_dim or input_dim * 4
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

            if self.mode == 'normalize':
                x = F.normalize(x,dim=-1)

            if self.mode == 'softplus':
                x = F.softplus(x)

            if self.range is not None:
                x = self.range[0] + (self.range[1] - self.range[0]) * x
        return x

@MODELS.register_module()
class GaussTRHead(BaseModule):

    def __init__(self,
                 regress_head,
                 opacity_head,
                 scale_head,
                 rot_head,
                 semantic_head,
                 rgb_head,
                 voxelizer,
                 near,
                 far,
                 use_sh = True,
                 background_color=[0.0, 0.0, 0.0],
                 num_classes = 18,
                 balance_cls_weight = True,
                 manual_class_weight=[
                    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],

                 ):
        super().__init__()

        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.regress_head = MODELS.build(regress_head)
        self.opacity_head = MODELS.build(opacity_head)
        self.scale_head = MODELS.build(scale_head)
        self.rot_head = MODELS.build(rot_head)
        self.semantic_head = MODELS.build(semantic_head)
        self.rgb_head = MODELS.build(rgb_head)
        self.voxelizer = MODELS.build(voxelizer)

        self.near = near
        self.far = far

        self.use_sh = use_sh

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

    def forward(self,
                x,
                ref_pts,
                image_shape,
                depth,
                cam2img,
                cam2ego,
                img_aug_mat=None,
                occ_gts=None,
                mode='tensor',
                **kwargs):

        bs, n = cam2img.shape[:2]
        device = cam2img.device
        x = x.reshape(bs, n, *x.shape[1:]) # (b,v,300,256)
        deltas = self.regress_head(x)

        ref_pts = (deltas[..., :2] + inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid() # (b v 300 2)

        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None],ref_pts.unsqueeze(2) * 2 - 1) # (b v 1 1 300)
        sample_depth = sample_depth[:, :, 0, 0, :, None] # (b v 1 1 n) -> (b v n 1)

        coordinates = rearrange(ref_pts,'b v g xy -> b v g () () xy') # (b v g 1 1 2)

        intrinsics = cam2img[...,:3,:3] # (b v 3 3)
        extrinsics = cam2ego # (b v 4 4)

        extrinsics_ = rearrange(extrinsics, "b v i j -> b v () () () i j")
        intrinsics_ = rearrange(intrinsics, "b v i j -> b v () () () i j") # 归一化的内参

        origins, directions = get_world_rays(coordinates, extrinsics_, intrinsics_) # (b v g 1 1 3) (b v g 1 1 3)
        origins = rearrange(origins,"b v g srf spp c -> b v g (srf spp c)") # (b v g 3)
        directions = rearrange(directions,"b v g srf spp c -> b v g (srf spp c)") # (b v g 3)

        means = origins + directions * sample_depth # (b v g 3)
        opacities = self.opacity_head(x) # (b v g 1)
        scales = self.scale_head(x) # (b v g 3)
        rotations = self.rot_head(x) # (b v g 4) 四元数格式
        semantics =self.semantic_head(x) # (b v g n_cls) n_cls = 17
        rgbs = self.rgb_head(x) # (b v g 3)


        covariances = build_covariance(scales, rotations) # (b v g 3 3)
        c2w_rotations = rearrange(cam2ego[...,:3,:3],'b v i j -> b v () i j')
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # (b v g 3 3)

        means3d = rearrange(means,'b v g c -> b (v g) c') # (b (v g) 3)
        opacities = rearrange(opacities,'b v g c -> b (v g) c') # (b (v g) 1)
        semantics = rearrange(semantics,'b v g c -> b (v g) c') # (b (v g) n_cls)
        rgbs = rearrange(rgbs,'b v g c -> b (v g) c') # (b (v g) 3)
        scales = rearrange(scales,'b v g c -> b (v g) c') # (b (v g) 3)
        rotations = rearrange(rotations,'b v g c -> b (v g) c') # (b (v g) 4)
        covariances = rearrange(covariances,'b v g i j -> b (v g) i j') # (b (v g) 3 3)

        # h, w = image_shape

        # near = self.near
        # near = torch.full((bs,n),near).to(device)
        # far = self.far
        # far = torch.full((bs,n),far).to(device)
        # colors, rendered_depth = render_cuda(
        #     extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
        #     intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
        #     image_shape = (h,w),
        #     near=rearrange(near, "b v -> (b v)"),
        #     far=rearrange(far, "b v -> (b v)"),
        #     background_color=repeat(self.background_color, "c -> (b v) c", b=bs, v=n),

        #     gaussian_means=repeat(means3d, "b g xyz -> (b v) g xyz", v=n),
        #     gaussian_sh_coefficients=
        #         repeat(rgbs, "b g c d_sh -> (b v) g c d_sh", v=n) if self.use_sh else repeat(rgbs, "b g rgb -> (b v) g rgb ()", v=n),
        #     gaussian_opacities=repeat(opacities.squeeze(-1), "b g -> (b v) g", v=n),

        #     gaussian_scales=repeat(scales, "b g c -> (b v) g c", v=n),
        #     gaussian_rotations=repeat(rotations, "b g c -> (b v) g c", v=n),
        #     # gaussian_covariances=repeat(covariances, "b g i j -> (b v) g i j", v=n) if covariances is not None else None,
        #     scale_invariant = False,
        #     use_sh= self.use_sh,
        # )
        # colors = rearrange(colors,'(bs n) c h w -> bs n c h w',bs=bs) # (b v c h w)
        # rendered_depth = rearrange(rendered_depth,'(bs n) c h w -> bs n c h w',bs=bs).squeeze(2) # (b v h w)

        # colors, rendered_depth = rasterize_gaussians(
        #     extrinsics=cam2ego,
        #     intrinsics=cam2img[...,:3,:3],
        #     image_shape = (h,w),
        #     means3d=means3d,
        #     rotations=rotations,
        #     scales=scales,
        #     covariances=covariances,
        #     opacities=opacities.squeeze(-1),
        #     colors=rgbs, # (b n c d_sh)
        #     use_sh=self.use_sh,
        #     img_aug_mats=img_aug_mat,

        #     near_plane=self.near,
        #     far_plane=self.far,

        #     render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
        #     channel_chunk=32)

        # occ占用预测
        grid_density, grid_feats = self.voxelizer(
            means3d,
            opacities,
            semantics,
            covariances,
            scales,
            rotations,
        )
        if mode == 'predict':
            occ_preds = grid_feats.argmax(-1) # (b 200 200 16 18) -> (b 200 200 16)
            outputs = [{
                'occ_pred': occ_preds,
            }]
            return outputs

        losses = {}
        probs = rearrange(grid_feats,"b H W D C -> b C (H W D)")
        target = rearrange(occ_gts,"b H W D ->b (H W D)").long()
        losses['loss_ce'] = 1.0 * CE_ssc_loss(probs, target, self.class_weights.type_as(probs), ignore_index=255)

        inputs = torch.softmax(probs, dim=1).transpose(1,2).flatten(0,1)
        target = occ_gts.flatten()
        ignore = 17
        valid = (target != ignore)
        probas = inputs[valid]
        labels = target[valid]
        losses['loss_lovasz'] = 10.0 * lovasz_softmax_flat(probas,labels)

        return losses

