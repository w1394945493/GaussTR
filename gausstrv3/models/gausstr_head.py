import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from mmdet3d.registry import MODELS
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from .utils import flatten_bsn_forward
from ..geometry import get_world_rays
from .encoder.common.gaussians import build_covariance
from ..loss import *

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
                 voxelizer,
                 num_classes = 18,
                 balance_cls_weight = True,
                 manual_class_weight=[
                    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                    1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                    1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],

                 ):
        super().__init__()

        self.regress_head = MODELS.build(regress_head)
        self.opacity_head = MODELS.build(opacity_head)
        self.scale_head = MODELS.build(scale_head)
        self.rot_head = MODELS.build(rot_head)
        self.semantic_head = MODELS.build(semantic_head)
        self.voxelizer = MODELS.build(voxelizer)

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
                depth,
                cam2img,
                cam2ego,
                img_aug_mat=None,
                occ_gts=None,
                mode='tensor',
                **kwargs):

        bs, n = cam2img.shape[:2]
        x = x.reshape(bs, n, *x.shape[1:]) # (b,v,300,256)
        deltas = self.regress_head(x)

        ref_pts = (deltas[..., :2] + inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid() # (b v 300 2)

        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None],ref_pts.unsqueeze(2) * 2 - 1) # (b v 1 1 300)
        sample_depth = sample_depth[:, :, 0, 0, :, None] # (b v 1 1 n) -> (b v n 1)

        coordinates = rearrange(ref_pts,'b v g xy -> b v g () () xy') # (b v g 1 1 2)

        intrinsics = cam2img[...,:3,:3] # (b v 3 3)
        extrinsics = cam2ego # (b v 4 4)
        extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
        intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j") # 归一化的内参

        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics) # (b v g 1 1 3) (b v g 1 1 3)
        origins = rearrange(origins,"b v g srf spp c -> b v g (srf spp c)") # (b v g 3)
        directions = rearrange(directions,"b v g srf spp c -> b v g (srf spp c)") # (b v g 3)

        means = origins + directions * sample_depth # (b v g 3)
        opacities = self.opacity_head(x) # (b v g 1)
        scales = self.scale_head(x) # (b v g 3)
        rotations = self.rot_head(x) # (b v g 4) 四元数格式
        semantics =self.semantic_head(x) # (b v g n_cls) n_cls = 17

        covariances = build_covariance(scales, rotations) # (b v g 3 3)
        c2w_rotations = rearrange(cam2ego[...,:3,:3],'b v i j -> b v () i j')
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # (b v g 3 3)

        means = rearrange(means,'b v g c -> b (v g) c') # (b (v g) 3)
        opacities = rearrange(opacities,'b v g c -> b (v g) c') # (b (v g) 1)
        semantics = rearrange(semantics,'b v g c -> b (v g) c') # (b (v g) n_cls)
        scales = rearrange(scales,'b v g c -> b (v g) c') # (b (v g) 3)
        rotations = rearrange(rotations,'b v g c -> b (v g) c') # (b (v g) 4)
        covariances = rearrange(covariances,'b v g i j -> b (v g) i j') # (b (v g) 3 3)

        # occ占用预测
        grid_density, grid_feats = self.voxelizer(
            means,
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

        losses = {}
        probs = rearrange(grid_feats,"b H W D C -> b C (H W D)")
        target = rearrange(occ_gts,"b H W D ->b (H W D)").long()
        losses['loss_ce'] = 10.0 * CE_ssc_loss(probs, target, self.class_weights.type_as(probs), ignore_index=255)

        inputs = torch.softmax(probs, dim=1).transpose(1,2).flatten(0,1)
        target = occ_gts.flatten()
        ignore = 17
        valid = (target != ignore)
        probas = inputs[valid]
        labels = target[valid]
        losses['loss_lovasz'] = 1.0 * lovasz_softmax_flat(probas,labels)

        return losses

