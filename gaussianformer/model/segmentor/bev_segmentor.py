from collections.abc import Iterable

import numpy as np
import torch

from mmdet3d.registry import MODELS

from .base_segmentor import CustomBaseSegmentor


from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@MODELS.register_module()
class BEVSegmentor(CustomBaseSegmentor):
    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

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

    #     return
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                mode = 'loss',
                **kwargs,):

        return