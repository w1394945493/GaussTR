
from mmdet3d.registry import MODELS


@MODELS.register_module()
class LossMse:
    def __init__(self,weight = 1):
        self.weight = weight
    def forward(self,
                gt_imgs,
                pred_imgs):
        delta = gt_imgs - pred_imgs
        return self.weight * (delta**2).mean()