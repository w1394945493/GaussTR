import torch.nn as nn
from lpips import LPIPS
from mmdet3d.registry import MODELS
from monosplat.misc.nn_module_tools import convert_to_buffer

@MODELS.register_module()
class LossLpips(nn.Module):
    lpips: LPIPS
    def __init__(self,weight=0.05):
        super().__init__()
        self.weight = weight
        self.lpips = LPIPS(net="vgg")
        # 把所有模型
        convert_to_buffer(self.lpips, persistent=False)

    def forward(self,gt_imgs,pred_imgs):
        loss = self.lpips.forward(pred_imgs,gt_imgs)
        return self.weight * loss.mean()
