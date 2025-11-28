import torch
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
        # self.global_step = 0
        # self.apply_after_step = 15000


    def forward(self,gt_imgs,pred_imgs):
        # self.global_step += 1
        # if self.global_step < self.apply_after_step:
        #      return torch.tensor(0, dtype=torch.float32, device=gt_imgs.device)

        loss = self.lpips.forward(pred_imgs,gt_imgs,normalize=True)
        return self.weight * loss.mean()
