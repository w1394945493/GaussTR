import torch.nn as nn
from lpips import LPIPS
from mmdet3d.registry import MODELS

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

@MODELS.register_module()
class LossLpips(nn.Module):
    lpips: LPIPS
    def __init__(self,weight=0.05):
        super().__init__()
        self.weight = weight # todo 取0.05
        self.lpips = LPIPS(net="vgg")

        convert_to_buffer(self.lpips, persistent=False)

    def forward(self,gt_imgs,pred_imgs):
        loss = self.lpips.forward(pred_imgs,gt_imgs)
        return self.weight * loss.mean()
