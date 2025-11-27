import torch
from einops import repeat,rearrange

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from .cuda_splating import render_cuda

@MODELS.register_module()
class MonoSplatDecoder(BaseModule):

    def __init__(self,
                 loss_lpips,
                 background_color=[0.0, 0.0, 0.0]):
        super().__init__()

        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.loss_lpips = MODELS.build(loss_lpips)


    def forward(self,
                pixel_gaussians,
                inputs,
                image_shape, # tuple(h w)
                near, # (b v)
                far,  # (b v)
                cam2img,
                cam2ego,
                img_aug_mat, # (b v 4 4)
                mode,
                **kwargs):
        bs, n = cam2img.shape[:2]
        # device = cam2img.device
        h,w = image_shape

        means3d = pixel_gaussians.means # (b n 3)
        harmonics = pixel_gaussians.harmonics # (b n c d_sh)
        opacities = pixel_gaussians.opacities # (b n)
        scales = pixel_gaussians.scales
        rotations = pixel_gaussians.rotations
        covariances = pixel_gaussians.covariances  # (b n 3 3)


        intrinsics = cam2img[...,:3,:3] # (b v 3 3)
        extrinsics = cam2ego # (b v 4 4)

        # 光栅化
        colors,depths,alphas = render_cuda(
            extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
            intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
            near=rearrange(near, "b v -> (b v)"),
            far=rearrange(far, "b v -> (b v)"),
            image_shape = (h,w),
            background_color=repeat(self.background_color, "c -> (b v) c", b=bs, v=n),

            gaussian_means=repeat(means3d, "b g xyz -> (b v) g xyz", v=n),
            gaussian_covariances=repeat(covariances, "b g i j -> (b v) g i j", v=n),
            gaussian_sh_coefficients=repeat(harmonics, "b g c d_sh -> (b v) g c d_sh", v=n),
            gaussian_opacities=repeat(opacities, "b g -> (b v) g", v=n),

        )

        colors = rearrange(colors,'(bs n) c h w -> bs n c h w',bs=bs) # (b v c h w)
        depths = rearrange(depths,'(bs n) c h w -> bs n c h w',bs=bs).squeeze(2) # (b v c h w)
        alphs = rearrange(alphas,'(bs n) c h w -> bs n c h w',bs=bs).squeeze(2)

        if mode == 'predict':
            outputs = [{
                'img_pred': colors,
                'depth_pred': depths,
                'img_gt': inputs,
            }]
            return outputs

        losses = {}

        rgb = colors.flatten(0,1)
        rgb_gt = inputs.flatten(0,1)
        losses['loss_mae'] = ((rgb - rgb_gt)**2).mean()
        losses['loss_lpips'] = self.loss_lpips(rgb,rgb_gt)

        return losses



