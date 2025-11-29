import torch
import torch.nn.functional as F
from einops import repeat,rearrange
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from .cuda_splating import render_cuda
from .gsplat_rasterization import rasterize_gaussians

@MODELS.register_module()
class MonoSplatDecoder(BaseModule):

    def __init__(self,
                 loss_mae,
                 loss_lpips,
                 depth_limit=100.,
                 background_color=[0.0, 0.0, 0.0],
                 use_sh = True,
                 renderer_type: str = "vanilla"
    ):
        super().__init__()

        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.loss_mae = MODELS.build(loss_mae)
        self.loss_lpips = MODELS.build(loss_lpips)
        self.renderer_type = renderer_type
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg'))
        self.use_sh = use_sh

        self.depth_limit =depth_limit


    def forward(self,
                pixel_gaussians,
                rgb_gts,
                image_shape, # tuple(h w)
                near, # (b v)
                far,  # (b v)
                cam2img,
                cam2ego,
                img_aug_mat, # (b v 4 4)
                depth,
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
        if self.renderer_type == "vanilla":
            colors, rendered_depth = render_cuda(
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
            rendered_depth = rearrange(rendered_depth,'(bs n) c h w -> bs n c h w',bs=bs).squeeze(2) # (b v h w)
        else:

            colors, rendered_depth = rasterize_gaussians(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                image_shape = (h,w),
                means3d=means3d,
                rotations=rotations,
                scales=scales,
                opacities=opacities.squeeze(-1),
                colors=harmonics,
                use_sh=self.use_sh,
                img_aug_mat = img_aug_mat,
                near_plane=0.1,
                far_plane=100.,
                render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
                channel_chunk=32,
            )



        if mode == 'predict':
            outputs = [{
                'img_pred': colors,
                'depth_pred': rendered_depth,
                'img_gt': rgb_gts / 255.,
            }]
            return outputs

        losses = {}

        rendered_depth = rendered_depth.flatten(0,1)
        depth = depth.clamp(max=self.depth_limit).flatten(0,1)
        losses['loss_depth'] = self.depth_loss(rendered_depth, depth)

        rgb = colors.flatten(0,1)
        rgb_gt = rgb_gts.flatten(0,1) / 255.

        # temp0 = rgb_input[-1]

        # mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_input.device).view(3,1,1)
        # std  = torch.tensor([0.229, 0.224, 0.225], device=rgb_input.device).view(3,1,1)
        # temp1 = temp0 * std + mean

        # temp2 = rgb_gt[-1]

        losses['loss_mae'] = self.loss_mae.forward(rgb_gt,rgb)
        losses['loss_lpips'] = self.loss_lpips.forward(rgb_gt, rgb)

        return losses

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

