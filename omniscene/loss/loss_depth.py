import torch
from einops import reduce
from mmdet3d.registry import MODELS






@MODELS.register_module()
class LossDepth:
    def __init__(self,weight=0.25,sigma_image=None,use_second_derivative=False):
        self.weight = weight
        self.sigma_image = sigma_image
        self.use_second_derivative = use_second_derivative
    def forward(
        self,
        pred_depth,
        gt_imgs,
        near,
        far,
    ):
        # Scale the depth between the near and far planes.
        near = near[..., None, None].log()
        far = far[..., None, None].log()
        depth = pred_depth.minimum(far).maximum(near)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        # If desired, add bilateral filtering.
        if self.sigma_image is not None:
            color_gt = gt_imgs
            color_dx = reduce(color_gt.diff(dim=-1), "b v c h w -> b v h w", "max")
            color_dy = reduce(color_gt.diff(dim=-2), "b v c h w -> b v h w", "max")
            if self.use_second_derivative:
                color_dx = color_dx[..., :, 1:].maximum(color_dx[..., :, :-1])
                color_dy = color_dy[..., 1:, :].maximum(color_dy[..., :-1, :])
            depth_dx = depth_dx * torch.exp(-color_dx * self.sigma_image)
            depth_dy = depth_dy * torch.exp(-color_dy * self.sigma_image)

        return self.weight * (depth_dx.abs().mean() + depth_dy.abs().mean())
