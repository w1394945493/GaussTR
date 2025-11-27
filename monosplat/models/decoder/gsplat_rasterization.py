import torch
from gsplat import rasterization
from math import isqrt
from .utils import unbatched_forward


@unbatched_forward
def rasterize_gaussians(
    extrinsics,
    intrinsics,
    image_shape,
    means3d,
    rotations,
    scales,
    opacities,
    colors,
    use_sh = True,
    **kwargs
    ):

    # cam2world to world2cam
    R = extrinsics[:, :3, :3].mT
    T = -R @ extrinsics[:, :3, 3:4]
    viewmat = torch.zeros_like(extrinsics)
    viewmat[:, :3, :3] = R
    viewmat[:, :3, 3:] = T
    viewmat[:, 3, 3] = 1

    if intrinsics.shape[-2:] == (4, 4):
        intrinsics = intrinsics[:, :3, :3]

    H, W = image_shape
    # Denormalize the intrinsics into standred format
    cam2img = intrinsics.clone()
    cam2img[:,0] = cam2img[:,0] * W
    cam2img[:,1] = cam2img[:,1] * H

    sh_degree = None
    if use_sh:
        d_sh = colors.shape[-1]
        sh_degree = isqrt(d_sh) - 1


    # from gsplat import rasterization
    rendered, alpha, _ = rasterization(
        means3d,
        rotations,
        scales,
        opacities,
        colors.transpose(-2,-1) if sh_degree else colors,
        viewmat,
        cam2img,
        width=W,
        height=H,
        sh_degree=sh_degree,
        **kwargs)

    rendered_image = rendered[...,:-1] # (b h w c)
    rendered_depth = rendered[...,-1:] # (b h w c)
    return rendered_image.permute(0, 3, 1, 2),rendered_depth.permute(0,3,1,2) # (b c h w)
