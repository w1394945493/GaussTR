import torch
from gsplat import rasterization
from math import isqrt
from gausstr.models.utils import unbatched_forward


@unbatched_forward
def rasterize_gaussians(
    extrinsics,
    intrinsics,
    image_shape,
    means3d,
    rotations,
    scales,
    covariances,
    opacities,
    colors, # 颜色
    use_sh = True,
    img_aug_mats=None,
    **kwargs):

    # cam2world to world2cam
    R = extrinsics[:, :3, :3].mT
    T = -R @ extrinsics[:, :3, 3:4]
    viewmat = torch.zeros_like(extrinsics)
    viewmat[:, :3, :3] = R
    viewmat[:, :3, 3:] = T
    viewmat[:, 3, 3] = 1

    if intrinsics.shape[-2:] == (4, 4):
        intrinsics = intrinsics[:, :3, :3]

    H,W = image_shape
    # Denormalize the intrinsics into standred format
    cam2imgs = intrinsics.clone()
    cam2imgs[:,0] = cam2imgs[:,0] * W
    cam2imgs[:,1] = cam2imgs[:,1] * H


    if use_sh:
        d_sh = colors.shape[-1]
        sh_degree = isqrt(d_sh) - 1
    else:
        sh_degree = None



    # from gsplat import rasterization
    rendered, alpha, _ = rasterization(
        means=means3d, # (n,3)
        quats=rotations, # (n,4)
        scales=scales, # (n,3)
        opacities=opacities, # (n)
        colors=colors, # (n,c)
        covars=covariances,
        viewmats=viewmat, # viewmat: world2cam变换矩阵
        Ks=cam2imgs, # (v 3 3)
        width=W, # 192
        height=H, # 112
        sh_degree=sh_degree, # None or int
        **kwargs) # near_plane:0.1 far_plane:100 render_mode:RGB+D channel_chunk:32

    rendered = rendered.permute(0, 3, 1, 2)

    rendered_image = rendered[:,:-1] # (v c h w)
    rendered_depth = rendered[:,-1] # (v h w)
    return rendered_image, rendered_depth
