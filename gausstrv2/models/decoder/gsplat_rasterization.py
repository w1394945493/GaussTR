import torch
from gsplat import rasterization

from gausstr.models.utils import unbatched_forward


@unbatched_forward
def rasterize_gaussians(means3d,
                        colors, # 颜色
                        opacities,
                        scales,
                        rotations,
                        cam2imgs,
                        cam2egos,
                        image_shape,
                        img_aug_mats=None,
                        **kwargs):

    # cam2world to world2cam
    R = cam2egos[:, :3, :3].mT
    T = -R @ cam2egos[:, :3, 3:4]
    viewmat = torch.zeros_like(cam2egos)
    viewmat[:, :3, :3] = R
    viewmat[:, :3, 3:] = T
    viewmat[:, 3, 3] = 1

    if cam2imgs.shape[-2:] == (4, 4):
        cam2imgs = cam2imgs[:, :3, :3]

    H,W = image_shape
    # Denormalize the intrinsics into standred format
    cam2imgs[:,0] = cam2imgs[:,0] * W
    cam2imgs[:,1] = cam2imgs[:,1] * H

    # from gsplat import rasterization
    rendered_image = rasterization(
        means3d, # (n,3)
        rotations, # (n,4)
        scales, # (n,3)
        opacities, # (n)
        colors, # (n,c)
        viewmat, # viewmat: world2cam变换矩阵
        cam2imgs, # (v 3 3)
        width=W, # 192
        height=H, # 112
        **kwargs)[0] # near_plane:0.1 far_plane:100 render_mode:RGB+D channel_chunk:32
    return rendered_image.permute(0, 3, 1, 2)
