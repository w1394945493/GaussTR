import torch

from math import isqrt
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer



from ...geometry.projection import get_fov, homogenize_points

def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result

def render_cuda(
    extrinsics,
    intrinsics,
    image_shape,
    near,
    far,
    background_color,

    gaussian_means,
    gaussian_sh_coefficients,
    gaussian_opacities,
    gaussian_scales=None,
    gaussian_rotations=None,
    gaussian_covariances=None,
    scale_invariant = True,
    use_sh = True,
):
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near # ((bv))
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        if gaussian_covariances is not None:
            gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    if use_sh:
        _, _, _, d_sh = gaussian_sh_coefficients.shape
        degree = isqrt(d_sh) - 1
    else:
        degree = 0

    shs = rearrange(gaussian_sh_coefficients, "b g c d_sh -> b g d_sh c").contiguous() # todo (bv g d_sh c) | (bv g 1 rgb)


    bsn, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    images = []
    depths = []
    for i in range(bsn):
        # Set up a tensor for the gradients of the screen-space means.
        # mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        # try:
        #     mean_gradients.retain_grad()
        # except Exception:
        #     pass

        means2D = torch.zeros_like(gaussian_means[i])
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],

            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        image, radii, depth, alpha = rasterizer(
            means3D=gaussian_means[i], # (N 3)
            # means2D=mean_gradients,
            means2D = means2D,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :], # (N 3)
            opacities=gaussian_opacities[i, ..., None], # (N 1)
            scales = gaussian_scales[i], # (N 3)
            rotations = gaussian_rotations[i], # (N 4)
            cov3D_precomp=gaussian_covariances[i, :, row, col] if gaussian_covariances is not None else None,
        )

        image = torch.clamp(image,min=0.0,max=1.0) # todo 参考Omni-Scene中的工作

        images.append(image)
        depths.append(depth)

    return torch.stack(images,dim=0), torch.stack(depths,dim=0)