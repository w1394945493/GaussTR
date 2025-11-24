import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from .position import PositionEmbeddingSine
from .geometry import coords_grid

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid(
        [
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device),
        ],
    )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255.0 - mean) / std
    img1 = (img1 / 255.0 - mean) / std

    return img0, img1


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b_new, c, h_new, w_new)
        )  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = (
            splits.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = (
            splits.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(new_b, c, num_splits * h, num_splits * w)
        )  # [B, C, H, W]

    return merge


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def mv_feature_add_position(features, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    assert features.dim() == 4  # [B*V, C, H, W]

    if attn_splits > 1:  # add position in splited window
        features_splits = split_feature(features, num_splits=attn_splits)
        position = pos_enc(features_splits)
        features_splits = features_splits + position
        features = merge_splits(features_splits, num_splits=attn_splits)
    else:
        position = pos_enc(features)
        features = features + position

    return features





def prepare_feat_proj_data_lists(features, intrinsics, extrinsics, num_reference_views, idx):
    b, v, c, h, w = features.shape
    idx = idx[:, :, 1:]  # remove the current view

    if extrinsics is not None:
        # extract warp poses
        idx_to_warp = repeat(idx, "b v m -> b v m fw fh", fw=4, fh=4) # [b, v, m, 1, 1]
        extrinsics_cur = repeat(extrinsics.clone().detach(), "b v fh fw -> b v m fh fw", m=num_reference_views)  # [b, v, 4, 4]
        poses_others = extrinsics_cur.gather(1, idx_to_warp)  # [b, v, m, 4, 4]     # 按照 idx 取出对应参考视角的外参 [b, v, m, 4, 4]
        poses_others_inv = torch.linalg.inv(poses_others)  # [b, v, m, 4, 4]    # 计算这些外参的逆矩阵，变成参考视角→世界的变换
        poses_cur = extrinsics.clone().detach().unsqueeze(2)  # [b, v, 1, 4, 4]
        poses_warp = poses_others_inv @ poses_cur  # [b, v, m, 4, 4]    # 计算参考视角到当前视角的相对变换 [b, v, m, 4, 4]
        poses_warp = rearrange(poses_warp, "b v m ... -> (b v) m ...")  # [bxv, m, 4, 4]
    else:
        poses_warp = None

    if features is not None:
        # extract warp features
        idx_to_warp = repeat(idx, "b v m -> b v m c h w", c=c, h=h, w=w) # [b, v, m, 1]
        features_cur = repeat(features, "b v c h w -> b v m c h w", m=num_reference_views)  # [b, v, m, c, h, w]
        feat_warp = features_cur.gather(1, idx_to_warp)  # [b, v, m, c, h, w]
        feat_warp = rearrange(feat_warp, "b v m c h w -> (b v) m c h w")  # [bxv, m, c, h, w]
    else:
        feat_warp = None

    if intrinsics is not None: # 内参
        # extract warp intrinsics
        intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
        intr_curr[:, :, 0, :] *= float(w)  # 乘以特征图宽度，缩放内参的第一行（fx, cx）
        intr_curr[:, :, 1, :] *= float(h) # 乘以特征图高度，缩放内参的第二行（fy, cy） 根据特征图大小调整内参，确保内参和特征图尺寸对应。
        idx_to_warp = repeat(idx, "b v m -> b v m fh fw", fh=3, fw=3) # [b, v, m, 1, 1]
        intr_curr = repeat(intr_curr, "b v fh fw -> b v m fh fw", m=num_reference_views)  # [b, v, m, 3, 3]
        intr_warp = intr_curr.gather(1, idx_to_warp)  # [b, v, m, 3, 3]
        intr_warp = rearrange(intr_warp, "b v m ... -> (b v) m ...")  # [bxv, m, 3, 3]
    else:
        intr_warp = None

    return feat_warp, intr_warp, poses_warp


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics, # 内参
    pose,  # 外参
    depth, # 候选深度
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size() # d: 候选深度数量
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates # todo 准备像素网格
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W] 构建每个像素的齐次坐标
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]  将像素反投影到相机坐标系
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W] 把点按旋转矩阵旋转，从
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W] # 加上相机平移量
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature