# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast as autocast

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

def pos2posemb(pos, num_pos_feats=128, temperature=10000):
    with autocast(enabled=False):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_tmp = pos[..., None] / dim_t
        posemb = torch.stack((pos_tmp[..., 0::2].sin(), pos_tmp[..., 1::2].cos()), dim=-1).flatten(-2).flatten(-2)

    # print(posemb.shape)
    return posemb

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


class NerfPositionalEncoder(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=False, log_sampling=True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input

        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(0.0, num_encoding_functions - 1, num_encoding_functions)
        else:
            frequency_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions)

        self.register_buffer('frequency_bands', frequency_bands)

    def forward(self, tensor): # todo 位置编码：NeRF中的频率编码：将低维的原始数值映射到高维空间，让神经网络更容易学习到数据中的高频细节
        # tensor shape: [B, N, C]
        B, N, C = tensor.shape
        F = self.num_encoding_functions

        # 1. 计算乘法: [B, N, C, 1] * [F] -> [B, N, C, F] # todo (1 3600 13 1) * (6) -> (1 3600 13 6)
        x = tensor.unsqueeze(-1) * self.frequency_bands # todo self.frequency_bands: 预设的频率 将输入值乘以一系列预设的频率[1,2,4,8,16,32]  (1 3600 13 6)

        # 2. 计算 sin 和 cos: [B, N, C, F]
        sin_enc = torch.sin(x)
        cos_enc = torch.cos(x)

        # 3. 关键修正：为了匹配原始循环顺序 [freq, func, channel]
        # 原始顺序是：对每个 freq，先算所有通道的 sin，再算所有通道的 cos
        # 我们先 stack 得到 [B, N, C, F, 2] (2 分别是 sin 和 cos)
        enc = torch.stack([sin_enc, cos_enc], dim=-1)  # [B, N, C, F, 2] # todo (1 3600 13 6 2)

        # 调整维度顺序以匹配原始循环:
        # 原始循环: F -> func(2) -> C
        # 所以目标 shape 顺序应该是 [B, N, F, 2, C]
        enc = enc.permute(0, 1, 3, 4, 2).reshape(B, N, -1) # todo (1 3600 156)

        if self.include_input:
            return torch.cat([tensor, enc], dim=-1)
        return enc



