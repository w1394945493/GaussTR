# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from mmseg.ops import resize
from mmseg.models.utils import resize

from mmdet3d.registry import MODELS
# @HEADS.register_module()

@MODELS.register_module()
class BNHead(nn.Module):
    """Just a batchnorm."""

    def __init__(self,
                 in_channels,
                 in_index=[0, 1, 2, 3],
                 input_transform='resize_concat',
                 channels=1536,
                 num_classes=21,
                 align_corners=False,
                 dropout_ratio=0.1,
                 resize_factors=None, **kwargs):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)

        self.out_channels = num_classes
        self.align_corners = align_corners

        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output