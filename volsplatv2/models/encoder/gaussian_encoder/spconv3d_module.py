import torch
import torch.nn as nn
import spconv.pytorch as spconv 
from mmdet3d.registry import MODELS

@MODELS.register_module()
class SparseConv3DModule(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        kernel_size=5,
    ):
        super().__init__()

        self.layer = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False)
        
        self.output_proj = nn.Linear(embed_channels, embed_channels)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float)) # todo [0.5 0.5 0.5]

    def forward(self, instance_feature, anchor): 
        bs, g, _ = instance_feature.shape # todo (b 25600 128)

        anchor_xyz = anchor[..., :3].flatten(0,1) # todo (25600 3)
        
        indices = anchor_xyz - self.pc_range[None, :3] # todo (25600 3)
        indices = indices / self.grid_size[None, :] # bg, 3
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
            bs, 1, 1).expand(-1, g, -1).flatten(0, 1), indices], dim=-1) # todo (25600 4)

        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        spatial_shape = spatial_shape.to(torch.int32) # todo (3)

        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), 
            indices=batched_indices, 
            spatial_shape=spatial_shape,
            batch_size=bs) # todo input.features: (25600 128)

        output = self.layer(input) # todo output.features: (25600 128)
        output = output.features.unflatten(0, (bs, g)) # todo (1 25600 128)

        return self.output_proj(output) # todo (1 25600 128)
