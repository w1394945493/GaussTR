import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class DepthUNet(BaseModule):

    def __init__(self,
                 down_block=None,
                 mid_block=None,
                 up_block=None,
                 patch_sizes=None,
                 out_embed_dims=[128, 256, 512, 512],
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        
        self.down_blocks = nn.ModuleList([])
        # in_channels = out_embed_dims[0] + 1 + 1 # concat pseudo depth and conf 
        in_channels = out_embed_dims[0] + 1 # todo concat pseudo depth (without conf)
        for i, out_embed_dim in enumerate(out_embed_dims):
            is_final_block = i == len(out_embed_dims) - 1
            patch_size = patch_sizes[i] if patch_sizes is not None else None
            down_block.update(kv_compress_ratio=patch_size)
            down_block.update(attention_head_dim=out_embed_dim // down_block["num_attention_heads"])
            down_block.update(in_channels=in_channels)
            down_block.update(out_channels=out_embed_dim)
            down_block.update(add_downsample=not is_final_block)
            if i == 0:
                down_block.update(resnet_groups=1)
            else:
                down_block.update(resnet_groups=32)
            in_channels = out_embed_dim
            down_block_module = MODELS.build(down_block)
            self.down_blocks.append(down_block_module)

        # build middle block
        mid_block.update(in_channels=out_embed_dims[-1])
        mid_block.update(out_channels=out_embed_dims[-1])
        mid_block.update(attention_head_dim=out_embed_dims[-1] // mid_block["num_attention_heads"])
        self.mid_block = MODELS.build(mid_block)

        # build upsample blocks
        reversed_out_embed_dims = out_embed_dims[::-1]
        reversed_patch_sizes = patch_sizes[::-1] if patch_sizes is not None else [None] * len(out_embed_dims)
        out_channels = reversed_out_embed_dims[0]
        self.up_blocks = nn.ModuleList([])
        prev_output_channel = out_channels
        for i, (out_embed_dim, patch_size) in enumerate(zip(reversed_out_embed_dims, reversed_patch_sizes)):
            out_channels = reversed_out_embed_dims[i]
            in_channels = reversed_out_embed_dims[i]
            is_final_block = i == len(reversed_out_embed_dims) - 1
            up_block.update(attention_head_dim=out_embed_dim // up_block["num_attention_heads"])
            up_block.update(kv_compress_ratio=patch_size)
            up_block.update(in_channels=in_channels)
            up_block.update(prev_output_channel=prev_output_channel)
            up_block.update(out_channels=out_channels)
            up_block.update(add_upsample=not is_final_block)
            up_block_module = MODELS.build(up_block)
            self.up_blocks.append(up_block_module)
            prev_output_channel = out_channels
        
        self.feature_norm = nn.GroupNorm(num_channels=out_embed_dims[0], num_groups=32, eps=1e-6) 
        # todo group norm(组归一化)：稳定特征分布，加快收敛；BatchNorm效果依赖于batch size，
    def forward(self,
                img_feats,
                depths_in,
                status="train"):
        
        depths_in = rearrange(depths_in, "b v h w -> (b v) () h w")
        # img_feats = torch.cat([img_feats, depths_in / 20.0], dim=1) 
        img_feats = torch.cat([img_feats, depths_in], dim=1) # todo (01.20)
        
        # todo-----------------------------#
        # todo 3. U-Net主体(多视图U-Net，见论文3.2节)
        # downsample
        sample = img_feats # (bs*6,128+1,h,w) 特征维度128+1维深度+1维深度置信度
        down_block_res_samples = (sample,)
        for block_id, down_block in enumerate(self.down_blocks): # todo 4 层下采样层
            if self.use_checkpoint and status != "test":
                sample, res_samples = torch.utils.checkpoint.checkpoint(
                    down_block, sample, use_reentrant=False)
            else:
                sample, res_samples = down_block(sample)
            down_block_res_samples += res_samples
        
        # middile
        sample = self.mid_block(sample)
        
        # upsample
        for block_id, up_block in enumerate(self.up_blocks): # todo 4 层上采样层
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            if self.use_checkpoint and status != "test":
                input_vars = (sample, res_samples)
                sample = torch.utils.checkpoint.checkpoint(
                    up_block, *input_vars, use_reentrant=False
                )
            else:
                sample = up_block(sample, res_samples)

        # todo 做了一个GruopNorm
        features = self.feature_norm(sample) # todo：features：得到每个像素位置的最终特征：用于解码高斯分布 (bs*6,128,h,w)
        return features

