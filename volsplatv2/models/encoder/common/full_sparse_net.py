import torch
import torch.nn as nn
import MinkowskiEngine as ME
from mmdet3d.registry import MODELS

class FullResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
                    ME.MinkowskiConvolution(in_channels, out_channels, kernel_size, dilation=dilation, dimension=3),
                    ME.MinkowskiBatchNorm(out_channels),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(out_channels, out_channels, kernel_size, dilation=1, dimension=3),
                    ME.MinkowskiBatchNorm(out_channels)
                )
        self.act = ME.MinkowskiReLU(inplace=True)    
            
    def forward(self, x: ME.SparseTensor):
        return self.act(self.conv(x))

class FullResDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            ME.MinkowskiConvolution(in_channels+skip_channels, out_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True)
        )
        
    def forward(self, x, skip_x):
        x = ME.cat(x, skip_x)
        return self.fusion(x)
    
@MODELS.register_module()
class FullResSparseUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4):
        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoder_channels = []
        
        current_ch = in_channels
        # 1. Encoder 阶段：分辨率不变，空洞率倍增
        for i in range(num_blocks):
            out_ch = 64 * (2 ** i)
            d = 2 ** i 
            self.encoders.append(FullResBlock(current_ch, out_ch, dilation=d))
            self.encoder_channels.append(out_ch)
            current_ch = out_ch

        bottleneck_in = current_ch
        bottleneck_out = bottleneck_in * 2
        self.bottleneck = FullResBlock(bottleneck_in, bottleneck_out, dilation=2**(num_blocks-1))

        current_decoder_ch = bottleneck_out
        for i in range(num_blocks):
            skip_ch = self.encoder_channels[-1-i]
            self.decoders.append(FullResDecoderBlock(current_decoder_ch, skip_ch, skip_ch))
            current_decoder_ch = skip_ch
            
        self.final_layer = ME.MinkowskiConvolution(current_decoder_ch,
                                                   out_channels, 
                                                   kernel_size=1, 
                                                   dimension=3)

    def forward(self, x: ME.SparseTensor):
        # 这里的 x.C 在整个 forward 过程中保持不变
        encoder_outputs = []
        
        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (Skip Connections)
        for i in range(len(self.decoders)):
            enc_idx = len(encoder_outputs) - 1 - i
            skip_x = encoder_outputs[enc_idx]
            x = self.decoders[i](x, skip_x)
            
        return self.final_layer(x)