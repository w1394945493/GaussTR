import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
from mmdet3d.registry import MODELS


# todo -------------------------------------#
@MODELS.register_module()
class SparseGaussianHeadSpconv(nn.Module):
    def __init__(self, in_channels=128, out_channels=14):
        super().__init__()
        self.num_gaussian_parameters = out_channels
        # ME.MinkowskiConvolution -> spconv.SubMConv3d
        self.conv1 = spconv.SubMConv3d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            padding=1, # SubMConv 通常需要 padding 来保持原尺寸
            bias=True
        )
        self.act = nn.GELU() 
        self.conv2 = spconv.SubMConv3d(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
    
        self.init_weights()
    
    def forward(self, sparse_input: spconv.SparseConvTensor):
        x = self.conv1(sparse_input)
        x = x.replace_feature(self.act(x.features))
        x = self.conv2(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (spconv.SparseConv3d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, spconv.SparseBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, indice_key=None):
        super().__init__()
        if stride > 1:
            # 步长大于1，使用普通稀疏卷积进行下采样
            self.conv = spconv.SparseConv3d(
                in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, bias=False,
                indice_key=indice_key  # 关键点：下采样层记录索引
            )
        else:
            # 步长为1，使用子流形卷积保持稀疏性
            self.conv = spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, bias=False,
                indice_key=indice_key
            )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.replace_feature(self.relu(self.bn(out.features)))
        return out

class SparseUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, indice_key=None):
        super().__init__()
        
        #! InverseConv3d 必须指定 indice_key 来寻找对应的 Encoder 坐标
        self.upconv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size=kernel_size, bias=False,
            indice_key=indice_key
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.upconv(x)
        out = out.replace_feature(self.relu(self.bn(out.features)))
        return out
    

@MODELS.register_module()
class SparseUNetSpconv(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        # todo --- Encoder ---
        curr_ch = in_channels
        self.stage_channels = []
        for i in range(num_blocks):
            out_ch = 64 * (2 ** i)
            #! 为每一层指定唯一的 key，如 'cp1', 'cp2'...
            key = f"cp{i}"
            self.encoder_layers.append(SparseConvBlock(curr_ch, out_ch, stride=2, indice_key=key))
            self.stage_channels.append(out_ch)
            curr_ch = out_ch
        
        # todo --- Bottleneck ---
        bottleneck_ch = curr_ch * 2
        #! Bottleneck 的下采样也需要 key
        self.bottleneck_down = SparseConvBlock(curr_ch, bottleneck_ch, stride=2, indice_key="cp_bt")
        self.bottleneck_conv = SparseConvBlock(bottleneck_ch, bottleneck_ch, stride=1)
        
        # todo --- Decoder ---
        curr_decoder_ch = bottleneck_ch
        for i in range(num_blocks):
            enc_ch = self.stage_channels[-(i+1)]
            
            #! 这里的逻辑是：Decoder 的第 0 层对应 Bottleneck 的下采样 key
            if i == 0:
                up_key = "cp_bt"
            else:
                up_key = f"cp{num_blocks - i}"            
            
            self.decoder_layers.append(nn.ModuleDict({
                'up': SparseUpConvBlock(curr_decoder_ch, enc_ch, indice_key=up_key),
                'fuse': spconv.SubMConv3d(enc_ch * 2, enc_ch, kernel_size=1, bias=False),
                'norm': nn.BatchNorm1d(enc_ch),
                'relu': nn.ReLU(inplace=True)
            }))
            curr_decoder_ch = enc_ch

        # todo --- Final Head ---
        #! 最后一层上采样对应 Encoder 的第一层 (cp0)
        self.final_up = SparseUpConvBlock(curr_decoder_ch, out_channels, indice_key="cp0")

    def forward(self, x):
        # todo x: spconv.SparseConvTensor
        enc_outputs = []

        # todo Encoder path
        for layer in self.encoder_layers:
            x = layer(x)
            enc_outputs.append(x)
        # todo Bottleneck
        x = self.bottleneck_down(x)
        x = self.bottleneck_conv(x)
        
        # todo Decoder path
        for i, layer in enumerate(self.decoder_layers):
            x = layer['up'](x)
            skip_x = enc_outputs[-(i+1)]
            
            features_cat = torch.cat([x.features, skip_x.features], dim=1)
            x = x.replace_feature(features_cat)
            
            x = layer['fuse'](x)
            x = x.replace_feature(layer['relu'](layer['norm'](x.features)))

        return self.final_up(x)