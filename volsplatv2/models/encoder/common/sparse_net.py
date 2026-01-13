import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

from mmdet3d.registry import MODELS

@MODELS.register_module()
class SparseGaussianHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=14):

        super().__init__()
        
        self.num_gaussian_parameters = out_channels
        
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            dimension=3
        )
        self.act = ME.MinkowskiGELU()
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            dimension=3
        )
    
        self.init_weights()
    
    def forward(self, sparse_input: ME.SparseTensor):
  
        x = self.conv1(sparse_input)
        x = self.act(x)
        x = self.conv2(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                try:
                    ME.utils.kaiming_normal_(m.kernel,
                                                mode='fan_out',
                                                nonlinearity='relu')
                except Exception:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
             
                if hasattr(m, 'bias') and m.bias is not None:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except Exception:
                        pass

         
            elif isinstance(m, ME.MinkowskiBatchNorm):
                if hasattr(m, 'bn'):
                    try:
                        nn.init.constant_(m.bn.weight, 1)
                        nn.init.constant_(m.bn.bias, 0)
                    except Exception:
                        pass






class AttentionBlock(nn.Module):
    """基于Flash Attention的AttentionBlock"""
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}")
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = ME.MinkowskiBatchNorm(channels)
        
        self.qkv = ME.MinkowskiLinear(channels, channels * 3)
        self.proj_out = ME.MinkowskiLinear(channels, channels)

    def _attention(self, qkv: torch.Tensor):
        length, width = qkv.shape
        ch = width // (3 * self.num_heads) # todo self.num_heads: 1
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3)  # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1)  # (1, num_heads, length, ch)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                values = F.scaled_dot_product_attention(q, k, v)[0] # 注意力算子 注意力算子
        else:
            values = F.scaled_dot_product_attention(q, k, v)[0]
        
        values = values.permute(1, 0, 2).reshape(length, -1)
        return values

    def forward(self, x: ME.SparseTensor):
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm) # (n,d) -> (n,3d)
        
        feature_dense = self._attention(qkv.F) # 做一个全局的自注意力 (n,3d) -> (n,d)
        feature = ME.SparseTensor(
            features=feature_dense,
            coordinate_map_key=qkv.coordinate_map_key,
            coordinate_manager=qkv.coordinate_manager
        )
        
        output = self.proj_out(feature)
        return output + x 

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels, out_channels, # todo 输入/输出通道
            kernel_size=kernel_size,   # todo 卷积核大小：稀疏卷积仅处理有值(非零点)，空缺地方不占用计算资源
            stride=stride, # todo 步长 stride=2：下采样：将空间网格坐标除以2
            dimension=3 # todo 维度: 声明在3D空间进行操作
        ) # todo 注：不需要手动设置padding: 稀疏卷积中，以“点”为中心
        self.norm = ME.MinkowskiBatchNorm(out_channels)
        self.act = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x: ME.SparseTensor):
        return self.act(self.norm(self.conv(x))) # todo 输入x：应该是一个ME.SparseTensor: 主要包括两部分：坐标(N,4) bs,x,y,z Features：特征

class SparseUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        # todo ME.MinkowskiConvolutionTranspose: 稀疏转置卷积：主要用于上采样，把被压缩的低分辨率特征图放大回高分辨率
        self.upconv = ME.MinkowskiConvolutionTranspose( 
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, # todo stride=2：定义了输出网格相对于输入网格的放大倍数
            dimension=3
        ) # todo： 转置卷积回将一个输入的非空点，按照kernel_size扩散出一组新的坐标点
        self.norm = ME.MinkowskiBatchNorm(out_channels)
        self.act = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x: ME.SparseTensor):
        return self.act(self.norm(self.upconv(x)))

@MODELS.register_module()
class SparseUNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4, use_attention=False):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList() 
        self.use_attention = use_attention
        
        self.encoder_channels = []
        
        current_ch = in_channels
        for i in range(num_blocks):
            out_ch = 64 * (2 ** i)  
            self.encoders.append(SparseConvBlock(current_ch, out_ch, kernel_size=3, stride=2))
            self.encoder_channels.append(out_ch)  
            current_ch = out_ch

      
        bottleneck_in = current_ch
        bottleneck_out = bottleneck_in * 2
        
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(SparseConvBlock(bottleneck_in, bottleneck_out, kernel_size=3, stride=2))
        
        # todo -------------------------#
        # todo 是否引入注意力层
        if use_attention:
            self.bottleneck.append(AttentionBlock(bottleneck_out))
        
        self.bottleneck.append(SparseConvBlock(bottleneck_out, bottleneck_out, kernel_size=3, stride=1))

        current_decoder_ch = bottleneck_out
        
        for i in range(num_blocks):
            decoder_out = self.encoder_channels[-1-i]
            
            upconv = SparseUpConvBlock(current_decoder_ch, decoder_out, kernel_size=3, stride=2)
            
            after_cat = ME.MinkowskiConvolution(
                decoder_out + self.encoder_channels[-1-i], 
                decoder_out, 
                kernel_size=1, 
                stride=1, 
                dimension=3
            )
            
            self.decoder_blocks.append(nn.ModuleList([upconv, after_cat]))
            
            current_decoder_ch = decoder_out

        self.final_upsample = SparseUpConvBlock(
            self.encoder_channels[0], 
            out_channels,              
            kernel_size=3, 
            stride=2
        )

    def forward(self, x: ME.SparseTensor):
        encoder_outputs = []
        
        # todo 编码器层：稀疏卷积进行下采样，
        for encoder in self.encoders: # todo encoder: SparseConvBlock: 稀疏卷积
            x = encoder(x)
            encoder_outputs.append(x) # todo 保存每一层输出：分辨率降低(UNet网络前一部分模块)

        for layer in self.bottleneck: # todo 继续进行下采样
            x = layer(x)

        for i, decoder_block in enumerate(self.decoder_blocks): # todo UNet网络结构，上采样
            upconv, after_cat = decoder_block
            
            x = upconv(x) # todo 进行上采样
            
            enc_index = len(encoder_outputs) - i - 1
            if enc_index >= 0 and enc_index < len(encoder_outputs):
                x = ME.cat(x, encoder_outputs[enc_index])
                x = after_cat(x)
        
        output = self.final_upsample(x) # todo 最终上采样层，将特征图恢复到特定的目标分辨率
        del encoder_outputs
        
        return output
