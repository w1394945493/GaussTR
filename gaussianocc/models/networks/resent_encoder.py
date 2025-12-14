
import torch
import torch.nn as nn
import torchvision




class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class Upsampling_4(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample):
        x_to_upsample = self.upsample(x_to_upsample)

        return self.conv(x_to_upsample)

class Encoder_res101(nn.Module):
    def __init__(self, network_type = '101',path = None):
        super().__init__()

        if network_type == '101':
            resnet = torchvision.models.resnet101(pretrained=True)

        elif network_type == '50':
            resnet = torchvision.models.resnet50(pretrained=True)

        elif network_type == '34':
            resnet = torchvision.models.resnet34(pretrained=True)

        else:
            print('please define the network size')
            exit()

        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.upsampling_layer = UpsamplingConcat(1536, 512)
        self.Upsampling_4 = Upsampling_4(512, 256)
        self.depth_layer = nn.Conv2d(256, 64, kernel_size=1, padding=0)

        if path is not None:
            print('loading the encoder pretrain from {}'.format(path))
            self.load_pretrain(path)

    def load_pretrain(self, pretrain_path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x1 = self.backbone(x) # 1/8 channel: 512
        x2 = self.layer3(x1) # 1/16  channel: 1024
        x3 = self.upsampling_layer(x2, x1) # 1/8  512
        x4 = self.Upsampling_4(x3) # 1/4
        x4 = self.depth_layer(x4)  # 1/4
        return [x4]