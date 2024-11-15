import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.efficientnet import get_efficientnet_image_size

class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True
    ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(init_channels)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False
            )
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, :, :]
    
class EffiSegNetBN(nn.Module):
    def __init__(
        self,
        classes=29,
        channel=64,
        pretrained=True,
        freeze_encoder=False,
        deep_supervision=False,
        encoder_name="efficientnet-b0"
    ):
        super(EffiSegNetBN, self).__init__()
        self.model_name = encoder_name
        self.encoder = EfficientNetBNFeatures(
            model_name=encoder_name,
            pretrained=pretrained
        )

        del self.encoder._avg_pooling
        del self.encoder._dropout
        del self.encoder._fc

        b = int(encoder_name[-1])

        num_channels_per_output = [
            (16, 24, 40, 112, 320),
            (16, 24, 40, 112, 320),
            (16, 24, 48, 120, 352),
            (24, 32, 48, 136, 384),
            (24, 32, 56, 160, 448),
            (24, 40, 64, 176, 512),
            (32, 40, 72, 200, 576),
            (32, 48, 80, 224, 640),
            (32, 56, 88, 248, 704),
            (72, 104, 176, 480, 1376),
        ]

        channels_per_output = num_channels_per_output[b]

        self.deep_supervision = deep_supervision

        self.efficient_size = get_efficientnet_image_size(encoder_name)
        self.target_size = (1024, 1024)
        self.up1 = nn.Upsample(size=self.efficient_size, mode="nearest")
        self.up2 = nn.Upsample(size=self.efficient_size, mode="nearest")
        self.up3 = nn.Upsample(size=self.efficient_size, mode="nearest")
        self.up4 = nn.Upsample(size=self.efficient_size, mode="nearest")
        self.up5 = nn.Upsample(size=self.efficient_size, mode="nearest")

        self.conv1 = nn.Conv2d(
            channels_per_output[0], channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(
            channels_per_output[1], channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = nn.Conv2d(
            channels_per_output[2], channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channel)

        self.conv4 = nn.Conv2d(
            channels_per_output[3], channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(channel)

        self.conv5 = nn.Conv2d(
            channels_per_output[4], channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(channel)

        self.relu = nn.ReLU(inplace=True)

        if self.deep_supervision:
            self.conv7 = nn.Conv2d(
                channel, classes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn7 = nn.BatchNorm2d(channel)
            self.conv8 = nn.Conv2d(
                channel, classes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn8 = nn.BatchNorm2d(channel)
            self.conv9 = nn.Conv2d(
                channel, classes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn9 = nn.BatchNorm2d(channel)
            self.conv10 = nn.Conv2d(
                channel, classes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn10 = nn.BatchNorm2d(channel)
            self.conv11 = nn.Conv2d(
                channel, classes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn11 = nn.BatchNorm2d(channel)

        self.bn6 = nn.BatchNorm2d(channel)
        self.ghost1 = GhostModule(channel, channel)
        self.ghost2 = GhostModule(channel, channel)

        self.conv6 = nn.Conv2d(channel, classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.interpolate(x, size=self.efficient_size, mode='bilinear', align_corners=True)
        x0, x1, x2, x3, x4 = self.encoder(x)

        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.bn1(x0)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.bn3(x2)

        x3 = self.conv4(x3)
        x3 = self.relu(x3)
        x3 = self.bn4(x3)
        
        x4 = self.conv5(x4)
        x4 = self.relu(x4)
        x4 = self.bn5(x4)

        x0 = self.up1(x0)
        x1 = self.up2(x1)
        x2 = self.up3(x2)
        x3 = self.up4(x3)
        x4 = self.up5(x4)

        x = x0 + x1 + x2 + x3 + x4
        x = self.bn6(x)
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.conv6(x)

        if self.deep_supervision:
            x0 = self.bn7(x0)
            x0 = self.conv7(x0)

            x1 = self.bn8(x1)
            x1 = self.conv8(x1)

            x2 = self.bn9(x2)
            x2 = self.conv9(x2)
            
            x3 = self.bn10(x3)
            x3 = self.conv10(x3)

            x4 = self.bn11(x4)
            x4 = self.conv11(x4)

            return x, [x0, x1, x2, x3, x4]
        
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        return x