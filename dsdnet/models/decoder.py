import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpFusionBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(in_channels + out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, feature, skip):
        feature = F.interpolate(feature, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(torch.cat((feature, self.skip_proj(skip)), dim=1))


class LightweightDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = UpFusionBlock(512, 256, 256)
        self.up2 = UpFusionBlock(256, 128, 128)
        self.up1 = UpFusionBlock(128, 64, 64)
        self.refine = ConvBNReLU(64, 64)
        self.classifier = nn.Conv2d(64, 1, 1)

    def prediction(self, feature, output_size):
        logits = self.classifier(self.refine(feature))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=True)


class Stage1PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_c1 = ConvBNReLU(64, 64, 1, 0)
        self.proj_c2 = ConvBNReLU(128, 64, 1, 0)
        self.proj_c3 = ConvBNReLU(256, 64, 1, 0)
        self.proj_c4 = ConvBNReLU(512, 64, 1, 0)
        self.classifier = nn.Conv2d(256, 1, 1)

    def forward(self, features, output_size):
        projected = [
            F.interpolate(module(features[name]), output_size, mode="bilinear", align_corners=True)
            for module, name in zip(
                (self.proj_c1, self.proj_c2, self.proj_c3, self.proj_c4),
                ("c1", "c2", "c3", "c4"),
            )
        ]
        return self.classifier(torch.cat(projected, dim=1))
