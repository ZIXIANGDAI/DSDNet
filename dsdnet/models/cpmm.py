import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ConvBNReLU


class CompetitiveSemanticFusion(nn.Module):
    def __init__(self, high_channels, low_channels, out_channels, reduction_channels):
        super().__init__()
        self.high_project = ConvBNReLU(high_channels, out_channels, 1, 0)
        self.low_project = ConvBNReLU(low_channels, out_channels, 1, 0)
        self.context = nn.Sequential(
            ConvBNReLU(out_channels, reduction_channels, 1, 0),
            ConvBNReLU(reduction_channels, out_channels, 1, 0),
        )
        self.high_logits = nn.Conv2d(out_channels, out_channels, 1)
        self.low_logits = nn.Conv2d(out_channels, out_channels, 1)
        self.refine = ConvBNReLU(out_channels, out_channels, 1, 0)

    def forward(self, high_feature, low_feature):
        high_feature = self.high_project(high_feature)
        high_feature = F.interpolate(high_feature, low_feature.shape[-2:], mode="bilinear", align_corners=True)
        low_feature = self.low_project(low_feature)
        descriptor = F.adaptive_avg_pool2d(high_feature + low_feature, 1)
        descriptor = self.context(descriptor)
        logits = torch.stack((self.high_logits(descriptor), self.low_logits(descriptor)), dim=2)
        weights = torch.softmax(logits, dim=2)
        return self.refine(weights[:, :, 0] * high_feature + weights[:, :, 1] * low_feature)


class ReflectWindowStd(nn.Module):
    def __init__(self, kernel_size=3, eps=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.eps = eps

    def forward(self, feature):
        padded = F.pad(feature, (self.padding,) * 4, mode="reflect")
        mean = F.avg_pool2d(padded, self.kernel_size, 1)
        squared = F.pad(feature.square(), (self.padding,) * 4, mode="reflect")
        mean_squared = F.avg_pool2d(squared, self.kernel_size, 1)
        return torch.sqrt(torch.clamp(mean_squared - mean.square(), min=self.eps))


class CorePriorEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.feature = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.Sigmoid())
        self.output = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, feature):
        local_mean = F.avg_pool2d(feature, 3, 1, 1)
        return self.output(self.feature(feature) * self.gate(local_mean))


class EdgePriorEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.feature = nn.Conv2d(channels, channels, 3, padding=1)
        self.window_std = ReflectWindowStd(3)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.Sigmoid())
        self.output = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, feature):
        return self.output(self.feature(feature) * self.gate(self.window_std(feature)))


class CorePriorGenerationBlock(nn.Module):
    """CPGB: competitive fusion followed by core-prior enhancement."""

    def __init__(self):
        super().__init__()
        self.fusion = CompetitiveSemanticFusion(256, 128, 128, 32)
        self.enhancement = CorePriorEnhancement(128)

    def forward(self, c3, c2):
        return self.enhancement(self.fusion(c3, c2))


class EdgePriorGenerationBlock(nn.Module):
    """EPGB: competitive fusion followed by edge-prior enhancement."""

    def __init__(self):
        super().__init__()
        self.fusion = CompetitiveSemanticFusion(128, 64, 64, 16)
        self.enhancement = EdgePriorEnhancement(64)

    def forward(self, c2, c1):
        return self.enhancement(self.fusion(c2, c1))


class _SplitModulationBase(nn.Module):
    """Split channel-attention and spatial-attention modulation."""

    def __init__(self, feature_channels, prior_channels):
        super().__init__()
        channel_count = feature_channels // 2
        spatial_count = feature_channels - channel_count
        self.split_sizes = (channel_count, spatial_count)
        self.prior_channel = nn.Conv2d(prior_channels, channel_count, 1, bias=False)
        self.prior_spatial = nn.Conv2d(prior_channels, spatial_count, 1, bias=False)
        self.channel_gate = nn.Conv2d(channel_count, channel_count, 1, bias=False)
        self.spatial_gate = nn.Conv2d(spatial_count, 1, 1, bias=False)
        self.channel_norm = nn.BatchNorm2d(channel_count)
        self.spatial_norm = nn.BatchNorm2d(spatial_count)
        self.activation = nn.Hardsigmoid()

    def forward(self, feature, prior):
        if prior.shape[-2:] != feature.shape[-2:]:
            prior = F.interpolate(prior, feature.shape[-2:], mode="bilinear", align_corners=True)
        channel_feature, spatial_feature = torch.split(feature, self.split_sizes, dim=1)
        channel_condition = channel_feature + self.prior_channel(prior)
        spatial_condition = spatial_feature + self.prior_spatial(prior)
        channel_weight = self.activation(self.channel_gate(F.adaptive_avg_pool2d(channel_condition, 1)))
        spatial_weight = self.activation(self.spatial_gate(spatial_condition))
        return torch.cat(
            (
                self.channel_norm(channel_feature * channel_weight),
                self.spatial_norm(spatial_feature * spatial_weight),
            ),
            dim=1,
        )


class CoreGuidedSplitModulation(_SplitModulationBase):
    """CSM: channel-spatial split modulation guided by the core prior."""


class EdgeGuidedSplitModulation(_SplitModulationBase):
    """ESM: channel-spatial split modulation guided by the edge prior."""


class ChannelSpatialSplitModulationBlock(nn.Module):
    """CSMB: core-guided modulation, pointwise MLP, then edge-guided modulation."""

    def __init__(self, feature_channels, core_channels=128, edge_channels=64):
        super().__init__()
        self.csm = CoreGuidedSplitModulation(feature_channels, core_channels)
        self.pointwise_mlp = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
        )
        self.esm = EdgeGuidedSplitModulation(feature_channels, edge_channels)

    def forward(self, feature, core_prior, edge_prior):
        identity = feature
        feature = self.csm(feature, core_prior)
        feature = self.pointwise_mlp(feature)
        feature = self.esm(feature, edge_prior)
        return feature + identity


class AuxiliaryPriorHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.head = nn.Sequential(ConvBNReLU(channels, channels), nn.Conv2d(channels, 1, 1))

    def forward(self, feature, output_size):
        return F.interpolate(self.head(feature), output_size, mode="bilinear", align_corners=True)


class CoreEdgePriorGuidedModulationModule(nn.Module):
    """CPMM containing CPGM, EPGM, and two decoder-level CSMBs."""

    def __init__(self):
        super().__init__()
        self.cpgb = CorePriorGenerationBlock()
        self.epgb = EdgePriorGenerationBlock()
        self.b2_modulation = ChannelSpatialSplitModulationBlock(128)
        self.b1_modulation = ChannelSpatialSplitModulationBlock(64)
        self.core_auxiliary_head = AuxiliaryPriorHead(128)
        self.edge_auxiliary_head = AuxiliaryPriorHead(64)

    def generate_priors(self, encoder_features):
        core = self.cpgb(encoder_features["c3"], encoder_features["c2"])
        edge = self.epgb(encoder_features["c2"], encoder_features["c1"])
        return core, edge
