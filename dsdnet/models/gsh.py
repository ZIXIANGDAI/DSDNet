import torch.nn as nn


class AxialSpatialMLP(nn.Module):
    """Models global distributions independently along height and width."""

    def __init__(self, spatial_size=16, hidden_dim=32):
        super().__init__()
        self.spatial_size = spatial_size
        self.mlp_h = nn.Sequential(
            nn.Linear(spatial_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, spatial_size),
        )
        self.mlp_w = nn.Sequential(
            nn.Linear(spatial_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, spatial_size),
        )

    def forward(self, feature):
        if feature.shape[-2:] != (self.spatial_size, self.spatial_size):
            raise ValueError(
                f"GSH expects {self.spatial_size}x{self.spatial_size} features after pooling, "
                f"received {tuple(feature.shape[-2:])}"
            )
        height_feature = self.mlp_h(feature.transpose(2, 3)).transpose(2, 3)
        width_feature = self.mlp_w(feature)
        return height_feature + width_feature


class GroupedSemanticHead(nn.Module):
    """Height/Width MLP GSH used by both training stages."""

    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.axial_mlp = AxialSpatialMLP()
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature):
        feature = self.reduce(self.pool(feature))
        feature = self.fuse(self.axial_mlp(feature))
        return feature * self.gate(feature)


def build_grouped_semantic_heads(count=4):
    return nn.ModuleList([GroupedSemanticHead() for _ in range(count)])
