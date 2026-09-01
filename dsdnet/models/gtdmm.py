import math

import torch
import torch.nn as nn

from .gsh import build_grouped_semantic_heads
from .selective_scan import (
    ChannelFirstLinear,
    DropPath,
    gilbert_2d,
    initialize_mamba,
    selective_scan,
)


class TemporalHilbertMamba(nn.Module):
    """THMamba: temporal selective scan following a Hilbert spatial traversal."""

    def __init__(self, channels=128, state_size=16, expansion=2.0):
        super().__init__()
        self.channels = channels
        self.state_size = state_size
        self.inner_channels = int(channels * expansion)
        self.dt_rank = math.ceil(channels / 16)
        self.in_proj = nn.Linear(channels, self.inner_channels, bias=False)
        self.x_proj = ChannelFirstLinear(
            self.inner_channels,
            self.dt_rank + 2 * state_size,
            bias=False,
        )
        self.dt_proj = ChannelFirstLinear(self.dt_rank, self.inner_channels, bias=False)
        self.out_norm = nn.LayerNorm(self.inner_channels)
        self.out_proj = ChannelFirstLinear(self.inner_channels, channels, bias=False)
        self.activation = nn.SiLU()
        self.a_logs, self.ds, dt_weight, self.dt_bias = initialize_mamba(
            state_size, self.dt_rank, self.inner_channels
        )
        self.dt_proj.weight.data.copy_(dt_weight[0])
        self._index_cache = {}

    def _hilbert_index(self, height, width, device):
        key = (height, width, str(device))
        if key not in self._index_cache:
            coordinates = list(gilbert_2d(width, height))
            if height == width:
                coordinates = [(y, x) for x, y in coordinates]
            self._index_cache[key] = torch.tensor(
                [y * width + x for x, y in coordinates], device=device, dtype=torch.long
            )
        return self._index_cache[key]

    def forward(self, feature):
        batch, time, _, height, width = feature.shape
        spatial_count = height * width
        index = self._hilbert_index(height, width, feature.device)
        feature = feature.permute(0, 1, 3, 4, 2)
        feature = self.activation(self.in_proj(feature))
        feature = feature.reshape(batch, time, spatial_count, self.inner_channels)
        feature = feature.index_select(2, index).permute(0, 2, 1, 3)
        sequence = feature.reshape(batch, spatial_count * time, self.inner_channels).transpose(1, 2)
        projected = self.x_proj(sequence)
        delta, state_b, state_c = torch.split(
            projected, [self.dt_rank, self.state_size, self.state_size], dim=1
        )
        delta = self.dt_proj(delta)
        length = sequence.shape[-1]
        output = selective_scan(
            sequence.float(),
            delta.float(),
            -self.a_logs.float().exp(),
            state_b.view(batch, 1, self.state_size, length).float(),
            state_c.view(batch, 1, self.state_size, length).float(),
            self.ds.float(),
            self.dt_bias.reshape(-1).float(),
        )
        output = output.transpose(1, 2).reshape(batch, spatial_count, time, self.inner_channels)
        output = output.permute(0, 2, 3, 1)
        restored = torch.zeros_like(output)
        scatter_index = index.view(1, 1, 1, spatial_count).expand_as(output)
        restored.scatter_(-1, scatter_index, output)
        restored = restored.reshape(batch, time, self.inner_channels, height, width)
        restored = restored.permute(0, 1, 3, 4, 2)
        restored = self.out_norm(restored).permute(0, 1, 4, 2, 3)
        restored = self.out_proj(restored.reshape(batch * time, self.inner_channels, height * width))
        return restored.reshape(batch, time, self.channels, height, width)


class TemporalGroupFeatureIntegrationBlock(nn.Module):
    """TGFIB: THMamba modeling followed by gated group-feature reconstruction."""

    def __init__(self, channels=128, time_steps=4, drop_path=0.1):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.thmamba = TemporalHilbertMamba(channels)
        self.drop_path = DropPath(drop_path)
        total_channels = channels * time_steps
        self.mix = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, 1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(nn.Conv2d(total_channels, total_channels, 1), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, grouped_features):
        residual = grouped_features
        normalized = self.norm(grouped_features.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        fused = residual + self.drop_path(self.thmamba(normalized))
        batch, time, channels, height, width = fused.shape
        if time != self.time_steps or channels != self.channels:
            raise ValueError("Unexpected grouped feature shape for TGFIB")
        flattened = fused.reshape(batch, time * channels, height, width)
        mixed = self.mix(flattened)
        return self.refine(mixed * self.gate(flattened))


class GroupedTemporalDependencyModelingModule(nn.Module):
    """GTDMM: cyclic GSH extraction and temporal group-feature integration."""

    def __init__(self, group_count=4):
        super().__init__()
        self.group_count = group_count
        self.grouped_semantic_heads = build_grouped_semantic_heads(group_count)
        self.tgfib = TemporalGroupFeatureIntegrationBlock(time_steps=group_count)
        self.last_head_indices = None

    def head_indices(self, current_position):
        current_position = int(current_position) % self.group_count
        return [
            (current_position - self.group_count + 1 + offset) % self.group_count
            for offset in range(self.group_count)
        ]

    def extract_group_feature(self, encoded_frame, head_index):
        head_index = int(head_index) % self.group_count
        return self.grouped_semantic_heads[head_index](encoded_frame["c3"])

    def fuse_group_features(self, group_features):
        if len(group_features) != self.group_count:
            raise ValueError(f"Expected {self.group_count} group features, received {len(group_features)}")
        return self.tgfib(torch.stack(group_features, dim=1))

    def forward(self, encoded_frames, current_position):
        if len(encoded_frames) != self.group_count:
            raise ValueError(f"Expected {self.group_count} encoded frames, received {len(encoded_frames)}")
        indices = self.head_indices(current_position)
        self.last_head_indices = tuple(indices)
        features = [
            self.extract_group_feature(encoded, index)
            for encoded, index in zip(encoded_frames, indices)
        ]
        return self.fuse_group_features(features), features
