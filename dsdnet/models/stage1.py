import torch
import torch.nn as nn

from .backbone import ResNet18Encoder
from .decoder import Stage1PredictionHead
from .gsh import build_grouped_semantic_heads


class Stage1DSDNet(nn.Module):
    """Single-frame recomposable group representation learning network."""

    cycle_orders = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2))

    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.grouped_semantic_heads = build_grouped_semantic_heads(4)
        self.prediction_head = Stage1PredictionHead()

    def forward(self, image):
        output_size = image.shape[-2:]
        features = self.encoder(image)
        groups = [head(features["c3"]) for head in self.grouped_semantic_heads]
        branch_logits = []
        for order in self.cycle_orders:
            recomposed = torch.cat([groups[index] for index in order], dim=1)
            branch_logits.append(
                self.prediction_head({**features, "c4": recomposed}, output_size)
            )
        return {
            "logits": torch.stack(branch_logits).mean(0),
            "branch_logits": branch_logits,
            "group_features": groups,
        }
