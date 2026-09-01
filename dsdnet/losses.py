import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceBCELoss(nn.Module):
    """Combined binary cross-entropy and foreground Dice loss."""

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        intersection = (probability * target).sum(1)
        dice = (2 * intersection + 1) / (probability.sum(1) + target.sum(1) + 1)
        return bce + 1 - dice.mean()


class InterHeadRedundancySuppressionLoss(nn.Module):
    """L_irs: mean absolute channel correlation across every GSH pair."""

    def forward(self, group_features):
        normalized = []
        for feature in group_features:
            feature = feature.flatten(2)
            feature = feature - feature.mean(2, keepdim=True)
            normalized.append(F.normalize(feature, p=2, dim=2, eps=1e-6))
        losses = []
        for index, feature in enumerate(normalized[:-1]):
            for other in normalized[index + 1 :]:
                losses.append((feature @ other.transpose(1, 2)).abs().mean())
        return torch.stack(losses).mean()


class RecompositionConsistencyLoss(nn.Module):
    """L_rc: pairwise MSE between predictions from cyclic recompositions."""

    def forward(self, branch_logits):
        losses = []
        for index, logits in enumerate(branch_logits[:-1]):
            for other in branch_logits[index + 1 :]:
                losses.append(F.mse_loss(logits, other))
        return torch.stack(losses).mean()
