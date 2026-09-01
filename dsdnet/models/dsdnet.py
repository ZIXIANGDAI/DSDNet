from pathlib import Path

import torch
import torch.nn as nn

from dsdnet.losses import BinaryDiceBCELoss, InterHeadRedundancySuppressionLoss

from .backbone import ResNet18Encoder
from .cpmm import CoreEdgePriorGuidedModulationModule
from .decoder import LightweightDecoder
from .gtdmm import GroupedTemporalDependencyModelingModule


class DSDNet(nn.Module):
    """Paper model combining GTDMM for deep features and CPMM for shallow features."""

    def __init__(
        self,
        core_loss_weight=0.2,
        edge_loss_weight=0.1,
        redundancy_loss_weight=0.04,
    ):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.gtdmm = GroupedTemporalDependencyModelingModule(4)
        self.decoder = LightweightDecoder()
        self.cpmm = CoreEdgePriorGuidedModulationModule()
        self.segmentation_loss = BinaryDiceBCELoss()
        self.redundancy_loss = InterHeadRedundancySuppressionLoss()
        self.core_loss_weight = core_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.redundancy_loss_weight = redundancy_loss_weight
        self.reset_stream()

    @property
    def grouped_semantic_heads(self):
        return self.gtdmm.grouped_semantic_heads

    def load_stage1_initialization(self, checkpoint_path):
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("stage2_init_state", checkpoint)
        encoder_state = state.get("encoder_state", state.get("backbone_state"))
        head_state = state.get("grouped_semantic_heads_state", state.get("phase_heads_state"))
        if encoder_state is None or head_state is None:
            raise ValueError("Stage-I checkpoint must contain encoder and GSH states")

        expected = self.encoder.state_dict()
        compatible = {key: value for key, value in encoder_state.items() if key in expected}
        missing = sorted(set(expected) - set(compatible))
        mismatched = sorted(key for key, value in compatible.items() if value.shape != expected[key].shape)
        if missing or mismatched:
            raise RuntimeError(f"Invalid Stage-I encoder state: missing={missing}, mismatched={mismatched}")
        self.encoder.load_state_dict(compatible, strict=True)
        self.grouped_semantic_heads.load_state_dict(head_state, strict=True)

    def reset_stream(self):
        """Clear cached group features before starting a new video stream."""
        self._stream_group_features = []
        self._stream_frame_index = 0
        self.gtdmm.last_head_indices = None

    def _decode(self, c4, current_features, output_size):
        b3 = self.decoder.up3(c4, current_features["c3"])
        b2 = self.decoder.up2(b3, current_features["c2"])
        core_prior, edge_prior = self.cpmm.generate_priors(current_features)
        b2 = self.cpmm.b2_modulation(b2, core_prior, edge_prior)
        b1 = self.decoder.up1(b2, current_features["c1"])
        b1 = self.cpmm.b1_modulation(b1, core_prior, edge_prior)
        logits = self.decoder.prediction(b1, output_size)
        return logits, core_prior, edge_prior

    def stream_step(self, frame):
        """Segment one frame while reusing group features cached from the previous three frames.

        The first three calls warm up the cache and return ``None``. Call
        :meth:`reset_stream` before processing a different video.
        """
        if self.training:
            raise RuntimeError("stream_step is inference-only; call model.eval() first")
        if frame.ndim != 4:
            raise ValueError("stream_step expects a B x C x H x W tensor")

        current_features = self.encoder(frame)
        head_index = self._stream_frame_index % self.gtdmm.group_count
        group_feature = self.gtdmm.extract_group_feature(current_features, head_index)
        if self._stream_group_features and self._stream_group_features[-1].shape != group_feature.shape:
            raise ValueError("Frame batch size and spatial resolution must remain constant within a stream")

        self._stream_group_features.append(group_feature)
        if len(self._stream_group_features) > self.gtdmm.group_count:
            self._stream_group_features.pop(0)
        self._stream_frame_index += 1

        if len(self._stream_group_features) < self.gtdmm.group_count:
            return None

        self.gtdmm.last_head_indices = tuple(
            (self._stream_frame_index - self.gtdmm.group_count + offset) % self.gtdmm.group_count
            for offset in range(self.gtdmm.group_count)
        )
        c4 = self.gtdmm.fuse_group_features(self._stream_group_features)
        logits, _, _ = self._decode(c4, current_features, frame.shape[-2:])
        return logits

    def forward(
        self,
        frames,
        current_position=0,
        labels=None,
        core_labels=None,
        edge_labels=None,
    ):
        if len(frames) != 4:
            raise ValueError("DSDNet expects four consecutive frames")
        output_size = frames[-1].shape[-2:]
        encoded_frames = [self.encoder(frame) for frame in frames]
        c4, group_features = self.gtdmm(encoded_frames, current_position)
        current_features = encoded_frames[-1]
        logits, core_prior, edge_prior = self._decode(c4, current_features, output_size)

        if labels is None:
            return logits
        target = (labels == 1).float().unsqueeze(1)
        loss = self.segmentation_loss(logits, target)
        loss = loss + self.redundancy_loss_weight * self.redundancy_loss(group_features)
        if core_labels is not None:
            core_logits = self.cpmm.core_auxiliary_head(core_prior, output_size)
            loss = loss + self.core_loss_weight * self.segmentation_loss(
                core_logits, (core_labels == 1).float().unsqueeze(1)
            )
        if edge_labels is not None:
            edge_logits = self.cpmm.edge_auxiliary_head(edge_prior, output_size)
            loss = loss + self.edge_loss_weight * self.segmentation_loss(
                edge_logits, (edge_labels == 1).float().unsqueeze(1)
            )
        return loss
