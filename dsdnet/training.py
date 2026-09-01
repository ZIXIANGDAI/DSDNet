import logging
import random
from pathlib import Path

import numpy as np
import torch

from .metrics import BinarySegmentationMetrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(output_dir, name="dsdnet"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "training.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def unpack_video_batch(batch):
    if len(batch) == 2:
        frames, labels = batch
        return frames, labels, None, None
    if len(batch) == 4:
        return batch
    raise ValueError(f"Expected 2 or 4 batch items, received {len(batch)}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = BinarySegmentationMetrics()
    for index, batch in enumerate(loader):
        frames, labels, _, _ = unpack_video_batch(batch)
        frames = [frame.to(device, non_blocking=True) for frame in frames]
        logits = model(frames, current_position=index % 4)
        prediction = (torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy()
        metrics.update(prediction, labels.numpy())
    return metrics.compute()


@torch.no_grad()
def evaluate_streaming(model, loader, device):
    """Evaluate videos sequentially with the same cache used by deployment inference."""
    model.eval()
    metrics = BinarySegmentationMetrics()
    current_video = None
    for videos, frames, labels in loader:
        if frames.shape[0] != 1:
            raise ValueError("Streaming evaluation requires val_batch_size=1")
        video = videos[0]
        if video != current_video:
            model.reset_stream()
            current_video = video
        logits = model.stream_step(frames.to(device, non_blocking=True))
        if logits is None:
            continue
        prediction = (torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy()
        metrics.update(prediction, labels.numpy())
    return metrics.compute()
