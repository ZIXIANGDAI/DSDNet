from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _frame_id(path):
    return int(path.stem.rsplit("_", 1)[-1])


def _encode_mask(path, ignore_index=255):
    mask = np.asarray(Image.open(path), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    values = set(np.unique(mask).tolist())
    if values <= {0, 1}:
        encoded = mask
    else:
        encoded = np.full(mask.shape, ignore_index, dtype=np.uint8)
        encoded[mask == 0] = 0
        encoded[mask == 255] = 1
    return torch.from_numpy(encoded.astype(np.int64))


def _load_image(path):
    """Load an RGB image as a [0, 1] tensor without additional normalization."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


class _NoduleDatasetBase(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        self.images_root = self.root / "images" / split
        self.labels_root = self.root / "labels" / split
        if not self.images_root.is_dir() or not self.labels_root.is_dir():
            raise FileNotFoundError(f"Missing images/labels split under {self.root}")

    def image_path(self, video, frame):
        filename = frame if isinstance(frame, str) else f"{video}_{frame:06d}.png"
        return self.images_root / video / filename

    def label_path(self, video, frame, label_root=None):
        root = self.labels_root if label_root is None else label_root
        filename = frame if isinstance(frame, str) else f"{video}_{frame:06d}.png"
        return root / video / filename


class SingleFrameNoduleDataset(_NoduleDatasetBase):
    def __init__(self, root, split, **kwargs):
        super().__init__(root, split, **kwargs)
        self.samples = []
        for video_dir in sorted(path for path in self.images_root.iterdir() if path.is_dir()):
            self.samples.extend((video_dir.name, path.name) for path in sorted(video_dir.glob("*.png")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, frame = self.samples[index]
        return _load_image(self.image_path(video, frame)), _encode_mask(
            self.label_path(video, frame)
        )


class VideoNoduleDataset(_NoduleDatasetBase):
    def __init__(
        self,
        root,
        split,
        sequence_length=4,
        interval=1,
        core_labels_dir=None,
        edge_labels_dir=None,
        **kwargs,
    ):
        super().__init__(root, split, **kwargs)
        self.sequence_length = sequence_length
        self.interval = interval
        self.core_root = self.root / core_labels_dir / split if core_labels_dir else None
        self.edge_root = self.root / edge_labels_dir / split if edge_labels_dir else None
        self.samples = []
        for video_dir in sorted(path for path in self.images_root.iterdir() if path.is_dir()):
            frame_paths = sorted(video_dir.glob("*.png"), key=_frame_id)
            first_position = (sequence_length - 1) * interval
            for position in range(first_position, len(frame_paths)):
                history = [
                    frame_paths[position - interval * offset].name
                    for offset in reversed(range(sequence_length))
                ]
                self.samples.append((video_dir.name, frame_paths[position].name, history))
        if not self.samples:
            raise RuntimeError(f"No valid video sequences found in {self.images_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, current, history = self.samples[index]
        frames = [_load_image(self.image_path(video, frame)) for frame in history]
        items = [frames, _encode_mask(self.label_path(video, current))]
        if self.core_root is not None:
            items.append(_encode_mask(self.label_path(video, current, self.core_root)))
        if self.edge_root is not None:
            items.append(_encode_mask(self.label_path(video, current, self.edge_root)))
        return tuple(items)


class StreamingVideoNoduleDataset(_NoduleDatasetBase):
    """Frames in deterministic video order for stateful validation."""

    def __init__(self, root, split):
        super().__init__(root, split)
        self.samples = []
        for video_dir in sorted(path for path in self.images_root.iterdir() if path.is_dir()):
            frame_paths = sorted(video_dir.glob("*.png"), key=_frame_id)
            self.samples.extend((video_dir.name, path.name) for path in frame_paths)
        if not self.samples:
            raise RuntimeError(f"No video frames found in {self.images_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, frame = self.samples[index]
        return (
            video,
            _load_image(self.image_path(video, frame)),
            _encode_mask(self.label_path(video, frame)),
        )
