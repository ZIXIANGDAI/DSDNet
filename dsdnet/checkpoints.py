from pathlib import Path

import torch


def load_model_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    return checkpoint


def save_model_checkpoint(path, model, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, path)
