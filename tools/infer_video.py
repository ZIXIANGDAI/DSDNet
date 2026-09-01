import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsdnet.checkpoints import load_model_checkpoint
from dsdnet.config import load_config
from dsdnet.models import DSDNet


def preprocess(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Infer a consecutive-frame video folder")
    parser.add_argument("--config", default=ROOT / "configs/stage2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = DSDNet().to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.eval()
    paths = sorted(
        Path(args.input_dir).glob("*.png"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.reset_stream()
    with torch.no_grad():
        for path in paths:
            logits = model.stream_step(preprocess(path).to(device))
            if logits is None:
                continue
            mask = ((torch.sigmoid(logits[0, 0]) > 0.5).byte() * 255).cpu().numpy()
            Image.fromarray(mask).save(output_dir / path.name)


if __name__ == "__main__":
    main()
