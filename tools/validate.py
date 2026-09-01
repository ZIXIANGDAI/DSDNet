import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsdnet.checkpoints import load_model_checkpoint
from dsdnet.config import load_config
from dsdnet.data import StreamingVideoNoduleDataset
from dsdnet.models import DSDNet
from dsdnet.training import evaluate_streaming


def main():
    parser = argparse.ArgumentParser(description="Validate DSDNet")
    parser.add_argument("--config", default=ROOT / "configs/stage2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    device = torch.device(config["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = DSDNet().to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    if config["data"].get("interval", 1) != 1:
        raise ValueError("Streaming validation currently requires data.interval=1")
    dataset = StreamingVideoNoduleDataset(config["data"]["root"], args.split)
    loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=config["training"].get("num_workers", 4),
    )
    for name, value in evaluate_streaming(model, loader, device).items():
        print(f"{name.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
