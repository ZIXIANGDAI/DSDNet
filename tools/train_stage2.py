import argparse
import sys
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsdnet.checkpoints import save_model_checkpoint
from dsdnet.config import load_config
from dsdnet.data import StreamingVideoNoduleDataset, VideoNoduleDataset
from dsdnet.models import DSDNet
from dsdnet.training import create_logger, evaluate_streaming, set_seed, unpack_video_batch


def build_dataset(config, split, auxiliary=False):
    data = config["data"]
    return VideoNoduleDataset(
        data["root"], split, sequence_length=4, interval=data.get("interval", 1),
        core_labels_dir=data.get("core_labels_dir") if auxiliary else None,
        edge_labels_dir=data.get("edge_labels_dir") if auxiliary else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Stage II of DSDNet")
    parser.add_argument("--config", default=ROOT / "configs/stage2.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-auxiliary-supervision", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["training"]["save_dir"] = args.output_dir
    if args.no_auxiliary_supervision:
        config["training"]["auxiliary_supervision"] = False
        config["training"]["core_loss_weight"] = 0.0
        config["training"]["edge_loss_weight"] = 0.0
    training = config["training"]
    set_seed(training["seed"])
    device = torch.device(training.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = Path(training["save_dir"])
    logger = create_logger(output_dir, "dsdnet-stage2")
    use_auxiliary = bool(training.get("auxiliary_supervision", False))
    train_set = build_dataset(config, config["data"]["train_split"], use_auxiliary)
    val_set = StreamingVideoNoduleDataset(config["data"]["root"], config["data"]["val_split"])
    train_loader = DataLoader(
        train_set, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=training["val_batch_size"], shuffle=False,
        num_workers=training["num_workers"], pin_memory=True,
    )
    model = DSDNet(
        core_loss_weight=training.get("core_loss_weight", 0.2),
        edge_loss_weight=training.get("edge_loss_weight", 0.1),
        redundancy_loss_weight=training.get("redundancy_loss_weight", 0.04),
    ).to(device)
    model.load_stage1_initialization(training["stage1_init"])
    pretrained_parameters = list(model.encoder.parameters()) + list(model.grouped_semantic_heads.parameters())
    pretrained_ids = {id(parameter) for parameter in pretrained_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in pretrained_ids]
    optimizer = SGD(
        [
            {"params": pretrained_parameters, "lr": training["learning_rate"] * training["pretrained_lr_scale"]},
            {"params": other_parameters, "lr": training["learning_rate"]},
        ],
        momentum=training["momentum"], weight_decay=training["weight_decay"],
    )
    total_iterations = training.get("iterations")
    if total_iterations is None:
        total_iterations = training["epochs"] * len(train_loader)
    scheduler = LambdaLR(optimizer, lambda step: (1 - min(step / total_iterations, 1)) ** 0.9)
    best_dice = -1.0
    iteration = 0
    while iteration < total_iterations:
        model.train()
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            iteration += 1
            frames, labels, core_labels, edge_labels = unpack_video_batch(batch)
            frames = [frame.to(device, non_blocking=True) for frame in frames]
            labels = labels.to(device, non_blocking=True)
            core_labels = core_labels.to(device, non_blocking=True) if core_labels is not None else None
            edge_labels = edge_labels.to(device, non_blocking=True) if edge_labels is not None else None
            loss = model(
                frames, current_position=(iteration - 1) % 4, labels=labels,
                core_labels=core_labels, edge_labels=edge_labels,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if iteration % training["print_interval"] == 0:
                logger.info("Iter %d/%d loss=%.4f", iteration, total_iterations, loss.item())
            if iteration % training["validation_interval"] == 0:
                metrics = evaluate_streaming(model, val_loader, device)
                logger.info("Validation iter=%d %s", iteration, " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                save_model_checkpoint(output_dir / f"iter_{iteration:05d}.pth", model, iteration=iteration, **metrics)
                if metrics["dice"] > best_dice:
                    best_dice = metrics["dice"]
                    save_model_checkpoint(output_dir / "dsdnet_best.pth", model, iteration=iteration, **metrics)
                model.train()


if __name__ == "__main__":
    main()
