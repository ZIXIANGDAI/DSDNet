import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsdnet.checkpoints import save_model_checkpoint
from dsdnet.config import load_config
from dsdnet.data import SingleFrameNoduleDataset
from dsdnet.losses import (
    BinaryDiceBCELoss,
    InterHeadRedundancySuppressionLoss,
    RecompositionConsistencyLoss,
)
from dsdnet.models import Stage1DSDNet
from dsdnet.training import create_logger, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train Stage I of DSDNet")
    parser.add_argument("--config", default=ROOT / "configs/stage1.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["training"]["save_dir"] = args.output_dir
    training = config["training"]
    loss_config = config["loss"]
    set_seed(training["seed"])
    device = torch.device(training.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = Path(training["save_dir"])
    logger = create_logger(output_dir, "dsdnet-stage1")

    train_set = SingleFrameNoduleDataset(config["data"]["root"], config["data"]["train_split"])
    val_set = SingleFrameNoduleDataset(config["data"]["root"], config["data"]["val_split"])
    train_loader = DataLoader(
        train_set, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=training["val_batch_size"], shuffle=False,
        num_workers=training["num_workers"], pin_memory=True,
    )
    model = Stage1DSDNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    segmentation_loss = BinaryDiceBCELoss()
    redundancy_loss = InterHeadRedundancySuppressionLoss()
    consistency_loss = RecompositionConsistencyLoss()
    best_dice = -1.0

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            target = (labels == 1).float().unsqueeze(1)
            loss = torch.stack([segmentation_loss(logits, target) for logits in output["branch_logits"]]).mean()
            loss = loss + loss_config["redundancy_weight"] * redundancy_loss(output["group_features"])
            loss = loss + loss_config["consistency_weight"] * consistency_loss(output["branch_logits"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        intersection = prediction_sum = target_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))["logits"]
                prediction = torch.sigmoid(logits) > 0.5
                target = (labels.to(device) == 1).unsqueeze(1)
                intersection += (prediction & target).sum().item()
                prediction_sum += prediction.sum().item()
                target_sum += target.sum().item()
        dice = 2 * intersection / max(prediction_sum + target_sum, 1)
        logger.info("Epoch %d/%d loss=%.4f val_dice=%.4f", epoch, training["epochs"], running_loss / len(train_loader), dice)
        if dice > best_dice:
            best_dice = dice
            save_model_checkpoint(output_dir / "stage1_best.pth", model, epoch=epoch, dice=dice)
            torch.save(
                {
                    "stage2_init_state": {
                        "encoder_state": model.encoder.state_dict(),
                        "grouped_semantic_heads_state": model.grouped_semantic_heads.state_dict(),
                    },
                    "epoch": epoch,
                    "dice": dice,
                },
                output_dir / "stage1_to_stage2_init.pth",
            )


if __name__ == "__main__":
    main()
