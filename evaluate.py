"""Evaluation entrypoint for binary semantic segmentation autoresearch."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from datasets import SegDatasetConfig, SyntheticSegDataset
from losses import dice_score_from_logits
from model_zoo import build_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


@torch.no_grad()
def evaluate_dice(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total += dice_score_from_logits(logits, masks).item()
        steps += 1
    return total / max(1, steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--checkpoint", default="runs/latest/model.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))

    val_ds = SyntheticSegDataset(
        SegDatasetConfig(
            num_samples=int(cfg["val_samples"]),
            image_size=int(cfg["image_size"]),
            seed=int(cfg["seed"]) + 10_000,
        )
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = build_model(cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_dice = evaluate_dice(model, val_loader, device)
    print("---")
    print(f"val_dice: {val_dice:.6f}")


if __name__ == "__main__":
    main()
