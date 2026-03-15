"""Training entrypoint for binary semantic segmentation autoresearch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import SegDatasetConfig, SyntheticSegDataset
from losses import dice_score_from_logits, segmentation_loss
from model_zoo import build_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dice_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            dice_sum += dice_score_from_logits(logits, masks).item()
            count += 1
    return dice_sum / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--output-dir", default="runs/latest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg["seed"]))

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SyntheticSegDataset(
        SegDatasetConfig(
            num_samples=int(cfg["train_samples"]),
            image_size=int(cfg["image_size"]),
            seed=int(cfg["seed"]),
        )
    )
    val_ds = SyntheticSegDataset(
        SegDatasetConfig(
            num_samples=int(cfg["val_samples"]),
            image_size=int(cfg["image_size"]),
            seed=int(cfg["seed"]) + 10_000,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = build_model(cfg.get("model", {})).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    loss_cfg = cfg.get("loss", {})
    bce_w = float(loss_cfg.get("bce_weight", 0.5))
    dice_w = float(loss_cfg.get("dice_weight", 0.5))

    t0 = time.time()
    epochs = int(cfg["epochs"])
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = segmentation_loss(logits, masks, bce_weight=bce_w, dice_weight=dice_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))
        val_dice = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.6f} | val_dice: {val_dice:.6f}")

    ckpt_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    elapsed = time.time() - t0
    print("---")
    print(f"checkpoint:       {ckpt_path}")
    print(f"device:           {device}")
    print(f"training_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
