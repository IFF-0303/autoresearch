"""Dataset utilities for binary semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SegDatasetConfig:
    num_samples: int
    image_size: int
    seed: int


class SyntheticSegDataset(Dataset):
    """Deterministic synthetic binary segmentation dataset.

    Produces grayscale images with one random shape per sample and a matching mask.
    """

    def __init__(self, cfg: SegDatasetConfig):
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.cfg.seed + idx)
        size = self.cfg.image_size

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )

        shape_type = int(torch.randint(0, 2, (1,), generator=g).item())
        if shape_type == 0:
            cx = float(torch.randint(size // 4, 3 * size // 4, (1,), generator=g).item())
            cy = float(torch.randint(size // 4, 3 * size // 4, (1,), generator=g).item())
            radius = float(torch.randint(max(3, size // 10), max(4, size // 4), (1,), generator=g).item())
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
        else:
            x1 = int(torch.randint(0, size // 2, (1,), generator=g).item())
            y1 = int(torch.randint(0, size // 2, (1,), generator=g).item())
            w = int(torch.randint(max(3, size // 8), max(4, size // 3), (1,), generator=g).item())
            h = int(torch.randint(max(3, size // 8), max(4, size // 3), (1,), generator=g).item())
            x2 = min(size, x1 + w)
            y2 = min(size, y1 + h)
            mask = torch.zeros((size, size), dtype=torch.bool)
            mask[y1:y2, x1:x2] = True

        mask_f = mask.float()
        noise = 0.15 * torch.randn((size, size), generator=g)
        image = 0.2 + 0.7 * mask_f + noise
        image = image.clamp(0.0, 1.0)

        return image.unsqueeze(0), mask_f.unsqueeze(0)
