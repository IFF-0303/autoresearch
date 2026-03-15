# autoresearch (image segmentation edition)

This repository keeps the original **autoresearch loop** idea, but the task is now
**binary semantic image segmentation** instead of LLM pretraining.

Core closed loop (unchanged in spirit):
1. Propose a code change.
2. Run a fixed-budget experiment.
3. Evaluate with a fixed validation metric.
4. Keep/revert based on metric comparison.

## What is in this version

- **Task**: binary semantic segmentation (foreground/background mask).
- **Baseline model**: small UNet (minimal dependencies, pure PyTorch).
- **Dataset**: deterministic synthetic segmentation dataset (so the full loop runs without external data).
- **Fixed metric**: validation Dice (`val_dice`, higher is better).
- **Budgets**:
  - `smoke`: tiny data + 1 epoch for plumbing checks.
  - `short`: short budget for keep/revert experiment comparisons.

## Key files

- `train.py` — segmentation training entrypoint.
- `evaluate.py` — segmentation evaluation entrypoint (separate from train).
- `datasets.py` — synthetic dataset + dataset config.
- `model_zoo.py` — UNet baseline.
- `losses.py` — combined BCE + Dice loss and Dice metric.
- `configs/smoke.json` — minimal smoke config.
- `configs/short.json` — short budget config for comparisons.
- `program.md` — instructions for running the autoresearch keep/revert loop.

## Quick start

```bash
# install deps
uv sync

# smoke run (train + eval)
uv run train.py --config configs/smoke.json --output-dir runs/smoke
uv run evaluate.py --config configs/smoke.json --checkpoint runs/smoke/model.pt

# short run (for candidate comparisons)
uv run train.py --config configs/short.json --output-dir runs/short
uv run evaluate.py --config configs/short.json --checkpoint runs/short/model.pt
```

The evaluator prints:

```text
---
val_dice: 0.xxxxxx
```

Use that value as the fixed comparison metric in autoresearch.

## Notes for future research automation

The current setup is intentionally minimal and easy to mutate by an agent. Typical next search axes:
- Loss weights / loss variants (`losses.py`, config weights).
- Optimizer and LR schedule (`train.py` optimizer block + config).
- Data augmentation (`datasets.py`).
- Decoder/channel width (`model_zoo.py`, `model.base_channels`).

## License

MIT
