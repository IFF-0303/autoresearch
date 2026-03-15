# autoresearch (image segmentation edition)

This repo is an **autoresearch loop for binary semantic segmentation**.

The goal is not just training a segmentation model. The goal is to keep the
research closure:

> candidate change -> short run -> fixed metric -> compare -> keep/revert -> memory

## Autoresearch loop entrypoint

- `autoresearch_loop.py`

This script performs one autonomous iteration and can be repeatedly called by an
agent or scheduler.

## Task setup

- Task: binary semantic segmentation
- Baseline model: minimal UNet (`model_zoo.py`)
- Dataset: deterministic synthetic dataset (`datasets.py`)
- Fixed metric: `val_dice` from `evaluate.py` (higher is better)

## Files and responsibilities

- `train.py`: training shell (mutable by agent)
- `evaluate.py`: fixed evaluation logic and metric output (locked)
- `losses.py`: mutable loss definitions
- `model_zoo.py`: mutable model definitions
- `configs/short.json`: mutable short-budget experiment config
- `configs/smoke.json`: locked smoke plumbing config
- `autoresearch_loop.py`: candidate-run-compare-keep/revert-memory controller

## Controlled mutation surface

The loop enforces allowed mutable files:
- `train.py`
- `model_zoo.py`
- `losses.py`
- `configs/short.json`

And enforces locked files unchanged during candidate runs:
- `evaluate.py`
- `configs/smoke.json`

## Budgets

- Smoke budget: `configs/smoke.json` (tiny data, 1 epoch)
- Short budget: `configs/short.json` (candidate comparison)

Only short-budget `val_dice` is used for keep/revert decisions.

## Quick start

```bash
uv sync

# smoke plumbing
uv run train.py --config configs/smoke.json --output-dir runs/smoke
uv run evaluate.py --config configs/smoke.json --checkpoint runs/smoke/model.pt

# one autoresearch loop iteration (baseline + candidate if needed)
uv run autoresearch_loop.py --candidate lr_up_20pct
```

## Experiment memory

The loop records:
- `results/experiments.tsv`: hypothesis, changed files, command, metric, decision, best metric
- `results/best.json`: persistent best score state

This makes it easy for agents to continue iterative segmentation research without
changing comparison standards.

## License

MIT
