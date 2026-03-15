# autoresearch program (binary segmentation loop)

This repository is for **autonomous research**, not just model training scripts.

## Loop entrypoint

Use `autoresearch_loop.py` as the experiment loop entrypoint.

It executes:
1. Candidate mutation (small, controlled change)
2. Short-budget train + fixed evaluation
3. Metric comparison vs best
4. Keep/revert decision
5. Experiment memory update

## Controlled mutation surface

Allowed mutation files:
- `train.py`
- `model_zoo.py`
- `losses.py`
- `configs/short.json`

Locked files (must not change during candidate runs):
- `evaluate.py` (fixed metric logic)
- `configs/smoke.json` (plumbing baseline)

## Fixed budgets

- Smoke: `configs/smoke.json` (plumbing check)
- Short: `configs/short.json` (candidate comparison budget)

Keep/revert uses only short-budget metric.

## Fixed metric

- Metric name: `val_dice`
- Source of truth: `evaluate.py`
- Direction: **higher is better**

## Commands

Smoke check:

```bash
uv run train.py --config configs/smoke.json --output-dir runs/smoke
uv run evaluate.py --config configs/smoke.json --checkpoint runs/smoke/model.pt
```

Run one closed-loop iteration (includes baseline init if needed):

```bash
uv run autoresearch_loop.py --candidate lr_up_20pct
```

Run another candidate:

```bash
uv run autoresearch_loop.py --candidate unet_wider
```

## Experiment memory

Loop writes:
- `results/experiments.tsv` with columns:
  - timestamp
  - hypothesis
  - changed_files
  - command
  - metric
  - decision
  - best_metric
- `results/best.json` with persistent best metric state.
