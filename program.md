# autoresearch program (binary segmentation)

This program defines the autonomous experimentation loop for binary semantic segmentation.

## Setup

1. Create/use a run branch (e.g. `autoresearch/<tag>`).
2. Ensure dependencies are installed: `uv sync`.
3. Initialize `results.tsv` with header (tab-separated):

```
commit	val_dice	memory_gb	status	description
```

4. Baseline run (short budget):

```bash
uv run train.py --config configs/short.json --output-dir runs/baseline > run.log 2>&1
uv run evaluate.py --config configs/short.json --checkpoint runs/baseline/model.pt >> run.log 2>&1
```

## Rules

- The task is **binary semantic segmentation only**.
- Keep train/eval split:
  - `train.py`: train + save checkpoint.
  - `evaluate.py`: fixed validation Dice metric.
- The fixed comparison metric is `val_dice` (**higher is better**).
- Keep changes minimal and reviewable.

## Experiment loop

Repeat forever:

1. Record current commit as candidate base.
2. Make one experimental code change.
3. Commit the change.
4. Run short budget experiment:

```bash
uv run train.py --config configs/short.json --output-dir runs/current > run.log 2>&1
uv run evaluate.py --config configs/short.json --checkpoint runs/current/model.pt >> run.log 2>&1
```

5. Extract metric:

```bash
grep "^val_dice:" run.log
```

6. If command output is empty, mark as crash and inspect:

```bash
tail -n 80 run.log
```

7. Append one row to `results.tsv` with:
   - short commit hash
   - `val_dice` (0.000000 on crash)
   - memory_gb (0.0 if unavailable/crash)
   - `keep` / `discard` / `crash`
   - short description

8. Keep/revert logic:
   - If `val_dice` improves (higher), keep commit.
   - If `val_dice` is equal or worse, revert to previous best commit.

## Smoke check

Before larger edits, verify plumbing with smoke budget:

```bash
uv run train.py --config configs/smoke.json --output-dir runs/smoke
uv run evaluate.py --config configs/smoke.json --checkpoint runs/smoke/model.pt
```
