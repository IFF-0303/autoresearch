# autoresearch program (binary image segmentation)

This document defines the autonomous experimentation workflow for **binary semantic segmentation**.

## Setup

1. Work on the current local branch and ensure dependencies are installed:

```bash
uv sync
```

2. Initialize `results.tsv` with a tab-separated header:

```text
commit	val_dice	memory_gb	status	description
```

3. Keep baseline run commands available for short-budget comparison:

```bash
uv run train.py --config configs/short.json --output-dir runs/baseline > run.log 2>&1
uv run evaluate.py --config configs/short.json --checkpoint runs/baseline/model.pt >> run.log 2>&1
```

## Current task

- The task is no longer LLM/nanochat training.
- The current and only task is **binary semantic segmentation**.

## Fixed experiment loop

For each candidate change, follow this exact closed loop:

1. **propose change**
2. **run fixed-budget experiment**
3. **evaluate with fixed validation metric**
4. **keep/revert based on metric comparison**

Always apply one focused change per iteration so comparisons remain attributable.

## Fixed primary metric

- Use `val_dice` as the **only** primary comparison metric.
- Comparison rule: **higher is better**.
- Do not use other metrics as the main keep/revert criterion.

## Fixed budgets

- `smoke`: tiny data + 1 epoch (pipeline sanity check)
- `short`: short budget for candidate-to-candidate comparison

## Allowed files to modify

Only modify the following:

- `train.py`
- `losses.py`
- `model_zoo.py`
- Configuration files related to model, optimizer, loss, or augmentation

## Forbidden modifications

The following must remain unchanged:

- validation metric definition
- dataset split logic
- evaluation comparison logic
- keep/revert decision rule

## Experiment execution details

Repeat the loop below:

1. Record the current commit as the candidate base.
2. Make one experimental change.
3. Run the short-budget experiment:

```bash
uv run train.py --config configs/short.json --output-dir runs/current > run.log 2>&1
uv run evaluate.py --config configs/short.json --checkpoint runs/current/model.pt >> run.log 2>&1
```

4. Extract the metric:

```bash
grep "^val_dice:" run.log
```

5. If no metric is found, treat as crash and inspect logs:

```bash
tail -n 80 run.log
```

6. Append one row to `results.tsv` with:
   - short commit hash
   - `val_dice` (`0.000000` on crash)
   - memory usage in GB (`0.0` if unavailable)
   - `keep` / `discard` / `crash`
   - short description of the change

7. Apply keep/revert decision using only `val_dice`:
   - If `val_dice` improves, keep the change.
   - If `val_dice` is equal or worse, revert the change.

## Persistence policy for better configurations

When a candidate outperforms the current best (by `val_dice`):

- Do **not** create a GitHub branch or PR.
- Save a full local project snapshot to:
  - `snapshots/best_<metric>_<timestamp>/`
- Write:
  - `summary.txt` (change details, config, metric, conclusion)
  - `snapshots/index.md` (append/update snapshot index)

## Smoke check

Before larger edits, run the smoke budget:

```bash
uv run train.py --config configs/smoke.json --output-dir runs/smoke
uv run evaluate.py --config configs/smoke.json --checkpoint runs/smoke/model.pt
```

## Future search space

Prioritized axes for later exploration:

- loss variants / weights
- optimizer / learning-rate schedule
- augmentation
- decoder width / model capacity
