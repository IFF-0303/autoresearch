"""Autoresearch loop runner for binary segmentation.

Implements a minimal closed loop:
- propose candidate mutation
- run fixed-budget short experiment
- evaluate fixed Dice metric
- compare against best
- keep/revert candidate changes
- log experiment memory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


SHORT_TRAIN_CMD = ["uv", "run", "train.py", "--config", "configs/short.json", "--output-dir", "runs/current"]
SHORT_EVAL_CMD = ["uv", "run", "evaluate.py", "--config", "configs/short.json", "--checkpoint", "runs/current/model.pt"]
METRIC_PATTERN = re.compile(r"^val_dice:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)
MUTABLE_FILES = {
    "train.py",
    "model_zoo.py",
    "losses.py",
    "configs/short.json",
}
LOCKED_FILES = {
    "evaluate.py",
    "configs/smoke.json",
}


@dataclass
class RunResult:
    metric: float
    command: str
    stdout: str


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def parse_metric(output: str) -> float:
    match = METRIC_PATTERN.search(output)
    if not match:
        raise ValueError(f"Could not parse val_dice from output:\n{output}")
    return float(match.group(1))


def run_short(output_dir: str = "runs/current") -> RunResult:
    train_cmd = SHORT_TRAIN_CMD.copy()
    train_cmd[-1] = output_dir
    eval_cmd = SHORT_EVAL_CMD.copy()
    eval_cmd[-1] = f"{output_dir}/model.pt"
    train_out = run_cmd(train_cmd)
    eval_out = run_cmd(eval_cmd)
    metric = parse_metric(eval_out)
    command = " && ".join([" ".join(train_cmd), " ".join(eval_cmd)])
    return RunResult(metric=metric, command=command, stdout=train_out + "\n" + eval_out)


def ensure_log_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["timestamp", "hypothesis", "changed_files", "command", "metric", "decision", "best_metric"])


def read_short_config() -> dict:
    with open("configs/short.json", "r", encoding="utf-8") as f:
        return json.load(f)


def write_short_config(cfg: dict) -> None:
    with open("configs/short.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def snapshot(paths: set[str], root: Path) -> dict[str, Path]:
    snap_dir = root / "snap"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_dir.mkdir(parents=True)
    snap_map: dict[str, Path] = {}
    for rel in sorted(paths):
        src = Path(rel)
        dst = snap_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        snap_map[rel] = dst
    return snap_map


def restore_snapshot(snap_map: dict[str, Path]) -> None:
    for rel, snap in snap_map.items():
        shutil.copy2(snap, rel)


def apply_candidate(candidate: str) -> tuple[str, list[str]]:
    if candidate == "lr_up_20pct":
        cfg = read_short_config()
        lr = float(cfg["learning_rate"])
        cfg["learning_rate"] = round(lr * 1.2, 8)
        write_short_config(cfg)
        return f"Increase short learning_rate from {lr} to {cfg['learning_rate']}", ["configs/short.json"]
    if candidate == "unet_wider":
        cfg = read_short_config()
        base = int(cfg["model"]["base_channels"])
        cfg["model"]["base_channels"] = base + 4
        write_short_config(cfg)
        return f"Increase UNet base_channels from {base} to {cfg['model']['base_channels']}", ["configs/short.json"]
    raise ValueError(f"Unknown candidate: {candidate}")


def enforce_mutation_surface(changed_files: list[str]) -> None:
    disallowed = [p for p in changed_files if p not in MUTABLE_FILES]
    if disallowed:
        raise ValueError(f"Candidate changed disallowed files: {disallowed}")


def ensure_locked_files_unchanged(before: dict[str, str]) -> None:
    for rel, prior in before.items():
        current = Path(rel).read_text(encoding="utf-8")
        if current != prior:
            raise ValueError(f"Locked evaluation/control file changed: {rel}")


def append_log(path: Path, hypothesis: str, changed_files: list[str], command: str, metric: float, decision: str, best_metric: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            hypothesis,
            ",".join(changed_files),
            command,
            f"{metric:.6f}",
            decision,
            f"{best_metric:.6f}",
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="lr_up_20pct", choices=["lr_up_20pct", "unet_wider"])
    parser.add_argument("--log", default="results/experiments.tsv")
    parser.add_argument("--state", default="results/best.json")
    parser.add_argument("--force-baseline", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    state_path = Path(args.state)
    ensure_log_header(log_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path("runs/.loop_tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    locked_before = {rel: Path(rel).read_text(encoding="utf-8") for rel in sorted(LOCKED_FILES)}

    if not state_path.exists() or args.force_baseline:
        baseline = run_short(output_dir="runs/baseline")
        state = {"best_metric": baseline.metric, "best_hypothesis": "baseline", "metric_name": "val_dice"}
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        append_log(log_path, "baseline", [], baseline.command, baseline.metric, "keep", baseline.metric)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))

    snap_map = snapshot(MUTABLE_FILES, tmp_root)
    try:
        hypothesis, changed_files = apply_candidate(args.candidate)
        enforce_mutation_surface(changed_files)
        ensure_locked_files_unchanged(locked_before)

        candidate = run_short(output_dir="runs/current")
        best_metric = float(state["best_metric"])
        if candidate.metric > best_metric:
            decision = "keep"
            best_metric = candidate.metric
            state["best_metric"] = best_metric
            state["best_hypothesis"] = hypothesis
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        else:
            decision = "revert"
            restore_snapshot(snap_map)

        append_log(log_path, hypothesis, changed_files, candidate.command, candidate.metric, decision, best_metric)
        print("---")
        print(f"hypothesis: {hypothesis}")
        print(f"candidate_metric: {candidate.metric:.6f}")
        print(f"best_metric: {best_metric:.6f}")
        print(f"decision: {decision}")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
