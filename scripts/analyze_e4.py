"""Aggregate E4 raw_*.jsonl files into a leaderboard + plot.

Usage:
    python scripts/analyze_e4.py e4/results/

Reads every raw_*.jsonl in the dir; produces:
    - results/leaderboard.csv  (one row per (condition, k_steps, ar_model, diff_model))
    - results/plot_accuracy_vs_flops.pdf  (curves per condition)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["condition"],
            r.get("k_steps", 0),
            r.get("ar_model", "?"),
            r.get("diff_model", "?"),
            r.get("seed", 0),
        )
        by_cell[key].append(r)
    out = []
    for (cond, k, ar, diff, seed), runs in sorted(by_cell.items()):
        n = len(runs)
        n_correct = sum(int(r["correct"]) for r in runs)
        flops = [r["flops"] for r in runs]
        out.append(
            {
                "condition": cond,
                "k_steps": k,
                "ar_model": ar,
                "diff_model": diff,
                "seed": seed,
                "n": n,
                "accuracy": n_correct / n if n else 0.0,
                "flops_mean": sum(flops) / n if n else 0,
                "flops_total": sum(flops),
            }
        )
    return out


def write_csv(path: Path, leaderboard: list[dict[str, Any]]) -> None:
    if not leaderboard:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)


def plot_pdf(path: Path, leaderboard: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[analyze] matplotlib not installed; skipping plot.")
        return False

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in leaderboard:
        by_cond[row["condition"]].append(row)
    for cond, rows in by_cond.items():
        rows.sort(key=lambda r: r["flops_mean"])

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"c1": "o", "c2": "s", "c3": "^", "c4": "D"}
    for cond, rows in sorted(by_cond.items()):
        if not rows:
            continue
        x = [r["flops_mean"] for r in rows]
        y = [r["accuracy"] * 100 for r in rows]
        ax.plot(
            x,
            y,
            marker=markers.get(cond, "x"),
            label=f"{cond} (n={rows[0]['n']})",
            linewidth=1.5,
            markersize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per problem (mean)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Sfumato E4: AR vs diffusion vs hybrid CoT on GSM8K")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"[analyze] plot -> {path}")
    return True


def main(results_dir: str) -> int:
    base = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for raw in sorted(base.glob("raw_*.jsonl")):
        with raw.open() as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        print(f"[analyze] no raw_*.jsonl in {base}", file=sys.stderr)
        return 1
    leaderboard = aggregate(rows)

    print(
        f"{'cond':6s} {'k':4s} {'n':4s} {'acc%':6s} {'flops':10s}"
        f" {'ar_model':30s} {'diff_model':30s}"
    )
    for r in leaderboard:
        print(
            f"{r['condition']:6s} {r['k_steps']:<4d} {r['n']:<4d} "
            f"{r['accuracy']*100:5.1f}% {r['flops_mean']:.2e}  "
            f"{r['ar_model']:30s} {r['diff_model']:30s}"
        )

    write_csv(base / "leaderboard.csv", leaderboard)
    plot_pdf(base / "plot_accuracy_vs_flops.pdf", leaderboard)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["e4/results"]
    sys.exit(main(args[0]))
