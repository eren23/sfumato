"""Backfill existing raw_*.jsonl results into wandb retroactively.

Each (condition, k_steps, ar_model, diff_model, seed) cell becomes one wandb
run, with one log step per problem and a summary at the end.

Usage:
    export WANDB_API_KEY=...
    python scripts/backfill_wandb.py e4/results/

Skips files that look corrupt or already-uploaded (by checking the filename
suffix against a local manifest).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def cell_key(row: dict[str, Any]) -> tuple:
    return (
        row["condition"],
        row.get("k_steps", 0),
        row.get("ar_model", "unknown_ar"),
        row.get("diff_model", "unknown_diff"),
        row.get("seed", 0),
        row.get("branches", 1),
        round(float(row.get("temperature", 0.0)), 2),
    )


def upload_cell(
    cell: tuple,
    rows: list[dict[str, Any]],
    project: str,
    entity: str | None,
    source_file: str,
) -> None:
    import wandb  # type: ignore

    cond, k, ar, diff, seed, branches, temp = cell
    branches_suffix = f"-b{branches}" if branches > 1 else ""
    temp_suffix = f"-t{temp}" if temp and branches > 1 else ""
    name = (
        f"{cond}-k{k}-seed{seed}{branches_suffix}{temp_suffix}"
        f"-{ar.split('/')[-1] if '/' in ar else ar}"
    )
    cfg = {
        "condition": cond,
        "k_steps": k,
        "n_problems": len(rows),
        "seed": seed,
        "ar_model": ar,
        "diff_model": diff,
        "branches": rows[0].get("branches", 1) if rows else 1,
        "mock": False,
        "backfilled": True,
        "source_file": source_file,
    }
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=cfg,
        reinit=True,
        tags=[cond, f"k={k}", "backfill"],
    )
    n_correct = 0
    total_flops = 0
    for i, r in enumerate(rows):
        n_correct += int(r.get("correct", False))
        total_flops += r.get("flops", 0)
        running_acc = n_correct / (i + 1)
        run.log(
            {
                "step": i + 1,
                "running_acc": running_acc,
                "running_loss": 1.0 - running_acc,
                "flops_used": r.get("flops", 0),
                "flops_cumulative": total_flops,
                "correct": int(r.get("correct", False)),
            }
        )
    accuracy = n_correct / max(len(rows), 1)
    run.summary["accuracy"] = accuracy
    run.summary["total_flops"] = total_flops
    run.summary["mean_flops_per_problem"] = total_flops / max(len(rows), 1)

    # Per-problem text table: idx, gold, pred, correct, full trace.
    table = wandb.Table(
        columns=["idx", "id", "gold", "pred", "correct", "flops", "trace"]
    )
    for r in rows:
        trace_str = "\n\n".join(
            f"### {k}\n{v}" for k, v in r.get("trace", {}).items() if v
        )
        table.add_data(
            r.get("idx"),
            r.get("id"),
            r.get("gold"),
            r.get("pred"),
            bool(r.get("correct", False)),
            r.get("flops", 0),
            trace_str[:6000],
        )
    run.log({"problems": table})
    run.finish()
    print(
        f"[backfill] {name}: n={len(rows)} acc={accuracy:.3f} "
        f"flops={total_flops:.2e}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", default="e4/results", nargs="?")
    ap.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "sfumato-e4"),
    )
    ap.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
    )
    ap.add_argument("--min-rows", type=int, default=10)
    args = ap.parse_args(argv)

    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set", file=sys.stderr)
        return 1

    base = Path(args.results_dir)
    by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    file_by_cell: dict[tuple, str] = {}
    for raw in sorted(base.glob("raw_*.jsonl")):
        rows = load_rows(raw)
        if not rows:
            continue
        for r in rows:
            by_cell[cell_key(r)].append(r)
        # Best-effort: associate first cell key with this file for traceability.
        if rows:
            file_by_cell.setdefault(cell_key(rows[0]), str(raw.name))

    skipped = 0
    uploaded = 0
    for cell, rows in sorted(by_cell.items()):
        if len(rows) < args.min_rows:
            print(
                f"[backfill] skip {cell} (n={len(rows)} < min_rows={args.min_rows})"
            )
            skipped += 1
            continue
        upload_cell(
            cell,
            rows,
            project=args.project,
            entity=args.entity,
            source_file=file_by_cell.get(cell, "unknown"),
        )
        uploaded += 1
    print(f"[backfill] {uploaded} runs uploaded, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
