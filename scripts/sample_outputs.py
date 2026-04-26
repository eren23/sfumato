"""Render sample model outputs from raw_*.jsonl files into readable markdown.

Usage:
    python scripts/sample_outputs.py e4/results/ --per-condition 6 \\
        --out e4/results/samples.md

Picks `--per-condition` rows per (condition, k_steps) cell, balanced between
correct and incorrect predictions when possible, and dumps the full per-stage
trace as markdown so you can eyeball generation quality.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in sorted(results_dir.glob("raw_*.jsonl")):
        with raw.open() as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def pick_balanced(
    rows: list[dict[str, Any]], k: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Pick up to k rows, balancing correct vs incorrect."""
    rng = random.Random(seed)
    correct = [r for r in rows if r.get("correct")]
    wrong = [r for r in rows if not r.get("correct")]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    half = k // 2
    out = correct[:half] + wrong[: k - half]
    if len(out) < k:
        # Top up from whichever pool is non-empty.
        rest = correct[half:] + wrong[k - half :]
        rng.shuffle(rest)
        out += rest[: k - len(out)]
    return out[:k]


def render(
    samples: dict[tuple, list[dict[str, Any]]],
    questions: dict[str, str],
) -> str:
    parts = ["# Sfumato E4 — sample outputs\n"]
    for (cond, k), rows in sorted(samples.items()):
        parts.append(f"## condition={cond} k_steps={k}  ({len(rows)} examples)\n")
        if rows:
            parts.append(
                f"_ar={rows[0].get('ar_model','?')}  "
                f"diff={rows[0].get('diff_model','?')}_\n"
            )
        for i, r in enumerate(rows):
            qid = r.get("id", "?")
            q = questions.get(qid, "(question text unavailable)")
            mark = "✓" if r.get("correct") else "✗"
            parts.append(
                f"### {cond} #{i+1} (idx={r['idx']}, gsm8k_id={qid}) {mark}"
            )
            parts.append(f"**question:** {q}\n")
            parts.append(f"**gold:** `{r['gold']}`  **pred:** `{r['pred']}`\n")
            for stage, text in r.get("trace", {}).items():
                if not text:
                    text = "(empty)"
                parts.append(f"**{stage}:**")
                parts.append(f"```\n{text.strip()[:1500]}\n```")
            parts.append("")
    return "\n".join(parts)


def load_gsm8k_questions(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Best effort: stream GSM8K test split for the unique ids referenced."""
    needed = {r["id"] for r in rows if "id" in r and r["id"].isdigit()}
    if not needed:
        return {}
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return {}
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception:
        return {}
    out: dict[str, str] = {}
    for idx_str in needed:
        idx = int(idx_str)
        if 0 <= idx < len(ds):
            out[idx_str] = ds[idx]["question"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", default="e4/results", nargs="?")
    ap.add_argument("--per-condition", type=int, default=6)
    ap.add_argument("--out", default="e4/results/samples.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base = Path(args.results_dir)
    rows = load_rows(base)
    if not rows:
        print(f"no rows in {base}")
        return 1

    by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["condition"], r.get("k_steps", 0))].append(r)
    # Skip the tiny pre-N=50 dev runs.
    by_cell = {k: v for k, v in by_cell.items() if len(v) >= 5}

    samples: dict[tuple, list[dict[str, Any]]] = {}
    for cell, cell_rows in by_cell.items():
        samples[cell] = pick_balanced(
            cell_rows, args.per_condition, seed=args.seed
        )

    all_picked = [r for rows in samples.values() for r in rows]
    questions = load_gsm8k_questions(all_picked)

    md = render(samples, questions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out} ({len(md)} chars, {sum(len(v) for v in samples.values())} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
