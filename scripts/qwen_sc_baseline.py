"""Qwen-AR self-consistency baseline (reviewer-resilience experiment).

Generates 5 branches per problem with Qwen2.5-0.5B-Instruct at t=0.7, extracts
the answer per branch, majority-votes. Tells us whether the diffusion-cmaj
+6pp lift is diffusion-specific or generic self-consistency.

Usage on pod:
  python scripts/qwen_sc_baseline.py --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--branches", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--ar_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset  # type: ignore

    from e4 import ar_qwen, grade  # type: ignore

    print(f"[qwen-sc] loading {args.ar_model}", flush=True)
    model = ar_qwen.load(args.ar_model, mock=False)

    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    correct_count = 0
    start = time.monotonic()

    for i in range(min(args.n, len(ds))):
        ex = ds[i]
        question = ex["question"]
        gold_full = ex["answer"]
        gold = gold_full.rsplit("####", 1)[-1].strip()

        branch_texts: list[str] = []
        branch_answers: list[str] = []
        for b in range(args.branches):
            text, _ = model.generate(
                prompt=question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed * 1000 + b,
            )
            branch_texts.append(text)
            branch_answers.append(grade.extract_answer(text))

        counts = Counter(a for a in branch_answers if a)
        if counts:
            majority = counts.most_common(1)[0][0]
        else:
            majority = branch_answers[0] or ""

        is_correct = grade.is_correct(majority, gold)
        if is_correct:
            correct_count += 1
        rows.append({
            "idx": i,
            "gold": gold,
            "pred": majority,
            "correct": bool(is_correct),
            "branch_answers": branch_answers,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.monotonic() - start
            print(
                f"[qwen-sc] {i+1}/{args.n}  acc={correct_count/(i+1):.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    out_dir = REPO_ROOT / "e2" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qwen_sc_baseline_b5_t07_N200.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    final_acc = correct_count / len(rows) if rows else 0.0
    print(f"[qwen-sc] DONE n={len(rows)} accuracy={final_acc:.4f} → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
