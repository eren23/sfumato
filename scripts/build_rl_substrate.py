"""Build the schedule-RLHF training substrate for Phase-4 Direction A.

Mixes GSM8K-train + MATH-train + (optionally) AIME-train into a single
JSONL of (question, gold_answer) pairs. Used by
`scripts/train_track2_commit_rl.py` as the rollout-prompt source for
GRPO over commit-LoRA's K2 schedule.

Output schema (one JSON object per line):
    {
        "id": str,                # unique within file
        "source": "gsm8k_train" | "math_train" | "aime_train",
        "question": str,          # plain text, no chat template
        "gold": str,              # canonicalized numeric / latex answer
        "answer_extract": "####" | None,  # for graders that parse a marker
        "subject": str | None,    # MATH-style category if available
    }

Default mixture per Phase-4 PRE_REG:
    GSM8K-train  : 1000 rows
    MATH-train   :  500 rows (numeric-only)
    AIME-train   :   50 rows (or skipped if hub-only)

Usage:
    python scripts/build_rl_substrate.py \
        --out e4/data/rl_substrate_phase4_v1.jsonl \
        --gsm8k 1000 --math 500 --aime 50

No GPU. Runs locally in <2 min.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_gsm8k_train(n: int):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    rows = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        # Gold is "<reasoning>\n#### N" — extract trailing number for canonical form.
        gold_match = re.search(r"####\s*(-?[\d,\.]+)", r["answer"])
        gold = gold_match.group(1).replace(",", "") if gold_match else r["answer"].strip()
        rows.append({
            "id": f"gsm8k_train_{i}",
            "source": "gsm8k_train",
            "question": r["question"],
            "gold": gold,
            "answer_extract": "####",
            "subject": None,
        })
    return rows


def _load_math_train(n: int):
    """Load MATH-train numeric subset.

    Hendrycks MATH dataset has 7,500 train problems. Filter to
    numeric-only answers (matches sfumato's grader scope).
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("hendrycks/competition_math", split="train")
    except Exception:
        # Fallback: use HuggingFaceH4/MATH-500 train slice if competition_math
        # is gated. (Some datasets require auth.) MATH-500 only has test split,
        # so we synthetically take the first n MATH-500 numeric problems as
        # in-distribution train proxies. NOT ideal — real MATH-train preferred.
        print("[build_rl_substrate] WARN: hendrycks/competition_math unavailable; "
              "falling back to HuggingFaceH4/MATH-500 train-style subset (problems "
              "we have NOT used in T2.A eval — careful split discipline).",
              flush=True)
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        # Use indices NOT in our T2.A eval (idx ≥ 200 from numeric subset).
        rows = []
        with (REPO_ROOT / "e4/data/math500_numeric_indices.json").open() as f:
            spec = json.load(f)
        eval_idxs = set(spec["indices"][:200])  # T2.A.4 N=200 used these
        train_idxs = [i for i in spec["indices"] if i not in eval_idxs][:n]
        for i in train_idxs:
            r = ds[i]
            ans = str(r["answer"]).strip()
            if not re.fullmatch(r"-?\d+(?:\.\d+)?", ans):
                continue
            rows.append({
                "id": f"math_train_proxy_{i}",
                "source": "math_train",
                "question": r["problem"],
                "gold": ans,
                "answer_extract": None,
                "subject": r.get("subject"),
            })
        return rows[:n]
    rows = []
    for i, r in enumerate(ds):
        if len(rows) >= n:
            break
        # MATH gold is "...$\\boxed{<answer>}$." — extract last \\boxed.
        gold_match = re.search(r"\\boxed\{([^}]*)\}", r["solution"])
        if not gold_match:
            continue
        gold = gold_match.group(1).strip()
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
            continue
        rows.append({
            "id": f"math_train_{i}",
            "source": "math_train",
            "question": r["problem"],
            "gold": gold,
            "answer_extract": None,
            "subject": r.get("subject") or r.get("type"),
        })
    return rows


def _load_aime_train(n: int):
    """AIME problems. Prefer Maxwell-Jia/AIME_2024 or AI-MO/aimo-validation-aime.
    Hub gating + format variability is real; fall back to no-op if unavailable.
    """
    if n <= 0:
        return []
    from datasets import load_dataset
    candidates = [
        ("AI-MO/aimo-validation-aime", "train", "Problem", "Answer"),
        ("Maxwell-Jia/AIME_2024", "train", "Problem", "Answer"),
    ]
    for repo, split, q_col, a_col in candidates:
        try:
            ds = load_dataset(repo, split=split)
        except Exception:
            continue
        rows = []
        for i, r in enumerate(ds):
            if len(rows) >= n:
                break
            ans = str(r.get(a_col, "")).strip()
            if not re.fullmatch(r"-?\d+", ans):
                continue
            rows.append({
                "id": f"aime_train_{i}",
                "source": "aime_train",
                "question": r[q_col],
                "gold": ans,
                "answer_extract": None,
                "subject": "competition",
            })
        if rows:
            return rows
    print("[build_rl_substrate] WARN: no AIME dataset accessible; skipping.",
          flush=True)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e4/data/rl_substrate_phase4_v1.jsonl")
    ap.add_argument("--gsm8k", type=int, default=1000)
    ap.add_argument("--math", type=int, default=500)
    ap.add_argument("--aime", type=int, default=50)
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_rl_substrate] loading GSM8K-train (n={args.gsm8k})...", flush=True)
    gsm = _load_gsm8k_train(args.gsm8k)
    print(f"  got {len(gsm)} rows")
    print(f"[build_rl_substrate] loading MATH-train (n={args.math})...", flush=True)
    mat = _load_math_train(args.math)
    print(f"  got {len(mat)} rows")
    print(f"[build_rl_substrate] loading AIME-train (n={args.aime})...", flush=True)
    aim = _load_aime_train(args.aime)
    print(f"  got {len(aim)} rows")

    rows = gsm + mat + aim
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build_rl_substrate] wrote {len(rows)} rows -> {out_path}",
          flush=True)
    print(f"  source breakdown: gsm8k={len(gsm)}, math={len(mat)}, aime={len(aim)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
