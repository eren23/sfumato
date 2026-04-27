"""Build the Track 2 commit-adapter training mixture from consensus_raw.jsonl.

Three buckets, mixed by ratio:
  - rescue:  consensus_correct AND greedy_wrong            → "fix" signal
  - preserve_disagreement:
       greedy_correct AND consensus_correct AND
       len(set(branch_answers)) >= 2                       → "stabilize" signal
  - pure_agreement:
       greedy_correct AND all 5 branches identical         → "leave alone" signal

Default mix: rescue 40% / preserve 50% / pure 5% / val 5%.
(Pure agreement is downweighted — high overlap with input/target → near-zero
gradient signal — but kept as a small anchor.)

Also emits a branch-agreement-rate report (zero-cost falsifier metric):
  - distribution of unique-answer-counts per problem
  - mean unique count per bucket

Inputs:
  e2/data/consensus_raw.jsonl

Outputs:
  e2/data/commit_mixture.jsonl
  e2/data/commit_mixture_report.json
  pushed to HF Hub: eren23/sfumato-commit-mixture-gsm8k (if --push true)

Each output row: {question, target_cot, target_answer, bucket, source_idx}.

Target selection rule (per bucket):
  - rescue:               first branch whose answer == majority_answer
  - preserve_disagreement: greedy_output (teach: "you were right; don't change")
  - pure_agreement:       greedy_output

Usage (smoke):
  python scripts/build_commit_mixture.py --push False
Real run:
  HF_TOKEN=... python scripts/build_commit_mixture.py --push True \
      --repo_id eren23/sfumato-commit-mixture-gsm8k
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "e2" / "data"
RAW_PATH = DATA_DIR / "consensus_raw.jsonl"
OUT_PATH = DATA_DIR / "commit_mixture.jsonl"
REPORT_PATH = DATA_DIR / "commit_mixture_report.json"


def _ans_eq(a: str, b: str) -> bool:
    """Numeric-tolerant answer equality. Falls back to string-eq."""
    if not a or not b:
        return False
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def _classify(row: dict) -> str:
    """Return bucket name for a raw row, or 'discard' if it doesn't fit."""
    gold = row.get("gold_answer", "")
    greedy = row.get("greedy_answer", "")
    branch_answers = [a for a in row.get("branch_answers", []) if a]
    majority = row.get("majority_answer", "")
    consensus_correct = bool(row.get("consensus_correct"))
    greedy_correct = _ans_eq(greedy, gold)

    if consensus_correct and not greedy_correct:
        return "rescue"
    if consensus_correct and greedy_correct:
        unique_branch_answers = set(branch_answers)
        if len(unique_branch_answers) >= 2:
            return "preserve_disagreement"
        if len(unique_branch_answers) == 1 and len(branch_answers) >= 5:
            return "pure_agreement"
    return "discard"


def _select_target(row: dict, bucket: str) -> tuple[str, str]:
    """Return (target_cot, target_answer)."""
    if bucket == "rescue":
        majority = row.get("majority_answer", "")
        for cot, ans in zip(row.get("branches", []), row.get("branch_answers", [])):
            if _ans_eq(ans, majority):
                return cot, ans
        return row.get("branches", [""])[0], row.get("branch_answers", [""])[0]
    return row.get("greedy_output", ""), row.get("greedy_answer", "")


def _agreement_report(rows: list[dict]) -> dict:
    """Distribution of unique-answer-counts per problem (zero-cost falsifier).

    A diversity-collapse failure mode for commit-LoRA would shift this
    distribution toward 1; we record the pre-LoRA distribution so post-LoRA
    eval can compare directly.
    """
    counts = Counter()
    by_bucket: dict[str, list[int]] = {}
    for r in rows:
        unique = len(set(a for a in r.get("branch_answers", []) if a))
        counts[unique] += 1
        bucket = _classify(r)
        by_bucket.setdefault(bucket, []).append(unique)

    avg = {b: (sum(v) / len(v)) for b, v in by_bucket.items() if v}
    return {
        "n_total": len(rows),
        "unique_answer_count_distribution": dict(sorted(counts.items())),
        "mean_unique_count_by_bucket": avg,
        "bucket_counts": {b: len(v) for b, v in by_bucket.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=str(RAW_PATH))
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    parser.add_argument("--report", type=str, default=str(REPORT_PATH))
    parser.add_argument("--rescue_frac", type=float, default=0.40)
    parser.add_argument("--preserve_frac", type=float, default=0.50)
    parser.add_argument("--pure_frac", type=float, default=0.05)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--push", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument(
        "--repo_id", type=str, default="eren23/sfumato-commit-mixture-gsm8k"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"[mixture] missing raw jsonl: {raw_path}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"[mixture] loaded {len(rows)} raw rows from {raw_path}")

    report = _agreement_report(rows)
    print(f"[mixture] branch-agreement distribution: {report['unique_answer_count_distribution']}")
    print(f"[mixture] bucket counts: {report['bucket_counts']}")
    print(f"[mixture] mean unique count by bucket: {report['mean_unique_count_by_bucket']}")

    by_bucket: dict[str, list[dict]] = {
        "rescue": [],
        "preserve_disagreement": [],
        "pure_agreement": [],
    }
    for r in rows:
        b = _classify(r)
        if b in by_bucket:
            by_bucket[b].append(r)

    rng = random.Random(args.seed)
    for v in by_bucket.values():
        rng.shuffle(v)

    target_total = max(
        1,
        int(len(by_bucket["rescue"]) / max(args.rescue_frac, 1e-6)),
    )
    n_rescue = min(len(by_bucket["rescue"]), int(target_total * args.rescue_frac))
    n_preserve = min(
        len(by_bucket["preserve_disagreement"]),
        int(target_total * args.preserve_frac),
    )
    n_pure = min(
        len(by_bucket["pure_agreement"]),
        max(1, int(target_total * args.pure_frac)),
    )

    print(f"[mixture] sized: rescue={n_rescue} preserve={n_preserve} pure={n_pure}")

    out_rows: list[dict] = []
    for bucket, n_take in (
        ("rescue", n_rescue),
        ("preserve_disagreement", n_preserve),
        ("pure_agreement", n_pure),
    ):
        for r in by_bucket[bucket][:n_take]:
            cot, ans = _select_target(r, bucket)
            if not cot or not ans:
                continue
            out_rows.append({
                "question": r["question"],
                "target_cot": cot,
                "target_answer": ans,
                "bucket": bucket,
                "source_idx": r.get("source_idx", -1),
            })

    rng.shuffle(out_rows)
    val_n = max(1, int(len(out_rows) * args.val_frac))
    val_rows = out_rows[:val_n]
    train_rows = out_rows[val_n:]
    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "validation"

    final_rows = train_rows + val_rows
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report["mixture_size"] = len(final_rows)
    report["mixture_train"] = len(train_rows)
    report["mixture_val"] = len(val_rows)
    report["mixture_bucket_counts"] = Counter(r["bucket"] for r in final_rows)
    report_path = Path(args.report)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[mixture] wrote {len(final_rows)} rows to {out_path}")
    print(f"[mixture] train={len(train_rows)} val={len(val_rows)}")
    print(f"[mixture] mixture bucket counts: {report['mixture_bucket_counts']}")
    print(f"[mixture] report: {report_path}")

    if args.push:
        from datasets import Dataset  # type: ignore

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            print("[mixture] no HF token; cannot push", file=sys.stderr)
            return 1
        ds = Dataset.from_list(final_rows)
        ds.push_to_hub(args.repo_id, token=token)
        print(f"[mixture] pushed {args.repo_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
