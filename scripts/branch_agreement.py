"""Compute branch-agreement-rate distribution from a cmaj/cmajc raw jsonl.

This is the diversity-preservation falsifier from the paper writeup. We compute
the fraction of problems where {5/5, 4/5, 3/5, 2/5, 1/5} branches agree on the
final numeric answer. A diversity-collapse failure would shift the distribution
toward 5/5 (full agreement = no exploration).

Inputs:
  --raw    path to raw_{cmaj,cmajc}_*.jsonl (each row has branch_answers list)
  --label  short label for the report

Output: prints histogram + fraction-with-5/5-same.

Usage:
  python scripts/branch_agreement.py \
      --raw e2/data/consensus_raw.jsonl --label "base"
  python scripts/branch_agreement.py \
      --raw e4/results/raw_cmajc_k64_seed0.jsonl --label "Track1+commit-v2"
"""

from __future__ import annotations

import argparse
import json
import statistics as stat
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--label", type=str, default="run")
    args = parser.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        print(f"missing: {raw}", file=sys.stderr)
        return 1

    rows = []
    with raw.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n = len(rows)
    if not n:
        print("no rows", file=sys.stderr)
        return 1

    # Find the right field — cmajc raw uses "branches" + "branch_answers";
    # consensus_raw also uses these. Some legacy files use "trace".
    ba_field = None
    for cand in ("branch_answers", "branches"):
        if cand in rows[0]:
            ba_field = cand
            break
    if ba_field is None:
        # Try to dig into a "trace" object.
        if "trace" in rows[0] and isinstance(rows[0]["trace"], dict):
            for cand in ("branch_answers", "branches"):
                if cand in rows[0]["trace"]:
                    ba_field = f"trace.{cand}"
                    break
    if ba_field is None:
        print(f"no branch_answers field found; row keys: {list(rows[0].keys())}", file=sys.stderr)
        return 1

    def get_branches(r):
        if "." in ba_field:
            a, b = ba_field.split(".", 1)
            return r.get(a, {}).get(b, [])
        return r.get(ba_field, [])

    unique_counts = []
    for r in rows:
        bs = get_branches(r)
        if not bs:
            continue
        # Branches may be CoTs (strings) or pre-extracted answers.
        # If they look like CoTs (contain newlines/spaces), grab the last token.
        if isinstance(bs[0], str) and "\n" in bs[0] + " ":
            # Probably full CoTs; extract answer via "Answer:" suffix or last number.
            # Easier: if `branch_answers` is present alongside, prefer it.
            pass
        # Filter out empty strings.
        bs = [str(b).strip() for b in bs if str(b).strip()]
        unique_counts.append(len(set(bs)))

    if not unique_counts:
        print("no usable branch answers", file=sys.stderr)
        return 1

    c = Counter(unique_counts)
    total = sum(c.values())
    print(f"\n=== Branch agreement: {args.label} ===")
    print(f"file: {raw}  n_rows={n}  n_with_branches={total}")
    print(f"distribution (unique answers per problem):")
    for k in sorted(c.keys()):
        bar = "█" * int(50 * c[k] / total)
        print(f"  {k}/N : {c[k]:4d}  ({c[k]/total:.2%})  {bar}")
    same_5 = c.get(1, 0)
    print(f"\n  5/5 same      : {same_5} ({same_5/total:.2%})  ← diversity-collapse signal")
    print(f"  ≥2 unique     : {total - same_5} ({(total - same_5)/total:.2%})")
    print(f"  mean unique   : {stat.mean(unique_counts):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
