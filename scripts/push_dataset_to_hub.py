"""Push a locally-generated parquet dataset to HF Hub.

Used because the safety hooks block transferring HF_TOKEN to the pod, so
the pod generates the parquet (no push), we rsync it back to local, and
this script pushes from local where HF_TOKEN already lives in
parameter-golf_dev/.env.

Usage:
    HF_TOKEN=... python scripts/push_dataset_to_hub.py \\
        e2/data/prefix_robust_dataset.parquet \\
        eren23/sfumato-prefix-robust-gsm8k \\
        --split-frac 0.05 --stratify-by prefix_tier
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path, help="local parquet")
    ap.add_argument("repo_id", help="HF Hub repo id e.g. eren23/sfumato-...")
    ap.add_argument("--split-frac", type=float, default=0.05)
    ap.add_argument("--stratify-by", default=None, help="column for stratified split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var not set", file=sys.stderr)
        return 1
    if not args.parquet.exists():
        print(f"missing {args.parquet}", file=sys.stderr)
        return 1

    from datasets import ClassLabel, Dataset, DatasetDict  # type: ignore

    ds = Dataset.from_parquet(str(args.parquet))
    print(f"[push] loaded {len(ds)} rows from {args.parquet}")
    n_test = max(int(round(len(ds) * args.split_frac)), 1)
    if args.stratify_by and args.stratify_by in ds.column_names:
        # Cast to ClassLabel if not already.
        col = ds.features[args.stratify_by]
        if not isinstance(col, ClassLabel):
            uniq = sorted(set(ds[args.stratify_by]))
            ds = ds.cast_column(args.stratify_by, ClassLabel(names=uniq))
            n_classes = len(uniq)
        else:
            n_classes = len(col.names)
        if n_test >= n_classes:
            split = ds.train_test_split(
                test_size=args.split_frac,
                seed=args.seed,
                stratify_by_column=args.stratify_by,
            )
        else:
            split = ds.train_test_split(test_size=args.split_frac, seed=args.seed)
    else:
        split = ds.train_test_split(test_size=args.split_frac, seed=args.seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    print(
        f"[push] pushing to https://huggingface.co/datasets/{args.repo_id} "
        f"(private={'no' if args.public else 'yes'})"
    )
    dd.push_to_hub(args.repo_id, private=not args.public, token=token)
    print("[push] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
