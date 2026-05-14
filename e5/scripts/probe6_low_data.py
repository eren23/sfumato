"""Probe #6 — low-data training (DiFFPO hypothesis): does composite help
when training data is scarce?

DiFFPO (arXiv 2510.02212) reports that discrete diffusion beats AR in
data-constrained settings. Our overnight result was at full GSM8K-train
(4M tokens). If composite > AR-only in a 10×-reduced regime, that's a
real signal for the vision.

Setup: subsample GSM8K-train to 1k problems (~400k tokens). Train at
200M scale, 3 seeds × {composite, ar_only}.

For wall-clock budget: 200M-3k-steps ~ 15 min on a 4090. 6 runs ~ 90 min
on a single pod.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.train import train_one  # noqa: E402
from e5.data import load_gsm8k_train_tokens, GSM8KStreamingDataset  # noqa: E402


def env_int(k, d): return int(os.environ.get(k, str(d)))


def main():
    n_seeds = env_int("N_SEEDS", 3)
    max_steps = env_int("MAX_STEPS", 3000)
    n_problems = env_int("N_PROBLEMS_LIMIT", 1000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    d_model = env_int("D_MODEL", 896)
    n_layers = env_int("N_LAYERS", 12)
    n_heads = env_int("N_HEADS", 16)

    # Build a sub-tokenstream from a 1k-problem subset of GSM8K-train.
    full_tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"full GSM8K-train tokens: {len(full_tokens):,}")
    # The cached file is concatenated questions+answers. We approximate 1k
    # problems by taking the first ~400k tokens (~10% of the 4M total).
    target_n_tokens = int(len(full_tokens) * n_problems / 7473)
    sub_tokens = full_tokens[:target_n_tokens]
    print(f"subsampled to {len(sub_tokens):,} tokens (≈ {n_problems} problems)")

    base_out = REPO_ROOT / "e5" / "results" / f"probe6_lowdata_{n_problems}p"
    base_out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in ("composite", "ar_only"):
        for seed in range(n_seeds):
            out_dir = base_out / variant / f"seed{seed}"
            print(f"\n=== {variant} seed={seed} on {len(sub_tokens):,} tokens ===")
            summary = train_one(
                variant=variant,
                d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                out_dir=out_dir,
                seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
                peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=sub_tokens,
            )
            summary["data_subset_n_problems"] = n_problems
            summary["data_subset_n_tokens"] = len(sub_tokens)
            summary_rows.append({**summary, "out_dir": str(out_dir)})

    (base_out / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"\nwrote {base_out/'summary.json'}")


if __name__ == "__main__":
    main()
