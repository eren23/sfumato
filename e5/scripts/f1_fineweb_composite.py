"""F1 — Composite vs ar_only on FineWeb subsamples.

Does the GSM8K crossover generalise to a broad-corpus substrate?

Parameter-Golf (FineWeb-Edu sample-10BT) substrate. Train composite +
ar_only at 200M for 3k steps at three token-budget levels:
  5M, 10M, 50M tokens.

3 seeds each. Score AR-NLL on a held-out FineWeb chunk.

Decision: if composite < ar_only at the smaller budgets (5M, 10M) but
loses at 50M, the crossover generalises beyond math reasoning. If
composite never wins on FineWeb, the GSM8K crossover is
substrate-specific.

Env:
  TOKENS_LIST=5000000,10000000,50000000
  SEEDS=110,111,112
  MAX_STEPS=3000
  OUT_NAME=f1_fineweb
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.train import train_one  # noqa: E402
from e5.data import load_fineweb_tokens  # noqa: E402
from e5.model_composite import CompositeConfig, CompositeLM  # noqa: E402


def env_int(k, d): return int(os.environ.get(k, str(d)))


@torch.no_grad()
def score_ar_nll_on_chunks(model, chunks, device, max_len=256):
    """AR-NLL on a list of token sequences. Used for FineWeb held-out."""
    total_nll = 0.0
    total_count = 0
    for ids in chunks:
        ids = ids[:max_len]
        if len(ids) < 2:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        pred = logits[0, :-1].float()
        target = idx[0, 1:]
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll += float(nlls.sum())
        total_count += int(nlls.size)
    avg = total_nll / max(1, total_count)
    return {"avg_nll": round(avg, 4), "perplexity": round(math.exp(avg), 2),
            "n_tokens": total_count, "n_chunks": len(chunks)}


def build_holdout(n_holdout_tokens: int = 200_000, n_chunks: int = 100, chunk_len: int = 200):
    """Use the LAST n_holdout_tokens of the cached FineWeb stream as held-out.

    The cache holds 50M tokens by default; we never train on the last
    chunk, so this is leakage-free for the 5M / 10M / 50M training
    splits as long as those slice from the beginning.
    """
    full = load_fineweb_tokens(n_tokens=50_000_000)
    if len(full) < n_holdout_tokens:
        n_holdout_tokens = len(full) // 5
    holdout_stream = full[-n_holdout_tokens:]
    # Cut into chunks of chunk_len
    chunks = []
    for start in range(0, len(holdout_stream) - chunk_len, chunk_len):
        chunks.append(holdout_stream[start:start + chunk_len].tolist())
        if len(chunks) >= n_chunks:
            break
    return chunks


def main():
    tokens_list = [int(x) for x in os.environ.get("TOKENS_LIST", "5000000,10000000,50000000").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "110,111,112").split(",")]
    max_steps = env_int("MAX_STEPS", 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    d_model = env_int("D_MODEL", 896)
    n_layers = env_int("N_LAYERS", 12)
    n_heads = env_int("N_HEADS", 16)

    out_name = os.environ.get("OUT_NAME", "f1_fineweb")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} tokens_list={tokens_list} seeds={seeds}")

    # Cache 50M FineWeb tokens up front (used for all subsets + held-out).
    print("loading FineWeb (50M tokens, may take a few min on first run)...")
    full_tokens = load_fineweb_tokens(n_tokens=50_000_000)
    print(f"loaded {len(full_tokens):,} train tokens")

    print("building held-out chunks...")
    holdout_chunks = build_holdout()
    print(f"  {len(holdout_chunks)} chunks of ~200 tokens each")

    all_rows = []
    for n_tok in tokens_list:
        if n_tok > len(full_tokens) - 200_000:
            print(f"skip n_tok={n_tok} (exceeds available)")
            continue
        sub_tokens = full_tokens[:n_tok]
        size_out = base_out / f"{n_tok//1_000_000}M"
        size_out.mkdir(parents=True, exist_ok=True)
        print(f"\n========== n_tok={n_tok:,} ==========")
        for variant in ("composite", "ar_only"):
            for seed in seeds:
                run_out = size_out / f"{variant}_seed{seed}"
                ckpt = run_out / "model.pt"
                if not ckpt.exists():
                    print(f"\n[{n_tok//1_000_000}M {variant} seed={seed}] training...")
                    t0 = time.time()
                    train_one(
                        variant=variant,
                        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        out_dir=run_out,
                        seed=seed, max_steps=max_steps,
                        batch_size=batch_size, block_size=block_size,
                        peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=sub_tokens,
                    )
                    print(f"  train wall_s={round(time.time()-t0, 1)}")

                # Score
                ck = torch.load(ckpt, map_location=device, weights_only=False)
                mcfg = CompositeConfig(**ck["config"])
                model = CompositeLM(mcfg).to(device)
                model.load_state_dict(ck["state_dict"])
                model.train(False)
                r = score_ar_nll_on_chunks(model, holdout_chunks, device, max_len=block_size)
                row = {"n_train_tokens": n_tok, "variant": variant, "seed": seed,
                       "steps": max_steps, **r}
                all_rows.append(row)
                print(f"  AR-NLL on FineWeb held-out = {r['avg_nll']}")
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
                (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    # Aggregate
    print("\n========== FINAL AGGREGATE ==========")
    by = {}
    for r in all_rows:
        by.setdefault(r["n_train_tokens"], {}).setdefault(r["variant"], []).append(r["avg_nll"])
    for n_tok in sorted(by):
        comp = by[n_tok].get("composite", [])
        ar = by[n_tok].get("ar_only", [])
        if comp and ar:
            cm = sum(comp)/len(comp); am = sum(ar)/len(ar); d = cm - am
            print(f"  {n_tok//1_000_000}M tok: composite={cm:.3f} ar_only={am:.3f} Δ={d:+.3f} n={len(comp)}")

    print(f"\nwrote {base_out/'summary.json'}")


if __name__ == "__main__":
    main()
