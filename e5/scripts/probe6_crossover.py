"""E3c — D3 crossover refinement.

Phase-D's D3 placed the AR-axis crossover (composite vs ar_only) between
1000 and 2000 problems. We need to localise it for the paper figure.

This script trains composite + ar_only at 200M-3k on three new data sizes:
800p, 1200p, 1500p, with 3 seeds each. Then scores AR-NLL on a held-out
chunk and reports Δ_NLL(composite − ar_only).

Existing D3 anchors (mean ar_only − composite, so positive = composite tax):
  500p   → −0.40  (composite WIN at 5.7σ)
  1000p  → −0.01  (tied)
  2000p  → +0.13  (composite tax)
  4000p  → +0.26
  7500p  → +0.22  (full data)

We expect the new 3 points to lie on the monotone curve between 1000p and
2000p, localising the crossover to ≤300p width.

Env:
  SIZES=800,1200,1500
  SEEDS=90,91,92
  MAX_STEPS=3000
  OUT_NAME=e3c_d3_crossover
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
from e5.data import load_gsm8k_train_tokens  # noqa: E402
from e5.model_composite import CompositeConfig, CompositeLM  # noqa: E402


def env_int(k, d): return int(os.environ.get(k, str(d)))


def load_gsm8k_probe_problems(n=100, offset=7000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(offset, min(offset + n, len(ds))):
        row = ds[idx]
        p = tok.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
        a = tok.encode(" " + row["answer"], add_special_tokens=False)
        out.append((p, a))
    return out


@torch.no_grad()
def score_ar_nll(model, problems, device, max_len=256):
    """AR mode: NLL on the answer-region tokens given the prompt."""
    total_nll = 0.0
    total_count = 0
    for prompt_ids, answer_ids in problems:
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            continue
        full = prompt_ids + answer_ids
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        ans_start = len(prompt_ids)
        pred = logits[0, ans_start - 1 : ans_start - 1 + len(answer_ids), :].float()
        target = torch.tensor(answer_ids, dtype=torch.long, device=device)
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll += float(nlls.sum())
        total_count += int(nlls.size)
    avg = total_nll / max(1, total_count)
    return {"avg_nll": round(avg, 4), "perplexity": round(math.exp(avg), 2),
            "n_tokens": total_count}


def main():
    sizes = [int(s) for s in os.environ.get("SIZES", "800,1200,1500").split(",")]
    seeds = [int(s) for s in os.environ.get("SEEDS", "90,91,92").split(",")]
    max_steps = env_int("MAX_STEPS", 3000)
    n_probe = env_int("N_PROBES", 100)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    # 200M defaults
    d_model = env_int("D_MODEL", 896)
    n_layers = env_int("N_LAYERS", 12)
    n_heads = env_int("N_HEADS", 16)

    out_name = os.environ.get("OUT_NAME", "e3c_d3_crossover")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} sizes={sizes} seeds={seeds} max_steps={max_steps}")

    full_tokens = load_gsm8k_train_tokens(include_reasoning=True)
    problems = load_gsm8k_probe_problems(n=n_probe)
    print(f"loaded {len(full_tokens):,} full train tokens, {len(problems)} probe problems")

    all_rows = []
    for size in sizes:
        target_n_tokens = int(len(full_tokens) * size / 7473)
        sub_tokens = full_tokens[:target_n_tokens]
        size_out = base_out / f"{size}p"
        size_out.mkdir(parents=True, exist_ok=True)
        print(f"\n========== {size} problems ({len(sub_tokens):,} tokens) ==========")
        for variant in ("composite", "ar_only"):
            for seed in seeds:
                run_out = size_out / f"{variant}_seed{seed}"
                ckpt = run_out / "model.pt"
                if not ckpt.exists():
                    print(f"\n[{size}p {variant} seed={seed}] training...")
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
                else:
                    print(f"[{size}p {variant} seed={seed}] ckpt exists, skip train")

                # Score AR-NLL
                ck = torch.load(ckpt, map_location=device, weights_only=False)
                mcfg = CompositeConfig(**ck["config"])
                model = CompositeLM(mcfg).to(device)
                model.load_state_dict(ck["state_dict"])
                model.train(False)
                r = score_ar_nll(model, problems, device, max_len=block_size)
                row = {"size": size, "variant": variant, "seed": seed,
                       "steps": max_steps, "n_train_tokens": len(sub_tokens), **r}
                all_rows.append(row)
                print(f"  AR-NLL = {r['avg_nll']}")
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

                (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    # Final aggregate
    print("\n========== FINAL AGGREGATE (Δ = composite − ar_only) ==========")
    by_size = {}
    for r in all_rows:
        by_size.setdefault(r["size"], {}).setdefault(r["variant"], []).append(r["avg_nll"])
    for size in sorted(by_size):
        comp_nlls = by_size[size].get("composite", [])
        ar_nlls = by_size[size].get("ar_only", [])
        if not comp_nlls or not ar_nlls:
            continue
        comp_m = sum(comp_nlls) / len(comp_nlls)
        ar_m = sum(ar_nlls) / len(ar_nlls)
        delta = comp_m - ar_m
        print(f"  {size}p: composite={comp_m:.3f} ar_only={ar_m:.3f} Δ={delta:+.3f} n={len(comp_nlls)}")

    print(f"\nwrote {base_out/'summary.json'}")


if __name__ == "__main__":
    main()
