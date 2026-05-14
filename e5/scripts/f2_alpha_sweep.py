"""F2 — alpha-schedule sweep.

Default composite uses alpha: 1.0 -> 0.5 linear over training. We sweep
three fixed-alpha settings (0.3, 0.5, 0.7) to test whether the default
schedule matters and what value is best.

Train at 200M, 3k steps on GSM8K-train (full). 3 seeds × 3 alphas.

Score both AR-NLL and diff-NLL on the held-out chunk. The interesting
plot is "alpha vs (AR-NLL, diff-NLL) at the same seed".

Env:
  ALPHAS=30,50,70
  SEEDS=120,121,122
  MAX_STEPS=3000
  OUT_NAME=f2_alpha_sweep
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
from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402


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
    total = 0.0; cnt = 0
    for prompt_ids, ans_ids in problems:
        if len(prompt_ids) + len(ans_ids) + 1 > max_len: continue
        full = prompt_ids + ans_ids
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        ans_start = len(prompt_ids)
        pred = logits[0, ans_start - 1 : ans_start - 1 + len(ans_ids), :].float()
        target = torch.tensor(ans_ids, dtype=torch.long, device=device)
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total += float(nlls.sum()); cnt += int(nlls.size)
    return round(total / max(1, cnt), 4)


@torch.no_grad()
def score_diff_nll(model, problems, device, max_len=256, mask_ratio=0.5, seed=0):
    total = 0.0; cnt = 0
    rng = np.random.RandomState(seed)
    for prompt_ids, ans_ids in problems:
        if len(prompt_ids) + len(ans_ids) + 1 > max_len: continue
        full = prompt_ids + ans_ids
        ans_start = len(prompt_ids)
        ans_len = len(ans_ids)
        n_mask = max(1, int(ans_len * mask_ratio))
        positions = rng.choice(ans_len, size=n_mask, replace=False) + ans_start
        idx = torch.tensor([full], dtype=torch.long, device=device)
        masked = idx.clone()
        for p in positions:
            masked[0, p] = MASK_TOKEN_ID
        logits = model(masked, mode="diff")
        pred = logits[0, positions].float()
        target = torch.tensor([full[p] for p in positions], dtype=torch.long, device=device)
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total += float(nlls.sum()); cnt += int(nlls.size)
    return round(total / max(1, cnt), 4)


def main():
    alphas = [int(x) for x in os.environ.get("ALPHAS", "30,50,70").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "120,121,122").split(",")]
    max_steps = env_int("MAX_STEPS", 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    d_model = env_int("D_MODEL", 896)
    n_layers = env_int("N_LAYERS", 12)
    n_heads = env_int("N_HEADS", 16)

    out_name = os.environ.get("OUT_NAME", "f2_alpha_sweep")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} alphas={alphas} seeds={seeds}")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    problems = load_gsm8k_probe_problems()
    print(f"loaded {len(tokens):,} train tokens, {len(problems)} probe problems")

    all_rows = []
    for alpha_pct in alphas:
        variant = f"composite_fixed_{alpha_pct}"
        for seed in seeds:
            run_out = base_out / variant / f"seed{seed}"
            ckpt = run_out / "model.pt"
            if not ckpt.exists():
                print(f"\n[{variant} seed={seed}] training...")
                t0 = time.time()
                train_one(
                    variant=variant,
                    d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    out_dir=run_out,
                    seed=seed, max_steps=max_steps,
                    batch_size=batch_size, block_size=block_size,
                    peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=tokens,
                )
                print(f"  train wall_s={round(time.time()-t0, 1)}")

            ck = torch.load(ckpt, map_location=device, weights_only=False)
            mcfg = CompositeConfig(**ck["config"])
            model = CompositeLM(mcfg).to(device)
            model.load_state_dict(ck["state_dict"])
            model.train(False)
            ar_nll = score_ar_nll(model, problems, device, max_len=block_size)
            diff_nll = score_diff_nll(model, problems, device, max_len=block_size, seed=seed)
            row = {"alpha_pct": alpha_pct, "variant": variant, "seed": seed,
                   "steps": max_steps, "ar_nll": ar_nll, "diff_nll": diff_nll}
            all_rows.append(row)
            print(f"  alpha={alpha_pct/100:.2f}: AR-NLL={ar_nll} diff-NLL={diff_nll}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    print("\n========== FINAL AGGREGATE ==========")
    by = {}
    for r in all_rows:
        by.setdefault(r["alpha_pct"], []).append(r)
    for alpha_pct in sorted(by):
        rows = by[alpha_pct]
        ar_m = sum(r["ar_nll"] for r in rows) / len(rows)
        diff_m = sum(r["diff_nll"] for r in rows) / len(rows)
        print(f"  alpha={alpha_pct/100:.2f}: AR-NLL={ar_m:.3f} diff-NLL={diff_m:.3f} n={len(rows)}")


if __name__ == "__main__":
    main()
