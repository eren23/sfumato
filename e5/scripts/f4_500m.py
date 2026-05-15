"""F4 — 500M-scale composite vs baselines, full GSM8K.

The scale leap. E3b showed composite advantage grows with scale: +0.35
at 60M, +0.74 at 300M. F4 tests whether the trend continues at 500M
and quantifies the AR tax at this scale.

Trains 5 variants at 500M for 3k steps on full GSM8K-train (~4M tokens),
plus compute-matched controls:

  composite-3k     × 3 seeds
  ar_only-3k       × 3 seeds (baseline for AR axis)
  diff_only-3k     × 3 seeds (baseline for diff axis)
  ar_only-6k       × 3 seeds (compute-matched AR control, mirror of E3b)
  diff_only-6k     × 3 seeds (compute-matched diff control, mirror of F3)

Scores both AR-NLL (ar_only-axis) and diff-NLL (diff-axis) for each.

Expected ~3.5 hours on a single A40.

Env:
  SEEDS=170,171,172
  MAX_STEPS_3K=3000
  MAX_STEPS_6K=6000
  OUT_NAME=f4_500m
  D_MODEL=1280  N_LAYERS=24  N_HEADS=20
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


def run_one(variant, seed, steps, label, base_out, tokens, probes, device,
            batch_size, block_size, d_model, n_layers, n_heads):
    run_out = base_out / label / f"seed{seed}"
    ckpt = run_out / "model.pt"
    if not ckpt.exists():
        print(f"\n[{label} seed={seed}] training {steps} steps...")
        t0 = time.time()
        train_one(
            variant=variant,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            out_dir=run_out,
            seed=seed, max_steps=steps,
            batch_size=batch_size, block_size=block_size,
            peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=tokens,
        )
        print(f"  train wall_s={round(time.time()-t0, 1)}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    mcfg = CompositeConfig(**ck["config"])
    model = CompositeLM(mcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())
    ar_nll = score_ar_nll(model, probes, device, max_len=block_size)
    diff_nll = score_diff_nll(model, probes, device, max_len=block_size, seed=seed)
    print(f"  n_params={n_params/1e6:.1f}M  AR-NLL={ar_nll}  diff-NLL={diff_nll}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "variant": variant, "seed": seed, "steps": steps,
            "n_params": n_params, "ar_nll": ar_nll, "diff_nll": diff_nll}


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "170,171,172").split(",")]
    steps_3k = env_int("MAX_STEPS_3K", 3000)
    steps_6k = env_int("MAX_STEPS_6K", 6000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    d_model = env_int("D_MODEL", 1280)
    n_layers = env_int("N_LAYERS", 24)
    n_heads = env_int("N_HEADS", 20)

    out_name = os.environ.get("OUT_NAME", "f4_500m")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={seeds} d={d_model} L={n_layers} H={n_heads}")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    probes = load_gsm8k_probe_problems()
    print(f"loaded {len(tokens):,} train tokens, {len(probes)} probe problems")

    cells = []
    # 3k-step cells
    for seed in seeds:
        cells.append(("composite", seed, steps_3k, "composite_3k"))
        cells.append(("ar_only",   seed, steps_3k, "ar_only_3k"))
        cells.append(("diff_only", seed, steps_3k, "diff_only_3k"))
    # 6k-step compute-matched cells
    for seed in seeds:
        cells.append(("ar_only",   seed, steps_6k, "ar_only_6k"))
        cells.append(("diff_only", seed, steps_6k, "diff_only_6k"))

    all_rows = []
    for (variant, seed, steps, label) in cells:
        row = run_one(variant, seed, steps, label, base_out, tokens, probes, device,
                      batch_size, block_size, d_model, n_layers, n_heads)
        all_rows.append(row)
        (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    # Aggregate
    print("\n========== FINAL AGGREGATE (500M, GSM8K full) ==========")
    by_label = {}
    for r in all_rows:
        by_label.setdefault(r["label"], []).append(r)
    for label in ("composite_3k", "ar_only_3k", "diff_only_3k", "ar_only_6k", "diff_only_6k"):
        rs = by_label.get(label, [])
        if not rs:
            continue
        ar_m = sum(r["ar_nll"] for r in rs)/len(rs)
        diff_m = sum(r["diff_nll"] for r in rs)/len(rs)
        print(f"  {label}: n={len(rs)}  AR-NLL={ar_m:.3f}  diff-NLL={diff_m:.3f}")

    # Trade-off comparison
    c3k = by_label.get("composite_3k", [])
    ar3k = by_label.get("ar_only_3k", [])
    ar6k = by_label.get("ar_only_6k", [])
    d3k = by_label.get("diff_only_3k", [])
    d6k = by_label.get("diff_only_6k", [])
    if c3k and ar3k and ar6k and d3k and d6k:
        c_ar = sum(r["ar_nll"] for r in c3k)/len(c3k)
        c_diff = sum(r["diff_nll"] for r in c3k)/len(c3k)
        ar3_ar = sum(r["ar_nll"] for r in ar3k)/len(ar3k)
        ar6_ar = sum(r["ar_nll"] for r in ar6k)/len(ar6k)
        d3_diff = sum(r["diff_nll"] for r in d3k)/len(d3k)
        d6_diff = sum(r["diff_nll"] for r in d6k)/len(d6k)
        print("\n--- Trade-off summary at 500M ---")
        print(f"  AR-axis tax (composite-3k vs ar_only-3k): {c_ar - ar3_ar:+.3f} NLL")
        print(f"  Compute-matched AR (composite-3k vs ar_only-6k): {c_ar - ar6_ar:+.3f} NLL")
        print(f"  Diff-axis advantage (composite-3k vs diff_only-3k): {c_diff - d3_diff:+.3f} NLL")
        print(f"  Compute-matched diff (composite-3k vs diff_only-6k): {c_diff - d6_diff:+.3f} NLL")


if __name__ == "__main__":
    main()
