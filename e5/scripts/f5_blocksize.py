"""F5 — block-size (context length) ablation.

Composite + ar_only at BLOCK_SIZE = 128 vs 256, 200M-3k, 3 seeds each.
Tests whether the trade-off persists at shorter context. Shorter context
means fewer training tokens per step but tighter dependencies.

Env:
  BLOCK_SIZES=128,256
  SEEDS=150,151,152
  MAX_STEPS=3000
  OUT_NAME=f5_blocksize
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


def main():
    blocks = [int(x) for x in os.environ.get("BLOCK_SIZES", "128,256").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "150,151,152").split(",")]
    max_steps = env_int("MAX_STEPS", 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    d_model = env_int("D_MODEL", 896)
    n_layers = env_int("N_LAYERS", 12)
    n_heads = env_int("N_HEADS", 16)

    out_name = os.environ.get("OUT_NAME", "f5_blocksize")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    problems = load_gsm8k_probe_problems()
    print(f"device={device} blocks={blocks} seeds={seeds}")

    all_rows = []
    for block_size in blocks:
        bsize_out = base_out / f"block{block_size}"
        bsize_out.mkdir(parents=True, exist_ok=True)
        for variant in ("composite", "ar_only"):
            for seed in seeds:
                run_out = bsize_out / f"{variant}_seed{seed}"
                ckpt = run_out / "model.pt"
                if not ckpt.exists():
                    print(f"\n[block={block_size} {variant} seed={seed}] training...")
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
                row = {"block_size": block_size, "variant": variant, "seed": seed,
                       "steps": max_steps, "ar_nll": ar_nll}
                all_rows.append(row)
                print(f"  AR-NLL = {ar_nll}")
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
                (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    print("\n========== FINAL AGGREGATE ==========")
    by = {}
    for r in all_rows:
        by.setdefault((r["block_size"], r["variant"]), []).append(r["ar_nll"])
    sizes = sorted({b for (b, _) in by})
    for b in sizes:
        c = by.get((b, "composite"), [])
        a = by.get((b, "ar_only"), [])
        if c and a:
            cm = sum(c)/len(c); am = sum(a)/len(a); d = cm - am
            print(f"  block={b}: composite={cm:.3f} ar_only={am:.3f} Δ={d:+.3f} n={len(c)}")


if __name__ == "__main__":
    main()
