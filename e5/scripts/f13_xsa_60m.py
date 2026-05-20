"""F13 — 60M composite WITH Exclusive Self Attention (XSA) from scratch.

Phase N.1: cheapest possible test of arxiv:2603.09078 (XSA — each token
forbidden from attending to its own position). Tests whether the
attention modification yields better K.2 routing signal at param-golf
scale where we have strong non-XSA baselines.

Arch matches the original T0 60M composite: d=512, L=8, H=8, BS=8,
T=256, ~3k steps. Same data (FineWeb-Edu cached, no GSM8K mix —
matches the T0 baseline exactly so the only changed knob is XSA).

ENV:
  USE_XSA=1                  (enable XSA in CompositeConfig)
  MAX_STEPS=3000
  N_TARGET_TOKENS=200000000  (200M FineWeb cache)
  BATCH_SIZE=8
  BLOCK_SIZE=256
  PEAK_LR=6e-4
  D_MODEL=512  N_LAYERS=8  N_HEADS=8
  SEED=0
  OUT_NAME=f13_xsa_60m
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    os.environ.setdefault("MAX_STEPS", "3000")
    os.environ.setdefault("N_TARGET_TOKENS", "200000000")
    os.environ.setdefault("BATCH_SIZE", "8")
    os.environ.setdefault("BLOCK_SIZE", "256")
    os.environ.setdefault("PEAK_LR", "6e-4")
    os.environ.setdefault("D_MODEL", "512")
    os.environ.setdefault("N_LAYERS", "8")
    os.environ.setdefault("N_HEADS", "8")
    os.environ.setdefault("SEED", "0")
    os.environ.setdefault("OUT_NAME", "f13_xsa_60m")
    os.environ.setdefault("USE_XSA", "1")
    os.environ.setdefault("DATA_LOADER", "fineweb")  # match T0 baseline, no GSM8K mix
    os.environ.setdefault("SAVE_EVERY", "1500")

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from e5.train import train_one
    from e5.data import load_fineweb_tokens, load_gsm8k_dev_questions
    from e5.model_composite import CompositeConfig, CompositeLM

    # Patch CompositeConfig to flip use_xsa from env. train.py builds the
    # config from explicit kwargs, so we need a small wrapper.
    use_xsa = os.environ.get("USE_XSA", "0") == "1"
    orig_init = CompositeConfig.__init__
    def _patched_init(self, **kw):
        kw.setdefault("use_xsa", use_xsa)
        orig_init(self, **kw)
    CompositeConfig.__init__ = _patched_init

    out_name = os.environ["OUT_NAME"]
    variant = "composite"
    base_out = REPO_ROOT / "e5" / "results" / out_name
    run_out = base_out / variant
    run_out.mkdir(parents=True, exist_ok=True)

    n_target = int(os.environ["N_TARGET_TOKENS"])
    seed = int(os.environ["SEED"])
    print(f"[F13] loading FineWeb tokens (~{n_target/1e9:.2f}B)...", flush=True)
    tokens = load_fineweb_tokens(n_tokens=n_target, seed=1337 + seed)
    print(f"[F13] {len(tokens):,} tokens loaded; USE_XSA={use_xsa}", flush=True)

    tok = AutoTokenizer.from_pretrained("gpt2")
    gsm_dev = load_gsm8k_dev_questions(n=20)

    train_one(
        variant=variant,
        d_model=int(os.environ["D_MODEL"]),
        n_layers=int(os.environ["N_LAYERS"]),
        n_heads=int(os.environ["N_HEADS"]),
        out_dir=run_out,
        seed=seed,
        max_steps=int(os.environ["MAX_STEPS"]),
        batch_size=int(os.environ["BATCH_SIZE"]),
        block_size=int(os.environ["BLOCK_SIZE"]),
        peak_lr=float(os.environ["PEAK_LR"]),
        eval_every=10**9,
        n_eval=0,
        tokens=tokens,
        val_problems=gsm_dev,
        sample_prompts=gsm_dev[:5],
        val_every=500,
        sample_every=1000,
        tokenizer_for_samples=tok,
        resume_from=None,
        save_every=int(os.environ["SAVE_EVERY"]),
    )


if __name__ == "__main__":
    main()
