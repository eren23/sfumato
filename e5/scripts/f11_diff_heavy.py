"""F11 — fine-tune F10 for 30k more steps with a diff-heavy α-schedule.

Phase K showed that percentile-ranked commit-time diff confidence is
the only routing signal that works on F10 at our scale. Absolute conf
signals fail because diff conf is universally low (mean ~0.03 across
50k vocab). F11 hypothesis: continue training with α fixed at 0.3
(70% of steps are diff steps, vs F10's average ~25%) to give the
diff head much more capacity to specialise — sharpening its conf
distribution and making K.2 routing more informative.

This is a fine-tune, NOT a from-scratch run. We resume from F10
final's optim state + RNG (per train.py's resumable checkpoint) and
push step count from 183k to 213k (30k additional steps).

Cost: ~$5-8 on A40 (~6-8h wall).

ENV:
  RESUME_FROM=path/to/f10/model.pt  (mandatory — F10 final ckpt with optim state)
  MAX_STEPS=213000                  (30k additional past F10's 183k)
  ALPHA_OVERRIDE=0.3                (fixed α; overrides default 1.0→0.5 schedule)
  All other knobs inherited from F10 / f7_1b_emerge.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    # F11 inherits F10's config; the deltas are MAX_STEPS, ALPHA_OVERRIDE, and
    # the resume path. RESUME_FROM must be set externally (or default to the
    # pod-side F10 ckpt below).
    os.environ.setdefault("VARIANTS", "composite_fixed_30")  # = composite with α=0.30 fixed
    os.environ.setdefault("MAX_STEPS", "213000")
    os.environ.setdefault("N_TARGET_TOKENS", "3000000000")
    os.environ.setdefault("D_MODEL", "1024")
    os.environ.setdefault("N_LAYERS", "20")
    os.environ.setdefault("N_HEADS", "16")
    os.environ.setdefault("BATCH_SIZE", "16")
    os.environ.setdefault("BLOCK_SIZE", "1024")
    os.environ.setdefault("PEAK_LR", "1e-4")  # lower LR for fine-tune
    os.environ.setdefault("SEED", "310")
    os.environ.setdefault("OUT_NAME", "f11_diff_heavy")
    os.environ.setdefault("DATA_LOADER", "mixed")
    os.environ.setdefault("MIXED_GSM8K_REPEATS", "20")
    os.environ.setdefault("MIXED_FINEWEB_TOKENS", "2850000000")
    os.environ.setdefault("SAVE_EVERY", "2500")

    # Bypass f7_1b_emerge's variant loop (it skips if ckpt exists, which
    # collides with the "pre-place F10 ckpt for resume" strategy). Call
    # train_one directly with explicit resume_from.
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from e5.train import train_one
    from e5.data import load_fineweb_tokens, load_mixed_tokens, load_gsm8k_dev_questions

    out_name = os.environ["OUT_NAME"]
    variant = os.environ["VARIANTS"].split(",")[0].strip()
    base_out = REPO_ROOT / "e5" / "results" / out_name
    run_out = base_out / variant
    run_out.mkdir(parents=True, exist_ok=True)
    target = run_out / "model.pt"

    resume_from = os.environ.get("RESUME_FROM")
    # If a "skip" ckpt from a previous bad-launch is sitting here, move it
    # aside so it does not collide. train_one writes model.pt as it trains.
    if resume_from and Path(resume_from).exists():
        if target.exists():
            # Previous bad-launch path; move it to model.pt.skip
            backup = run_out / "model.pt.prior_skip"
            print(f"[F11] moving prior {target.name} -> {backup.name}", flush=True)
            target.replace(backup)

    # Load tokens (will reuse cache if already on pod)
    data_loader = os.environ.get("DATA_LOADER", "fineweb").lower()
    if data_loader == "mixed":
        fineweb_target = int(os.environ.get("MIXED_FINEWEB_TOKENS", "2850000000"))
        gsm8k_repeats = int(os.environ.get("MIXED_GSM8K_REPEATS", "20"))
        seed = int(os.environ.get("SEED", "310"))
        print(f"[F11] loading mixed tokens fineweb={fineweb_target/1e9:.2f}B gsm8k_repeats={gsm8k_repeats}", flush=True)
        tokens = load_mixed_tokens(fineweb_tokens_target=fineweb_target,
                                   gsm8k_repeats=gsm8k_repeats, seed=1337 + seed)
    else:
        n_target = int(os.environ.get("N_TARGET_TOKENS", "3000000000"))
        seed = int(os.environ.get("SEED", "310"))
        tokens = load_fineweb_tokens(n_tokens=n_target, seed=1337 + seed)
    print(f"[F11] {len(tokens):,} tokens loaded", flush=True)

    tok = AutoTokenizer.from_pretrained("gpt2")
    gsm_probes = load_gsm8k_dev_questions(n=100)
    gsm_dev = load_gsm8k_dev_questions(n=10)

    print(f"[F11] launching train_one variant={variant} resume={resume_from}", flush=True)
    train_one(
        variant=variant,
        d_model=int(os.environ.get("D_MODEL", "1024")),
        n_layers=int(os.environ.get("N_LAYERS", "20")),
        n_heads=int(os.environ.get("N_HEADS", "16")),
        out_dir=run_out,
        seed=int(os.environ.get("SEED", "310")),
        max_steps=int(os.environ.get("MAX_STEPS", "213000")),
        batch_size=int(os.environ.get("BATCH_SIZE", "8")),
        block_size=int(os.environ.get("BLOCK_SIZE", "1024")),
        peak_lr=float(os.environ.get("PEAK_LR", "1e-4")),
        eval_every=10**9,
        n_eval=0,
        tokens=tokens,
        val_problems=gsm_probes,
        sample_prompts=gsm_dev,
        val_every=500,
        sample_every=1000,
        tokenizer_for_samples=tok,
        resume_from=Path(resume_from) if resume_from else None,
        save_every=int(os.environ.get("SAVE_EVERY", "2500")),
    )


if __name__ == "__main__":
    main()
