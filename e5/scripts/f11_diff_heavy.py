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
    # RESUME_FROM is set by the launcher (points to the F10 ckpt on pod).
    from e5.scripts.f7_1b_emerge import main as f7_main
    f7_main()


if __name__ == "__main__":
    main()
