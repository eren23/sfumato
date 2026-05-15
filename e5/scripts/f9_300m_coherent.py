"""F9 — 300M composite on 6B FineWeb tokens for coherent-paragraph generation.

Target: a model that produces multi-sentence coherent prose (not GSM8K math,
just readable English) so we can see what "the composite trade-off characterisation
study" actually looks like at a scale closer to compute-optimal.

Arch: d=1024, L=20, H=16 → 305M params
Recipe: BS=32 × T=1024, peak_lr=4e-4, ~183k steps → ~6B tokens trained on
Hardware: 1× A40, ~20 hours wall, ~$9 cost

ENV defaults match the writeup; override as needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    os.environ.setdefault("VARIANTS", "composite")
    os.environ.setdefault("MAX_STEPS", "183000")
    os.environ.setdefault("N_TARGET_TOKENS", "6000000000")
    os.environ.setdefault("D_MODEL", "1024")
    os.environ.setdefault("N_LAYERS", "20")
    os.environ.setdefault("N_HEADS", "16")
    os.environ.setdefault("BATCH_SIZE", "32")
    os.environ.setdefault("BLOCK_SIZE", "1024")
    os.environ.setdefault("PEAK_LR", "4e-4")
    os.environ.setdefault("SEED", "300")
    os.environ.setdefault("OUT_NAME", "f9_300m_coherent")
    from e5.scripts.f7_1b_emerge import main as f7_main
    f7_main()


if __name__ == "__main__":
    main()
