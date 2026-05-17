"""F10 — 305M composite trained on FineWeb-Edu MIXED with 5% GSM8K Q/A.

The F9 model trained on raw FineWeb-Edu has zero exposure to "Question: ...\\n
Answer: ..." structure. At eval time it produces topical but task-free prose
(see e5/results/T0_PROBES_FINAL.md "Phase H+ sample-quality investigation").

F10 fixes the train/eval distribution mismatch by interleaving GSM8K Q/A into
the FineWeb stream via e5/data.py::load_mixed_tokens:
  - 95% FineWeb-Edu prose (2.85B tokens)
  - 5% GSM8K-train formatted as "Question: ...\\nAnswer: ...\\n#### N\\n<EOT>"
    (gsm8k_repeats=20 copies of the 7,473 training problems ≈ 80M Q/A tokens)
  - Same total ~3B token budget as F9
  - Same 305M composite architecture

If F9 produced loops on Q/A prompts because the format was OOD, F10 should:
  1. Produce non-loopy, on-topic answers (model knows the "Answer: ..." format)
  2. Lift GSM8K-dev free-run accuracy above F9's 0% floor
  3. Validate Phase H+ Tier 3 prediction

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
    os.environ.setdefault("N_TARGET_TOKENS", "3000000000")
    os.environ.setdefault("D_MODEL", "1024")
    os.environ.setdefault("N_LAYERS", "20")
    os.environ.setdefault("N_HEADS", "16")
    os.environ.setdefault("BATCH_SIZE", "16")
    os.environ.setdefault("BLOCK_SIZE", "1024")
    os.environ.setdefault("PEAK_LR", "4e-4")
    os.environ.setdefault("SEED", "310")
    os.environ.setdefault("OUT_NAME", "f10_mixed")
    # F10-specific: use load_mixed_tokens instead of load_fineweb_tokens.
    # F7's main() reads this env knob; defaults to "fineweb" if unset.
    os.environ.setdefault("DATA_LOADER", "mixed")
    os.environ.setdefault("MIXED_GSM8K_REPEATS", "20")
    os.environ.setdefault("MIXED_FINEWEB_TOKENS", "2850000000")
    from e5.scripts.f7_1b_emerge import main as f7_main
    f7_main()


if __name__ == "__main__":
    main()
