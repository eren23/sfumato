"""ABL_B sanity probe.

Reviewer concern (Claude on Codex): ABL_B = +0.0pp delta versus
Track 1 v3 alone is doing real conceptual work in the mechanism
story. Confirm the adapter actually loads and modifies inference,
ruling out silent-no-op load.

Run on pod after pulling latest scripts. Estimated cost: ~$0.02
(5 problems × 2 conditions × ~10s each at temp=0).

Usage:
    python scripts/abl_b_sanity.py        # default 5 problems
    N_PROBLEMS=10 python scripts/abl_b_sanity.py

Output: how many of N problems decoded text differs between
"Track 1 v3 alone" and "Track 1 v3 + commit-v3 with n_blocks=1".

Pass criterion: at least 3/5 problems show text difference.
Fail criterion: 0/5 — adapter is silently not loading.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from e4 import diff_llada  # noqa: E402

BASE = "GSAI-ML/LLaDA-8B-Instruct"
TRACK1_V3 = os.environ.get("TRACK1_V3_REPO", "eren23/sfumato-prefix-robust-gsm8k-v3")
COMMIT_V3 = os.environ.get("COMMIT_V3_REPO", "eren23/sfumato-llada-commit-v3")

DEV_PATH = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"
N = int(os.environ.get("N_PROBLEMS", "5"))
K = int(os.environ.get("K_STEPS", "64"))


def main() -> None:
    with DEV_PATH.open() as f:
        problems = json.load(f)[:N]

    model = diff_llada.load(
        name=BASE,
        block_len=128,
        lora_path=TRACK1_V3,
        commit_lora_path=COMMIT_V3,
    )

    differs = 0
    for i, prob in enumerate(problems):
        q = prob["question"] if isinstance(prob, dict) else prob
        no_commit, _ = model.denoise_block(
            prompt=q,
            k_steps=K,
            seed=0,
            temperature=0.0,
            apply_commit=False,
        )
        with_commit, _ = model.denoise_block(
            prompt=q,
            k_steps=K,
            seed=0,
            temperature=0.0,
            apply_commit=True,
            commit_n_blocks=1,
        )
        is_diff = no_commit != with_commit
        differs += int(is_diff)
        print(
            f"[{i+1}/{N}] differs={is_diff}  "
            f"no_commit_tail='{no_commit[-60:]!r}'  "
            f"with_commit_tail='{with_commit[-60:]!r}'"
        )

    print()
    print(f"ABL_B sanity: {differs}/{N} problems show text difference.")
    print(
        "PASS" if differs >= max(3, N // 2)
        else "FAIL — adapter may be silently not loading"
    )


if __name__ == "__main__":
    main()
