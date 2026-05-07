"""DiffuLLaMA commit-LoRA training — SCAFFOLD with --smoke mock path.

Phase-4 T2.C-light scaffold counterpart of `scripts/train_track2_commit.py`
(LLaDA) and `scripts/train_bd3_commit_lora.py` (BD3-LMs sister scaffold).

Eventual purpose: train a fresh ~14M-param commit-LoRA on top of
`diffusionfamily/diffullama` (or stacked on `diffusionfamily/diffullama-gsm`
as Track-1 base), so the K-sweep cross-substrate test (cmajc k=2/3/4
vs c2c k=0) can run on a non-LLaDA mask-diffusion LM.

Currently:
  - `--smoke` runs a CPU-only mock training loop in <30s, exercising
    the dispatch infrastructure (Crucible BYO-trainer markers, sfumato
    runner) without touching real model weights. Honors MOCK_MODELS=1.
  - Default invocation prints the eng-prereq table and exits 0.

The mock path is functional end-to-end so the runner can wire against
a real script path before the actual training code lands. Compared to
BD3's training scaffold (which is print-and-exit-only because BD3's
PEFT-from-scratch prereq is the blocker), DiffuLLaMA's training is
mostly a fork of `train_track2_commit.py` with three deltas:

  1. MODEL_NAME: "diffusionfamily/diffullama" (vs LLaDA-8B-Instruct)
  2. LORA_TARGETS: standard LLaMA-2 ["gate_proj", "up_proj", "down_proj"]
     (vs LLaDA's custom ["ff_proj", "up_proj", "ff_out"])
  3. LORA_LAYERS_TO_TRANSFORM: list(range(24, 32)) — DiffuLLaMA-base
     is LLaMA-2 7B with 32 layers, same as LLaDA-8B (which is also
     ~32 layers). Last 8 layers, same recipe as Track-2 LLaDA.

PRE_REG: phase2/spikes/diffullama-cross-substrate/PRE_REG.md (this commit)

================================================================
TODO(T2.C-light): three engineering phases, ~$15-25 + 3-7 days
================================================================
  Phase 1: clone HKUNLP/DiffuLLaMA, audit AutoModel compat       1-2 days, $0
  Phase 2: port `inf_diffullama.py` sampler → `_generate(...)`   1-2 days, $0
  Phase 3: train commit-LoRA on diffullama-gsm  <-- THIS         1-2 days, $10-15
  Phase 4: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0)           <1 day,   $2-3
  Buffer (debugging + paper write-up)                            1-2 days, $3-7

If Phase 1 confirms `AutoModelForCausalLM.from_pretrained(model_id,
trust_remote_code=True)` works, Phase 3 reduces to a near-mechanical
fork of `train_track2_commit.py` (~50 LOC of edits). Standard LLaMA-2
module names mean PEFT works out of the box — no LLaDA-style fork
quirk.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time


PREREQ_TABLE = """\
T2.C-light DiffuLLaMA commit-LoRA training — eng-prereq table
============================================================
  Phase 1: clone HKUNLP/DiffuLLaMA, audit AutoModel compat       1-2 days, $0
  Phase 2: port `inf_diffullama.py` sampler -> `_generate(...)`  1-2 days, $0
  Phase 3: train commit-LoRA on diffullama-gsm  <-- THIS         1-2 days, $10-15
  Phase 4: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0)           <1 day,   $2-3
  Buffer (debugging + paper write-up)                            1-2 days, $3-7
  -------------------------------------------------------------------------
  Total                                                          3-7 days, $15-25

Status: BLOCKED on Phases 1-2 (eng audit + sampler port).
        Phase 3 fork-from-train_track2_commit is mostly mechanical
        once Phase 1 confirms AutoModel compat.

vs BD3 T2.C ($40-60 + 14-21 days): DiffuLLaMA saves the SFT-from-
scratch ($15-25) and PEFT-from-scratch ($0 eng but multi-day) prereqs
because `diffusionfamily/diffullama-gsm` LoRA already exists and
DiffuLLaMA repo ships LoRA training configs.

See: phase2/spikes/diffullama-cross-substrate/PRE_REG.md
"""


# Mirror Track-2 LLaDA recipe knobs for downstream parity.
DEFAULT_MODEL_NAME = "diffusionfamily/diffullama"
DEFAULT_TRACK1_LORA = "diffusionfamily/diffullama-gsm"  # GSM-symbolic LoRA base
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_TARGETS = ["gate_proj", "up_proj", "down_proj"]  # LLaMA-2 standard
DEFAULT_LORA_LAYERS = list(range(24, 32))  # last 8 of 32, mirror Track-2 LLaDA
DEFAULT_OUTPUT_REPO = "eren23/sfumato-diffullama-commit-v1"


def _smoke_mock_training_loop(steps: int = 8) -> int:
    """CPU-only mock training loop — exercises Crucible BYO-trainer markers.

    Runs in <30s. No model load, no GPU, no HF download. Emits the same
    stdout markers Crucible looks for (step:N/M train_loss:X) so the
    dispatch chain can be wired end-to-end before the real Phase-3
    training code lands.

    Honors MOCK_MODELS=1 by always succeeding without side effects.
    """
    print("=== DiffuLLaMA commit-LoRA SMOKE (mock) ===")
    print(f"  MODEL_NAME       = {DEFAULT_MODEL_NAME}")
    print(f"  TRACK1_LORA      = {DEFAULT_TRACK1_LORA}")
    print(f"  LORA_R           = {DEFAULT_LORA_R}")
    print(f"  LORA_ALPHA       = {DEFAULT_LORA_ALPHA}")
    print(f"  LORA_TARGETS     = {DEFAULT_LORA_TARGETS}")
    print(f"  LORA_LAYERS      = {DEFAULT_LORA_LAYERS}")
    print(f"  OUTPUT_REPO      = {DEFAULT_OUTPUT_REPO}")
    print(f"  MOCK_MODELS env  = {os.environ.get('MOCK_MODELS', '0')}")
    print()

    rng = random.Random(0)  # deterministic for CI
    t0 = time.time()
    for step in range(1, steps + 1):
        # Synthetic loss curve: starts ~2.5, decays geometrically with noise.
        loss = 2.5 * (0.85 ** (step - 1)) + rng.uniform(-0.05, 0.05)
        print(f"step:{step}/{steps} train_loss:{loss:.4f}")
        if step % max(1, steps // 4) == 0:
            val = loss + rng.uniform(0.02, 0.08)
            print(f"step:{step}/{steps} val_loss:{val:.4f} val_bpb:{step * 1024}")
        # Sleep ~10ms per step so the smoke takes a noticeable but
        # bounded amount of time.
        time.sleep(0.01)

    elapsed = time.time() - t0
    print()
    print(f"Serialized model /tmp/diffullama_commit_smoke 0 (mock)")
    print(f"smoke OK in {elapsed:.2f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: scaffold-only by default, mock training under --smoke.

    TODO(T2.C-light Phase 3): once Phase 1 (AutoModel audit) and
    Phase 2 (sampler port) land, replace `_real_training()` below with
    a fork of `scripts/train_track2_commit.py` whose only deltas are:
      - MODEL_NAME = "diffusionfamily/diffullama"
      - LORA_TARGETS = ["gate_proj", "up_proj", "down_proj"]
        (LLaMA-2 standard names, no LLaDA fork-quirk)
      - layers_to_transform = range(24, 32) (last 8 of 32)
      - mask probability U(0.3, 0.9)  -- consensus regime, same as LLaDA
      - loss on answer span only (delimited by 'Answer:' or '#### ')
      - dataset: regenerate `eren23/sfumato-consensus-gsm8k` equivalent
        from DiffuLLaMA-base+gsm cmaj b=5 majority-vote outputs that
        disagree with greedy
      - HF artifact target: `eren23/sfumato-diffullama-commit-v1`
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a CPU-only mock training loop (<30s). Used for "
        "dispatch-chain wiring before the real Phase-3 code lands.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of mock training steps under --smoke (default: 8).",
    )
    args = parser.parse_args(argv)

    if args.smoke or os.environ.get("MOCK_MODELS") == "1":
        return _smoke_mock_training_loop(steps=args.steps)

    # Default invocation: print the prereq table and exit 0.
    # Phase 3 implementation goes here once Phases 1-2 land.
    print(PREREQ_TABLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
