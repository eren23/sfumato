"""Placeholder for BD3-LMs commit-LoRA training — NOT IMPLEMENTED.

Phase-4 T2.C scaffold counterpart of `scripts/train_track2_commit.py`.
Eventual purpose: train a fresh 14M-param commit-LoRA on BD3-base + SFT
so the K-sweep cross-substrate test (cmajc k=2/3/4 vs c2c k=0) can run
on a non-LLaDA mask-diffusion LM.

Currently exits 0 after printing the eng-prereq table. This file exists
so the dispatch infrastructure (Crucible BYO-trainer + sfumato runner)
can be wired up end-to-end against a real script path before the actual
training code lands.

PRE_REG: phase2/spikes/bd3lms-cross-substrate/PRE_REG.md  (commit 998b0b0)

================================================================
TODO(T2.C-eng-prereq): six engineering phases, ~$40-60 + 14-21 days
================================================================
  Phase 1: fork BD3 Hydra repo, set up local + pod env       3-4 days, $0
  Phase 2: SFT BD3-base on GSM8K-train (~7.5k examples)      2-3 days, $15-25
  Phase 3: add PEFT/LoRA infra to BD3 model class            3-4 days, $0
  Phase 4: train commit-LoRA on BD3-base+SFT (THIS SCRIPT)   2-3 days, $5-10
  Phase 5: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0)       1 day,    $2-3
  Phase 6: buffer (debugging + paper write-up)               2-3 days, $10-20

This script is the Phase-4 entrypoint. It cannot run until Phases 1-3
land first.
"""

from __future__ import annotations

import sys


PREREQ_TABLE = """\
T2.C BD3-LMs commit-LoRA training — eng-prereq table
=====================================================
  Phase 1: fork BD3 Hydra repo + local/pod env             3-4 days,  $0
  Phase 2: SFT BD3-base on GSM8K-train (~7.5k examples)    2-3 days,  $15-25
  Phase 3: add PEFT/LoRA infra to BD3 model class          3-4 days,  $0
  Phase 4: train commit-LoRA on BD3-base+SFT  <-- THIS     2-3 days,  $5-10
  Phase 5: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0)     1 day,     $2-3
  Phase 6: buffer (debugging + paper write-up)             2-3 days,  $10-20
  -----------------------------------------------------------------------
  Total                                                    14-21 days, $40-60

Status: BLOCKED on Phases 1-3.

Direction A (schedule-RLHF) is the prioritized Phase-4 bet; T2.C
stays scaffolded but unfunded until budget approval.

See: phase2/spikes/bd3lms-cross-substrate/PRE_REG.md (commit 998b0b0)
"""


def main() -> int:
    """Print the eng-prereq table and exit 0.

    TODO(T2.C-eng-prereq Phase 4): once Phases 1-3 land, replace this
    function with the actual training loop. Recipe should mirror
    `scripts/train_track2_commit.py`:
      - LoRA r=8, alpha=16, FFN-only (gate/up/down proj equivalent in BD3 DiT)
      - target the last 8 of BD3's transformer layers (analogous to
        LLaDA's layers 24-31)
      - mask probability U(0.3, 0.9)  -- consensus regime
      - loss on answer span only (delimited by 'Answer:' or '#### ')
      - dataset: regenerate `eren23/sfumato-consensus-gsm8k` equivalent
        from BD3-base+SFT cmaj b=5 majority-vote outputs that disagree
        with greedy
      - HF artifact target: `eren23/sfumato-bd3-commit-v1`
    """
    print(PREREQ_TABLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
