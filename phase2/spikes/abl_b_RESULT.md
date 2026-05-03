# ABL_B Sanity Probe — Result

**Run date:** 2026-05-03 ~16:13 UTC
**Spike:** Phase-1 paper open-exposure closer (ABL_B = "is the commit-LoRA actually loading and modifying inference, ruling out silent no-op?").
**Outcome:** **PASS 5/5** (≥3/5 needed per pre-reg in the script).

## Procedure (per `scripts/abl_b_sanity.py`, fixed during night-1 — see commit `27da1c9`)

For each of 5 GSM8K-dev problems (frozen indices from `e4/data/gsm8k_dev_200.json`):
1. Run LLaDA-8B-Instruct + Track-1-v3 LoRA at temp=0, k=64, NO commit-LoRA → record output text
2. Run LLaDA-8B-Instruct + Track-1-v3 LoRA + commit-v3 LoRA (commit_n_blocks=1) at temp=0, k=64 → record output text
3. Compare: do the two outputs differ?

## Results (5/5 problems show text difference)

```
[1/5] differs=True
  no_commit_tail:    'resh duck egg, so she makes 9 * $2 = $18 every day.\n\n#### 18'
  with_commit_tail:  'sh duck egg, so she makes 9 * $2 = $18 every day.\nAnswer: 18'
[2/5] differs=True
  no_commit_tail:    'ber to the amount of white fiber:\n\\[ 2 + 1 = 3 \\]\n\nAnswer: 3'
  with_commit_tail:  'er to the amount of white fiber:\n\n\\[ 2 + 1 = 3 \\]\n\nAnswer: 3'
[3/5] differs=True
  no_commit_tail:    ',000\nSo his profit was 195,000-130,000 = $65,000\n\n#### 65,00'
  with_commit_tail:  '0\nSo his profit was 195,000-130,000 = $65,000\nAnswer: 65,000'
[4/5] differs=True
  no_commit_tail:    ' sprints/week * 60 meters/sprint = 540 meters/week\n\n#### 540'
  with_commit_tail:  'prints/week * 60 meters/sprint = 540 meters/week\nAnswer: 540'
[5/5] differs=True
  no_commit_tail:    's - 40 cups = 20 cups in the final meal of the day.\n\n#### 20'
  with_commit_tail:  's = 20 cups of feed in the final meal of the day.\nAnswer: 20'
```

## Verdict: PASS

**5/5 problems show text difference between Track-1-v3 alone vs Track-1-v3 + commit-v3.** Pre-reg pass criterion was ≥3/5 — passes with margin.

The systematic format shift across all 5 problems is **`#### {N}` → `Answer: {N}`**, which is the EXACT format the commit-LoRA was trained to produce (per Track-2 paper-prep work). The commit adapter is loading and applying its trained behavior at greedy decoding (temp=0). No silent no-op.

## What this closes

Phase-1 paper appendix had a reviewer concern (per `<sfumato>/scripts/abl_b_sanity.py` docstring): "ABL_B = +0.0pp delta versus Track-1 v3 alone is doing real conceptual work in the mechanism story. Confirm the adapter actually loads and modifies inference, ruling out silent-no-op load."

This concern is now **definitively closed**: commit-v3 is doing real work, just not work that improves cmaj-vote-extracted accuracy (which scores by extracted numeric answer regardless of format prefix). The format shift is itself a useful signal — it suggests the commit-LoRA shifted the *trajectory distribution* even when the *final extracted answer* didn't change.

This is consistent with the Phase-1 paper's "structural separation" finding: commit-LoRA changes WHAT the model writes (format, prefix), even when it doesn't change WHICH answer the runner extracts.

## Cost

~$0.05 of pod time (5 problems × ~10s each at temp=0, plus model load).

Total spike contribution to Phase-2 ledger: ~$0.05.

## Bug fix that enabled this run

Original `scripts/abl_b_sanity.py` had two bugs caught at first attempt:
1. `gsm8k_dev_200.json` is frozen-indices metadata (dict with `dataset/config/split/indices` keys), not a list of problems. Original code did `json.load(f)[:N]` which crashed `TypeError: unhashable type: 'slice'`. Fixed by mirroring `e4/runner.py:load_problems` — load HF dataset, slice by `spec["indices"][:N]`.
2. Default Track-1-v3 LoRA repo was `eren23/sfumato-prefix-robust-gsm8k-v3` (404). Correct: `eren23/sfumato-llada-prefix-robust-v3`.

Both fixed in commit `27da1c9` on `main`.
