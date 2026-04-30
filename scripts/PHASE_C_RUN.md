# Phase C — pod runs to close paper revision

Three runs, total ~$0.42, ~75 min on a fresh RTX 4090 spot pod.

## C1. cmajc on commit-v3 — REQUIRED, ~$0.20, ~60min

Closes the §6.5 compositionality story. cmajc-v2 = 82.0% is the cleanest
number we have; cmajc-v3 is the matching number for the c2c-fixed design.

```bash
CONDITION=cmajc \
  HF_LORA_REPO=eren23/sfumato-prefix-robust-gsm8k-v3 \
  HF_COMMIT_LORA_REPO=eren23/sfumato-llada-commit-v3 \
  COMMIT_N_BLOCKS=3 \
  BRANCHES=5 TEMP=0.7 \
  K_STEPS=64 N_PROBLEMS=200 \
  python e4/runner.py
```

Decision rule:
- `cmajc-v3 >= 79.0%` → paste into LaTeX `consensus_distillation.tex` §6.5
  alongside cmajc-v2; update abstract.
- `cmajc-v3 < 79.0%` but `>= 79.0%` (base test cmaj) → keep cmajc-v2 as
  cleanest signal, add cmajc-v3 below with one-sentence note.
- `cmajc-v3 < 79.0%` → that *is* a finding (commit-v3 and branch
  aggregation cannibalize each other); say so clearly.

## C2. ABL_B sanity probe — REQUIRED, ~$0.02, ~5min

Confirms the ABL_B adapter actually loads and modifies inference.

```bash
N_PROBLEMS=5 K_STEPS=64 python scripts/abl_b_sanity.py
```

Decision rule:
- ≥3/5 problems show text difference → adapter fires; ABL_B = +0.0pp is
  the mechanism story (commit lands downstream of pinned reasoning).
- 0/5 differ → silent loader failure; rerun ABL_B and rewrite §6.3
  before submission.

## C3. v1/v2 error-set diff — OPTIONAL, free if JSONLs exist

Currently no per-problem JSONLs at k=64 in `e2/results/`. Two paths:

(a) Pull v1/v2 c2c per-problem outputs from W&B run artifacts (run
    URLs in `e2/RESULTS_TRACK2.md`).
(b) Re-run v1 and v2 c2c with per-problem logging (~$0.05).

Then intersect failure sets. Pick the right `open_exposures.tex`
footnote variant (currently a hedge that admits non-determination).

## Where the numbers land in the LaTeX

After C1 lands, edits go in:
- `paper/sections/consensus_distillation.tex` §6.5 (line ~205)
- `paper/main.tex` abstract (the `+3$\,pp` sentence)
- `paper/appendix/open_exposures.tex` (strike the cmajc-v3 paragraph)

After C2 lands, optional improvement to:
- `paper/appendix/open_exposures.tex` (strike or upgrade the ABL_B
  sanity paragraph based on probe outcome)

The Phase A/B paper edits are already in place; these runs are
risk-reduction, not headline shifts.
