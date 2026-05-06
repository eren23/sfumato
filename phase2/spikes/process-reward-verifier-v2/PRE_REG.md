# Pre-registration — PRM verifier v2 (logit_shift_norm features)

**Date:** 2026-05-06 | **Spike:** Phase-3 follow-up to T1.B (LOSS at
commit `274c2fb`).

## Hypothesis

T1.B harvested per-step features (entropy mean/max + commit_lora_active
flag) and showed that an MLP reranker UNDERPERFORMS cmajc majority-vote
by 4pp on N=100 (`prm-rerank 0.76 vs cmajc-vote 0.80`). The amended
PRE_REG dropped `LOGIT_SHIFT_NORM=1` because the provisioner returned a
24GB pod and the shadow-forward OOMs.

This v2 re-runs the harvest on a 48GB pod with `LOGIT_SHIFT_NORM=1`
enabled. **Hypothesis:** the L2 norm of (logits_with_adapter -
logits_without_adapter) at the toggle boundary captures a
"did-the-adapter-actually-change-the-trajectory" signal that entropy
alone misses, and a PRM reranker including this feature closes ≥75% of
the cmajc voting-rule gap.

## Substrate

- **Source:** sfumato runner cmajc N=100 BATCHED=0 BRANCHES=5
  K_STEPS=64 TEMP=0.7 SEED=0 COMMIT_N_BLOCKS=3 TRACE_STEPS=1
  **LOGIT_SHIFT_NORM=1**.
- Same 100 problems as T1.B (`gsm8k_dev_200.json` idx 0..99).
- Pod: 48GB+ on-demand A6000 (24GB OOMs with shadow forward).
- Output: `e4/results/traces/cmajc-prm-v2-N100-seed0/branch_*_idx_*.jsonl`
  with logit_shift_norm populated per sub-block.

## Features (per branch trajectory)

Same 14 dims as T1.B PLUS new dims:

- `logit_shift_norm[1..3]` — 3 dims (skip block 0 because adapter
  toggle fires *between* blocks 0 and 1; block 0 has no shift)
- `mean_logit_shift` — 1 dim
- `max_logit_shift_argmax (norm)` — 1 dim

→ **19-dim** feature vector per (problem, branch).

## Decision rules (same thresholds as T1.B)

| Outcome | Verdict |
|---|---|
| PRM-rerank acc - cmajc-vote acc ≥ +6pp | **WIN** — productizes step-level features into a working reranker; unlocks T3.A real D1 mode router |
| +2pp ≤ Δ < +6pp | **PARTIAL** — logit_shift adds signal vs entropy-only but doesn't close the gap |
| Δ < +2pp | **LOSS-confirmed** — even with the original PRE_REG features, step-level data doesn't carry the signal at this scale |

## Cost

| Item | $ |
|---|---:|
| 48GB on-demand A6000 + bootstrap (~$0.33/hr × ~75 min) | ~$0.42 |
| PRM trainer + heuristic baselines (CPU, ~5 min) | $0 |
| **Total** | **~$0.42** |

## Anti-goals

- No multi-seed at this scale. Single seed=0 is sufficient to verify
  the feature-richness hypothesis. Multi-seed only if WIN.
- No N>100. Same scale as T1.B for direct comparability.
- No new MLP architecture. Same 32→16→1 head; the *features* are the
  variable.
- No raw token features (avoid Track A territory).

## Files

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled
- (Reuses `phase2/spikes/process-reward-verifier/{load_traces,train_prm}.py`
  with feature-list extension for logit_shift_norm dims.)
