# Pre-registration — D1 Sub-Block Mode Router v2 (T3.A)

**Date:** 2026-05-06 | **Spike:** Phase-3 T3.A — original E1 contribution
revival. **Status:** SCAFFOLDED, not yet running. Multi-week eng
prerequisite. Gated on T1.B-redux outcome.

## Background

The original E1 sfumato thesis was a **learned dynamic mode router**
that picks AR vs diffusion vs hybrid per-problem. Phase-2 Track C
falsified this at the per-problem-bandit-on-replay level using surface
features (LR 0.65 / RF 0.60 vs best-fixed 0.75 — D1-mode-router LOSS).

T3.A revives the question at finer granularity: a **sub-block-level
router** that, at every commit boundary, decides whether to extend the
current mechanism (AR / diffusion / branch / commit-LoRA toggle) or
switch. Trained with REINFORCE + counterfactual branch-agreement
reward at sub-block boundaries.

**Gating condition:** T1.B closed LOSS at N=100 with entropy +
commit_lora_active features. T3.A's premise was "if T1.B WINs, step-
level features unlock the router." Plan said T3.A is gated on T1.B WIN.

**Updated gating after T1.B-redux:** if T1.B-redux WINs (logit_shift_norm
features close the cmajc voting-rule gap), T3.A is unblocked. If
T1.B-redux LOSES, T3.A remains gated on a future feature-engineering
spike or paradigm shift.

## Hypothesis

A small policy network (32→32→32→3 mode logits, ReLU) trained with
REINFORCE on per-sub-block features (entropy, logit_shift_norm,
commit_lora_active flag, sub-block index, top-k mass) can pick a
per-sub-block mechanism that beats best-fixed cmajc by ≥4pp on N=200.

Reward signal: sub-block-level branch-agreement (when ≥3 of 5 branches
agree on the partial answer span at this sub-block, +1; else 0).
Discounted to terminal correctness with γ=0.9.

## Substrate

Same per-step trace substrate as T1.B (and T1.B-redux if it produces
better features). 5-fold CV by problem.

## Pre-reg WIN/LOSS thresholds

| Outcome | Verdict |
|---|---|
| Router-driven cmajc N=200 ≥ best-fixed cmajc + 4pp | **WIN** — original E1 thesis revived; sfumato becomes "the dynamic mode router that actually works" |
| Router-driven cmajc ∈ [best-fixed, +4pp) | **PARTIAL** — router beats fixed but doesn't justify the complexity |
| Router-driven cmajc < best-fixed cmajc | **LOSS** — feature richness still insufficient at sfumato substrate scale; same pathology as Track A / Track C / T1.B v1 |

## Eng prerequisites

1. **T1.B-redux must complete** (this PRE_REG is informational until then).
2. If T1.B-redux WINs, build the policy network: REINFORCE harness
   reading sidecar JSONLs from T1.B-redux, action space {continue,
   switch_to_AR, toggle_commit_LoRA}, reward = branch-agreement.
3. Train policy (CPU, ~1 day eng) on N=100 substrate from T1.B-redux,
   eval on held-out N=100.
4. Scale to N=200 if WIN at N=100.

## Cost (when this fires)

| Item | $ |
|---|---:|
| Eng days (~10-15 days) | $0 |
| Optional richer-feature substrate harvest at N=200 (if T1.B-redux WIN was on N=100) | ~$0.80 |
| Training (CPU) | $0 |
| Scale-up N=200 K-sweep eval | ~$0.50 |
| **Total** | **~$1.30 GPU + ~12 eng days** |

## Files (when this fires)

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled
- `phase2/proposals/adaptive-mode-router.md` — original proposal
- `train_router.py` — REINFORCE harness (new)
- `eval_router.py` — held-out CV evaluator (new)

## Status notes

**Why not auto-dispatched in this session:** premise depends on T1.B-redux
outcome. If LOSS, this stays gated. Pre-registering now so the
WIN/LOSS thresholds can't shift later.

**Trigger condition:** T1.B-redux verdict ∈ {WIN, PARTIAL} →
auto-trigger in next session.
