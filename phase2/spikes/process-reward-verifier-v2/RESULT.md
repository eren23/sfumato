# T1.B-redux Process-Reward Verifier v2 — RESULT

**Pre-reg:** `PRE_REG.md` (committed `d532dc1`).
**Run date:** 2026-05-06 | **Cost:** ~$0.42 (48GB A6000 on-demand,
64min run + 5min boot).
**Outcome:** **LOSS-CONFIRMED** — even with `LOGIT_SHIFT_NORM=1`
shadow-forward features added on top of the T1.B v1 entropy +
commit_lora flag features, no PRM reranker beats cmajc majority vote.

---

## Headline numbers

| Metric | Value | Pre-reg target | Verdict |
|---|---:|---:|---|
| cmajc-vote acc (N=100, seed=0, this run) | 0.8600 | n/a (baseline) | matches Phase-2 noise band |
| PRM-MLP rerank acc (5-fold CV by problem) | 0.7300 | ≥ 0.92 (cmajc + 6pp) | **LOSS** (Δ = −13pp!) |
| oracle (any-branch-correct) acc | 0.9200 | n/a (upper bound) | gap = 6pp |
| Δ(PRM − cmajc) | **−13.00pp** | ≥ +6pp | **LOSS** |
| gap closure | **−217%** | ≥ +75% | **LOSS** |

Note: cmajc-vote acc here (0.86) is higher than T1.B v1 (0.80) on the
same N=100. `LOGIT_SHIFT_NORM=1` enables a shadow-forward path that
adds determinism via toggling-and-untoggling the merged adapter; this
appears to slightly stabilize the cmajc trajectory. Either way, the
signal we wanted to test (logit_shift_norm as a PRM feature) failed.

W&B: [l4ogwdrs](https://wandb.ai/eren23/sfumato-e4/runs/l4ogwdrs).

## Sanity-check baselines (v2 19-d feature space)

To confirm the LOSS isn't an MLP-instability artifact:

| Reranker | Acc |
|---|---:|
| **cmajc majority-vote** (no rerank) | **0.8600** |
| MLP 19→32→16→1 (5-fold CV) | 0.7300 |
| Logistic regression (all 19 dims, 5-fold CV) | 0.7500 |
| Logistic regression (logit_shift-only 5 dims, 5-fold CV) | 0.7200 |
| argmin mean logit_shift_norm (heuristic) | 0.7900 |
| argmax mean logit_shift_norm | 0.6300 |
| argmin block-3 logit_shift_norm | 0.7600 |
| argmax block-3 logit_shift_norm | 0.6800 |
| argmin total logit_shift_norm | 0.7900 |

**Every** strategy underperforms cmajc-vote. The argmin/argmax
asymmetry (0.79 vs 0.63) suggests **larger logit_shift weakly
correlates with WRONG answers** (i.e., adapter changing the
trajectory more = bad), but the effect is small (≤7pp on its own)
and dominated by the vote.

## Pre-registered decision rules — outcome

| Rule | Triggered? |
|---|---|
| Δ ≥ +6pp → WIN | NO |
| +2pp ≤ Δ < +6pp → PARTIAL | NO |
| Δ < +2pp → **LOSS-CONFIRMED** | **YES** (Δ = −13pp) |

## What this means

**Both branches of T1.B's PRE_REG hit LOSS:**
- T1.B v1 (no logit_shift, 14-d features): Δ = −4pp → LOSS
- T1.B-redux (with logit_shift, 19-d features): Δ = **−13pp → LOSS**

Two independent confirmations of the same finding strengthen §2's
unified-negative diagnostic: **at sfumato's data scale, surface
features extractable from the LLaDA denoising trajectory cannot close
oracle gaps for per-branch verification, regardless of whether the
features include adapter-perturbation magnitude.**

This is a strong negative for the paper. The K2 inverted-U finding
(positive, Phase-2 §3, replicated cross-domain in T2.A) becomes more
mechanistically interesting precisely *because* the same step-level
features that produced the K2 schedule signal cannot, post-hoc,
reverse-engineer correctness.

## Implications

### What this kills (additionally to T1.B v1)

- The "logit_shift_norm captures adapter-trajectory-divergence signal"
  hypothesis. The shadow-forward L2 norm DOES exist (range 0 to 4512),
  varies meaningfully across branches, but DOESN'T predict correctness.
- The original PRE_REG's amendment-to-redux-trajectory (drop
  logit_shift_norm at v1, add it at v2). Both choices land at LOSS.

### What this leaves open

- **Token-level features.** If raw committed tokens were fed into a
  small encoder, would that work? That's Track A territory and
  Phase-2 already showed even Qwen-7B chat can't crack it at N=200.
  Probably not worth chasing.
- **Cross-step temporal patterns.** Current features aggregate per
  sub-block. A sequence model over the full denoising trajectory
  (not just sub-block summaries) might capture timing of the
  trajectory's "decisive" moment. Speculative; not pre-registered.
- **T3.A real D1 sub-block-level mode router** remains gated. Plan
  said "fire T3.A if T1.B WINs." T1.B-redux LOST. T3.A's premise
  ("step-level features unlock the router") needs a feature
  paradigm shift before it fires.

## Cost ledger

| Item | $ |
|---|---:|
| 48GB A6000 on-demand × ~70 min × $0.33/hr | ~$0.39 |
| Bootstrap + idle | ~$0.03 |
| MLP/heuristic eval (CPU) | $0 |
| **Total T1.B-redux spend** | **~$0.42** |

Total Phase-3 GPU spend so far: $0.10 (T1.C) + $0.23 (T1.B v1) + $0.60 (T2.A)
+ $0.42 (T1.B-redux) = **~$1.35**. Of the user's $20 increased budget,
~$1.35 consumed; ~$18.65 remains.

## Files

- `PRE_REG.md` — pre-registration (locked before harvest, commit `d532dc1`)
- `load_traces_v2.py` — 19-d feature extractor
- `train_prm_v2.py` — 5-fold CV trainer (reuses v1 MLP)
- `RESULT.md` — this file (LOSS-CONFIRMED)
- `e4/results/traces/cmajc-prm-v2-N100-seed0-shifted/` — 500 sidecar JSONLs
- `e4/results/raw_cmajc_k64_seed0_prm_v2.jsonl` — 100 outcome rows
- W&B: https://wandb.ai/eren23/sfumato-e4/runs/l4ogwdrs

## Headline for paper

The finding to add to §2 unified-negative diagnostic:

> *Two pre-registered passes at training a process-reward verifier on
> the LLaDA branch trajectory features both fail. v1 with entropy +
> adapter-active features alone underperforms cmajc by 4pp; v2 adding
> the shadow-forward logit_shift_norm — which we initially expected to
> help — underperforms by 13pp. The LLaDA denoising trajectory does
> not expose, at the granularity sfumato samples, surface features
> that distinguish correct from incorrect branches.*
