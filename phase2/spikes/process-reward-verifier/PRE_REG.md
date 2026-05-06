# Pre-registration — Process-Reward Verifier on Branch Trajectories

**Date:** 2026-05-06 | **Spike:** Phase-3 T1.B of
`/Users/eren/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`.

## Hypothesis

Per-step features harvested from the LLaDA denoising trajectory of
each cmajc branch (entropy mean/max per sub-block, logit_shift_norm
under the prefix-robust + commit-LoRA stack, commit_lora_active flag,
mechanism label) carry enough signal to train a small MLP reranker
(32→16→1) that closes ≥75% of the cmajc voting-rule gap (~9pp on
N=200 → ≥6.75pp lift) when used to pick among 5 sequential branches
per problem.

## Substrate

- **Source:** sfumato runner cmajc branch trajectories with new
  `TRACE_STEPS=1 LOGIT_SHIFT_NORM=1` plumbing landed in commit
  `04f01db` (T1.B.1).
- **Harvest:** cmajc N=100 problems × 5 branches = 500 trajectories.
  Pod: 48GB A40 spot (24GB OOMs with shadow forward enabled for
  `LOGIT_SHIFT_NORM=1`).
- **Env:** `BATCHED=0 BRANCHES=5 K_STEPS=64 TEMP=0.7 SEED=0
  COMMIT_N_BLOCKS=3 TRACE_STEPS=1 LOGIT_SHIFT_NORM=1`. Sequential
  branches required because trace dump is wired only into
  `denoise_block` (single-row); batched path uses BatchStepState
  which doesn't expose entropy / logit_shift_norm yet.
- **Frozen indices:** `e4/data/gsm8k_dev_200.json` idx 0..99.
- **Output:** `e4/results/traces/cmajc-prm-N100-seed0/branch_*_idx_*.jsonl`
  (500 sidecar files, one row per sub-block) + the main outcome
  JSONL `e4/results/raw_cmajc_k64_seed0.jsonl` with per-branch
  correct flags.

## Features (per branch trajectory)

For each sub-block s in {0,1,2,3} of branch b on problem i:

- `entropy_mean[s]` — mean of `state.entropy` (top-k entropy over
  the sub-block's committed positions)
- `entropy_max[s]` — max of `state.entropy`
- `logit_shift_norm[s]` — L2 norm of (logits_with_adapter -
  logits_without_adapter) at the toggle boundary (NaN before the
  first toggle)
- `commit_lora_active[s]` — bool, set by the runner toggle
- `n_committed[s]` — number of tokens committed in this sub-block

Aggregated to a 16-dim feature vector per (b, i):
`[entropy_mean[0..3], entropy_max[0..3], logit_shift_norm[1..3],
commit_lora_active_fraction, mean_entropy, std_entropy,
mean_logit_shift]`.

## Method

1. **Train PRM head:** MLP 16→32→16→1 with sigmoid output. Target:
   per-branch `correct ∈ {0, 1}` from outcome JSONL.
2. **Cross-validation:** 5-fold split *by problem index* (so the
   reranker never sees branches from a problem in its training
   fold). 100 problems → 5 folds × 20 problems × 5 branches = 100
   held-out trajectories per fold.
3. **Rerank eval:** for each held-out problem, score all 5 branches
   with the held-out PRM head, pick `argmax-PRM` branch's extracted
   answer. Compute accuracy across 100 problems.
4. **Baseline:** plain cmaj majority-vote on the same 5 branches
   (no commit-LoRA, no PRM); also cmajc majority-vote (commit-LoRA
   on, no PRM). Voting-rule gap = oracle (any-branch-correct) -
   cmajc.

## Decision rules

| Outcome | Verdict |
|---|---|
| PRM-rerank acc - cmajc-vote acc ≥ +6pp on N=100 | **WIN** — closes ≥75% of expected ~9pp gap; productizes step-level features as PRM substrate; unlocks T3.A real D1 mode router |
| PRM-rerank acc - cmajc-vote acc ∈ [+2pp, +6pp) | **PARTIAL** — step-level features carry signal but not at gap-closing scale; paper §3.5 grows a "step-level features lift, don't close" subsection |
| PRM-rerank acc - cmajc-vote acc < +2pp | **LOSS** — strengthens §2 unified-negative diagnostic ("even step-level features fail at this scale"); deeper paradigm shift needed |

## Anti-goals

- No multi-seed at this scale. Single seed=0 N=100 is sufficient
  to verify per-step features carry signal. Multi-seed lift only
  if N=100 hits PARTIAL or WIN.
- No N=200 substrate harvest in this spike. N=100 is the
  pre-registered training scale; doubling without a proven signal
  is wasted GPU.
- No deep MLP. 16→32→16→1 is the cap. The hypothesis is about
  feature signal, not capacity.
- No raw-token features. We're testing whether sub-block-level
  internals (entropy + logit_shift + adapter flag) suffice. Adding
  token-string features would muddy the diagnostic.
- No retraining commit-LoRA or prefix-robust. Frozen v3 adapters,
  same as Phase-2 cmajc baseline.

## Cost

- Substrate harvest: ~25 min on 48GB A40 spot @ $0.24/hr ≈ $0.10.
  With pod startup + bootstrap: ~$0.15 total.
- PRM trainer: CPU-only, ~5 min, $0.
- Eval + writeup: $0.
- **Total spike cost: ~$0.15.**

## Files

- `PRE_REG.md` — this file
- `load_traces.py` — joins sidecar JSONLs with main outcome JSONL,
  produces `(features_16d, correct)` tuples per (problem, branch)
- `train_prm.py` — 5-fold CV trainer + per-fold rerank evaluation
- `rerank_eval.py` — single-fold sanity check + final number
- `RESULT.md` — verdict, to be filled after harvest + train

## Implementation refs

- `e4/runner.py:_make_trace_dump_callback` — sidecar writer (commit
  04f01db)
- `e4/diff_llada.py:StepState` — feature source dataclass
- `e4/diff_llada.py:_maybe_logit_shift_norm` — shadow forward gate
