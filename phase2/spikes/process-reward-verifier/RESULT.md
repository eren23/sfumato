# T1.B Process-Reward Verifier — RESULT

**Pre-reg:** `PRE_REG.md` (committed `8830dda`, amended `9408b1a` before
harvest).
**Run date:** 2026-05-06 | **Cost:** ~$0.20 (24GB 4090 spot, ~55min run + ~5min boot).
**Outcome:** **LOSS** — PRM-rerank (0.76) loses to cmajc-vote (0.80) by 4pp.
Sub-block-level entropy + commit-LoRA-active features carry no usable
reranking signal at N=100.

---

## Headline numbers

| Metric | Value | Pre-reg target | Verdict |
|---|---:|---:|---|
| cmajc-vote acc (N=100, seed=0) | 0.8000 | n/a (baseline) | matches Phase-2 0.83±0.85pp |
| PRM-rerank acc (5-fold CV by problem) | 0.7600 | ≥ 0.86 (cmajc + 6pp) | **LOSS** |
| oracle (any-branch-correct) acc | 0.8900 | n/a (upper bound) | 9pp gap, exactly as planned |
| Δ(PRM − cmajc) | **−4.00pp** | ≥ +6.00pp | **LOSS** |
| gap closure | **−44%** | ≥ +75% | **LOSS** |

Source: `e4/results/traces/cmajc-prm-N100-seed0-trace/` (500 sidecar
JSONLs, 4 sub-blocks each) + `e4/results/raw_cmajc_k64_seed0_prm.jsonl`
(100 outcome rows).
Wandb: [vbtzsd6b](https://wandb.ai/eren23/sfumato-e4/runs/vbtzsd6b).

## Pre-registered decision rules — outcome

| Rule | Triggered? |
|---|---|
| Δ ≥ +6pp (≥75% closure) → WIN | **NO** |
| +2pp ≤ Δ < +6pp → PARTIAL | **NO** |
| Δ < +2pp → LOSS | **YES** (Δ = −4pp) |

## Sanity-check baselines

To confirm the LOSS is structural rather than an MLP-instability artifact,
ran every reasonable rerank strategy:

| Reranker | Acc |
|---|---:|
| **cmajc majority-vote** | **0.8000** |
| MLP 14→32→16→1 (5-fold CV) | 0.7600 |
| Logistic regression (5-fold CV) | 0.7500 |
| argmin block-3 entropy_mean (heuristic: most-committed final block) | 0.7400 |
| argmin overall mean_entropy | 0.7200 |
| branch_0 always | 0.7200 |
| random branch | 0.7200 |
| argmax block-3 entropy_mean | 0.6700 |
| argmax overall mean_entropy | 0.6000 |

**Every** rerank strategy underperforms the vote. The signal isn't there
to be found — this is not a model-capacity or training-instability issue.

## Root cause

At N=100 problems × 5 branches × 4 sub-blocks (= 500 trace records, 100
training labels), the feature stream we exposed:

- **entropy_mean / entropy_max per sub-block** (8 dims) — measures how
  spread the top-k predicted token distribution is at commit time
- **commit_lora_active per sub-block** (1 dim, fraction) — the K2
  inverted-U toggle from §3.3
- **derived stats** (5 dims) — block-3 entropy, block-3/block-0 ratio,
  argmax-subblock, mean/std

These features capture surface generation dynamics. They DO NOT encode
**semantic correctness signal** at the granularity needed to distinguish
"branch 2 will end up at 18 (correct)" from "branch 4 will end up at 16
(wrong)" before the answer span actually lands. The LLaDA semi-AR
parallel-decode confidence threshold already commits high-confidence
tokens regardless of which "trajectory" the branch is on; entropy *post-*
commit reflects token-level uncertainty, not chain-of-thought-level
correctness.

This is the same pathology Tracks A (verifier-encoder-scale) and C
(D1-bandit-on-replay) hit:

> Surface features cannot close oracle gaps for either per-branch
> verification (Track A), per-problem mode-routing (Track C), or — now —
> per-step process-reward verification (Track T1.B), at the data
> scale and with the feature set sfumato can extract from its existing
> branch-trace substrate.

## What this kills, what it leaves open

**Killed:**
- The "step-level features carry signal at N=100 cmajc" hypothesis.
- The lightweight Tier-1 productization path for §3 (commit-LoRA K2)
  + per-step PRM into a single deployable artifact.
- T3.A's "step-level features unlock real D1 mode router" gating
  premise. The Phase-3 plan said:
  > *Gate: only fire if T1.B (process-reward verifier) shows step-level
  > features are useful. If T1.B WINs, this is the natural next phase.*
  >
  > T1.B did NOT win. T3.A remains gated — needs richer features
  > (logit_shift_norm, sub-block-level activation diffs, raw token
  > sequences) before any further bandit-on-replay attempt.

**Open paths for Phase-3+ revision:**

1. **Add `LOGIT_SHIFT_NORM=1` features** on a 48GB pod. The original
   PRE_REG (commit 8830dda) called for these; the amended version
   (9408b1a) dropped them when the provisioner returned 24GB. Adding
   the shadow-forward L2-norm at the toggle boundary might surface a
   "did the adapter actually change the trajectory" signal that
   entropy alone misses. Cost: ~$0.40 (48GB A40 spot, ~70min for
   sequential N=100). Worth trying *only if* the §3 paper revision
   actively cites a follow-up direction; otherwise it's chasing a
   dead substrate.

2. **Token-level features** — bypass per-sub-block aggregation and
   feed the raw committed token sequence into a small encoder.
   This re-creates Track A's verifier-scaling substrate and runs into
   the same monotone-narrowing-but-never-crossing pathology. Probably
   not worth a fresh spike.

3. **Multi-seed substrate at N=100×3** — current N=100 single-seed
   is at the noise floor for a +6pp signal. With per-branch correct
   rate ≈ 0.70 and oracle 0.89, the standard error on a 100-problem
   estimate is ±5pp — the −4pp loss is borderline within noise.
   A clean LOSS verdict would normally need triple-seed, but given
   that EVERY heuristic and every classifier we tried lost, the
   noise-floor caveat doesn't rescue the hypothesis.

4. **Different paradigm entirely**: train a small LLM on
   ⟨branch_text, correct⟩ pairs rather than per-step features.
   That's Track A territory and Phase-2 already showed even Qwen-7B
   chat can't crack it at N=200. Defer.

## What stays in main

- `e4/runner.py:_make_trace_dump_callback` — the per-step trace dump
  plumbing (commit `04f01db`) is generally useful for any future
  step-level analysis, even though the PRM application failed. Keep.
- `phase2/spikes/process-reward-verifier/{load_traces,train_prm}.py` —
  the loader + 5-fold CV harness can be reused for any future
  step-level rerank attempt. Keep.
- This `RESULT.md` — feeds the §2 unified-negative diagnostic in the
  paper. Add the Track T1.B row to the "surface features cannot
  close oracle gaps" table.

## Cost ledger

| Item | $ |
|---|---:|
| Crucible pod boot + bootstrap | ~$0.05 |
| cmajc N=100 seed=0 BATCHED=0 BRANCHES=5 TRACE_STEPS=1 (~55 min, 24GB 4090 spot @ $0.20/hr) | ~$0.18 |
| MLP + heuristic baselines (CPU) | $0 |
| **Total T1.B spend** | **~$0.23** |

Of the <$1 Tier-1 GPU budget, ~$0.10 (T1.C) + ~$0.23 (T1.B) = $0.33
consumed; ~$0.67 remains for follow-up if the paper revision asks for it.

## Files

- `PRE_REG.md` — pre-reg + thresholds (amended for 24GB pod)
- `load_traces.py` — sidecar JSONL → 14-d feature loader (joins outcome
  by problem-idx, builds per-(problem, branch) features)
- `train_prm.py` — 5-fold CV MLP trainer + reranker + verdict logic
- `RESULT.md` — this file (LOSS)
- `e4/results/traces/cmajc-prm-N100-seed0-trace/` — 500 sidecar JSONLs
- `e4/results/raw_cmajc_k64_seed0_prm.jsonl` — 100 outcome rows
- wandb: https://wandb.ai/eren23/sfumato-e4/runs/vbtzsd6b
