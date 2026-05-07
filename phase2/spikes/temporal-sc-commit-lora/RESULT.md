# Phase-4 T3.C — Temporal-SC × commit-LoRA RESULT

**Pre-reg:** `PRE_REG.md` (locked at commit `9d12950`).
**Date:** 2026-05-07. **Cost:** ~$0.95 ($0.30 broken first harvest +
$0.30 re-harvest with `EMIT_PARTIAL_PREDS=1` actually exported +
~$0.35 idle/SSH/buffer).
**Outcome:** **LOSS.** Schedule-conditional temporal-SC voting at
1.5× active / 1.0× inactive **regresses** vs vanilla cmajc-vote on
GSM8K dev_200 idx 0..99: −4pp strict (0.78 vs 0.82), −39pp with
loose-extractor fallback (0.43 vs 0.82). Verdict reproduces R4's
prior-art warning (arXiv 2508.09138) almost exactly: the realistic
estimate of "+3–5pp PARTIAL territory" was already optimistic — the
actual signal is anti-correlated with the 1.5× weight.

---

## Headline numbers

| Method | GSM8K dev_200 idx 0..99 | Δ vs cmajc-vote |
|---|---:|---:|
| cmajc-vote (k=3, 5 branches, BATCHED=0) | **0.8200** | (baseline) |
| Temporal-SC strict (w_active=1.5, w_inactive=1.0) | 0.7800 | **−4.00 pp** |
| Temporal-SC loose-fallback | 0.4300 | **−39.00 pp** |

Re-harvest run wallclock: 2763s (46min) on RTX A6000 48GB at
on-demand pricing. Note that this re-harvest's cmajc-vote (0.82)
itself ran 3pp below the broken-first-harvest cmajc-vote (0.85).
Both fall within the GSM8K cmajc-vote noise band reported in
Phase-2 §3 (mean 0.822 ± 1pp); the difference is irrelevant for
the verdict — temporal-SC underperforms cmajc-vote on the *same*
substrate either way.

## Pre-registered decision rules — outcome

| Rule | Triggered? |
|---|---|
| Δ ≥ +6 pp → WIN | NO |
| Δ ∈ [+2, +6) → PARTIAL | NO |
| Δ < +2 pp → **LOSS** | **YES** (−4pp strict; −39pp loose) |

## Diagnostic: partial-answer coverage per sub-block

| sub-block | n | strict nonempty | loose nonempty | commit-LoRA |
|---:|---:|---:|---:|---|
| 0 | 500 | **0** (0.0%) | 480 (96.0%) | OFF (weight 1.0) |
| 1 | 500 | **3** (0.6%) | 499 (99.8%) | ON (weight 1.5) |
| 2 | 500 | 45 (9.0%) | 499 (99.8%) | ON (weight 1.5) |
| 3 | 500 | **468** (93.6%) | 499 (99.8%) | ON (weight 1.5) |

**Interpretation:** the strict `####` / `Answer:` pattern only
materializes at sub-block 3 in 94% of branches. Sub-blocks 0–1
contribute essentially zero signal under strict matching (0% +
0.6%). The schedule-conditional 1.5× weight thus amplifies a
mostly-null channel for sub-block 1, and only a partially-formed
answer span at sub-block 2 (9% strict-pattern hit). Net effect
is below cmajc-vote because the active-weighted noise crowds
out the sub-block-3 majority signal that cmajc-vote isolates.

The loose-extractor variant catches "any trailing number" and
reaches 96–99.8% non-empty across all sub-blocks, but those early
numbers are **false-positives** — pulled from intermediate
reasoning steps (`"... = 16 - 3 = 13 ..."`) rather than the final
answer. That's why the loose mode collapses to 0.43: it mostly
votes for the first arithmetic intermediate that appeared in
sub-block 0–1, not the final answer.

## Why this confirms R4's prior-art audit

`phase2/research/r4_temporal_aggregation_priorart.md` (commit
`dbc7f0d`) called out two specific risks ahead of the harvest:

> **(a) Partial-answer null hypothesis at sub-blocks 0 and 1.**
> Weight[t=1]=1.5× amplifies noise, not signal. The "active LoRA"
> condition is a fixed schedule side-effect, not a learned signal.
> Pre-reg +6pp claim risks confounding with late-stage dominance
> (sub-block 3 already does the heavy lifting).

The re-harvest data above is exactly this: sub-block 1 strict-
pattern hit rate is 0.6%, and sub-block 3 alone provides ~94% of
the signal. The 1.5× weight on a near-empty channel is what
drove the −4pp regression.

> **(b) K-schedule prior double-counting.** The model learned
> K-step-aware denoising AND the aggregator re-weights by K-step
> state — gain may be already encoded in the LoRA training
> objective.

Even controlling for the partial-answer null hypothesis (i.e.,
restricting the vote to sub-block 3 only — which is what
cmajc-vote already does), there's no headroom for temporal-SC
because the schedule prior is already absorbed by commit-LoRA's
training signal.

The arXiv 2508.09138 ("Time Is a Feature") direct-scoop risk is
moot in light of this LOSS — there was no novel claim to defend
in the first place.

## What this kills, what it leaves open

**Killed:**
- The "schedule-conditional 1.5×/1.0× weight extends temporal-SC"
  hypothesis. The 1.5× weight value was unjustified in PRE_REG,
  and the data confirms it's actively harmful at this schedule.
- Any near-term claim that aggregating across denoising sub-blocks
  beats final-block majority vote on this substrate. The
  unified-negative diagnostic from Phase-2 §2 holds: at sfumato's
  data scale, neither per-branch verification (Track A), nor
  per-problem mode routing (Track C), nor temporal aggregation
  (this spike) closes the cmaj→oracle gap.

**Open:**
- **Principled exponential-decay weights** (per arXiv 2508.09138's
  α=5 best schedule) might recover the small +1–2pp gain those
  authors reported on dLLM math benchmarks. Not pursuing — that's
  re-implementation of published work, not novel.
- **PRM-rerank** (T1.B-redux) already failed at this scale; this
  result cross-validates that step-level aggregation needs a
  fundamentally different signal source than what we can extract
  from raw branch trajectories.
- **Direction A schedule-RLHF** is unaffected — its hypothesis is
  trajectory-level RL on commit-LoRA, not schedule-weighted vote
  aggregation.

## Substrate harvest history (the wart)

The first harvest (`cmajc-t3c-harvest-N100-seed0-partials`,
~$0.30, 53min) ran with `EMIT_PARTIAL_PREDS=1` set in the shell
but the resulting sidecars contained no `partial_answer_*`
fields. Root cause: `_make_trace_dump_callback` evaluates
`getattr(diff_model, "_tokenizer", None)` at callback-creation
time, but `diff_model._tokenizer` is loaded lazily inside the
first `denoise_block` call — so the closure captures `None` and
the partial-decode branch is permanently disabled for that run.

Re-harvest with the same env vars produced sidecars with
partial_answer fields populated *for branches after the first
model-load* (branch_0_idx_0 still has None on this run too;
all subsequent sidecars have the fields).

This is a callback-timing wart in
`e4/runner.py:_make_trace_dump_callback` worth fixing in a
follow-up commit (look up `_tokenizer` at cb-call time, not at
factory time). Not blocking T3.C verdict — the LOSS is robust
across the 499/500 sidecars that did populate.

## Cost ledger

| Item | $ |
|---|---:|
| First harvest (broken sidecars, 53min A6000 on-demand) | ~$0.30 |
| Re-harvest with EMIT_PARTIAL_PREDS actually exported (46min A6000) | ~$0.30 |
| Pod idle / SSH / boot overhead | ~$0.10 |
| Buffer | ~$0.25 |
| **Total** | **~$0.95** |

Original PRE_REG estimated ~$0.45 for one harvest; came in higher
due to the EMIT_PARTIAL_PREDS callback-timing miss requiring a
re-harvest. Still cheap relative to the Phase-4 ~$200 budget.

## Files

- `PRE_REG.md` — pre-reg locked at `9d12950`
- `RESULT.md` — this file (LOSS)
- `aggregate_temporal_sc.py` — aggregator (commit `a0ed464`)
- `e4/results/raw_cmajc_t3c_reharvest_N100_seed0.jsonl` — 100 outcomes
- `e4/results/traces/cmajc-t3c-reharvest-N100-seed0-partials/` — 500 sidecars

## Headline for paper §4 (negative-results)

*"We tested whether schedule-conditional temporal-SC voting
(weighting denoising sub-blocks where commit-LoRA is active 1.5×
versus inactive 1.0×) closes the cmajc→oracle gap on GSM8K
dev_200. Pre-registered WIN threshold (+6pp over cmajc-vote)
was not met; the aggregator regressed by −4pp under strict
final-answer extraction. Diagnostic: the canonical answer span
(`####` / `Answer:`) materializes only at sub-block 3 of 4 in
~94% of branches, so the 1.5× weight on sub-blocks 1–2
amplifies a near-empty channel. We acknowledge concurrent work
on temporal SC for diffusion LMs (Zhang et al. 2025, arXiv
2508.09138) using principled exponential-decay weights; our
hardcoded schedule-conditional weight does not extend their
result. This negative joins our Phase-2 results as a third
unified-negative datapoint: at sfumato's data scale, neither
per-branch verification, nor per-problem mode routing, nor
temporal aggregation closes the cmaj→oracle gap with surface-
feature weighting alone."*
