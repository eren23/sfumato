# Pre-registration — Temporal Self-Consistency × Commit-LoRA (T3.C)

**Date:** 2026-05-06 | **Spike:** Phase-3 T3.C — vote across denoising
*steps* (not just final answers) using schedule-aware commit-LoRA flag.
**Status:** SCAFFOLDED, not yet running. Eng prerequisites (~5-7 days).

## Hypothesis

Standard self-consistency (cmaj) votes across 5 *final* branch outputs.
Temporal self-consistency (Lin et al. 2025) extends this to vote across
intermediate denoising steps. sfumato's commit-LoRA toggle adds a new
axis: at each sub-block boundary the adapter is either active (k=2,3)
or inactive (k=0,1,4-only). A **schedule-weighted temporal vote** —
weighting predictions from sub-blocks where commit-LoRA is active
HIGHER than from sub-blocks where it's inactive — should provide
information that ordinary cmaj branch voting misses.

Concretely: at each sub-block boundary t and branch b, decode the
partial committed prefix, extract the most recent `Answer: N` token
if present. Build a (T sub-blocks × B branches) matrix of partial
answer guesses. Apply temporal SC voting:

  weight[t, b] = w_commit(t) × w_branch_diversity(b)
  final_answer = argmax_a sum_{t, b} weight[t, b] · 1[partial[t, b] = a]

where `w_commit(t)` is HIGHER for t∈{2, 3} (commit-LoRA active in K=3
schedule) and lower for t∈{0, 1, 4}.

If this **closes the cmajc voting-rule gap** (which T1.B failed to
close at any feature richness), it becomes the first sfumato result
where a schedule-aware aggregator beats vanilla cmaj.

## Substrate

- GSM8K-test idx 0..199, cmajc k=3 substrate already harvested
  (T1.B-redux trace dump at `e4/results/traces/cmajc-prm-v2-N100-seed0-shifted/`).
- Each sidecar JSONL has 4 sub-block records per branch with per-block
  `commit_lora_active` flag.
- Need: per-sub-block partial decode of committed prefix → extracted
  partial answer string. NOT YET in the trace dump (would need a runner
  change to record partial decode at each sub-block boundary).

## Eng prerequisites

1. **Extend `_make_trace_dump_callback`** in `e4/runner.py` to also
   record `partial_pred[t]` (extracted answer from the committed
   prefix at sub-block t). ~50 LOC change.
2. **Re-harvest** with `TRACE_STEPS=1 EMIT_PARTIAL_PREDS=1` on a 48GB
   pod. ~$0.42, ~70min — same as T1.B-redux.
3. **Implement temporal SC aggregator** (CPU script, ~100 LOC).
   Tunable weights per sub-block t.
4. **5-fold CV by problem** across N=100. Compare against:
   - cmajc majority vote (winner)
   - Phase-2 K2 best (cmajc k=3 mean = 0.822)
   - Random / argmax-PRM (T1.B baselines for sanity)

## Decision rules

| Outcome | Verdict |
|---|---|
| Temporal-SC acc ≥ cmajc-vote + 6pp on N=100 | **WIN** — schedule-aware aggregation closes the voting-rule gap; first sfumato result where a non-trivial aggregator beats vanilla cmaj |
| Temporal-SC acc ∈ [cmajc, cmajc + 6pp) | **PARTIAL** — schedule weighting helps but not enough; refine weighting scheme |
| Temporal-SC acc < cmajc-vote | **LOSS** — even with the schedule signal, voting across sub-blocks doesn't add information; partial answers too noisy |

## Cost

| Item | $ |
|---|---:|
| Eng (5-7 days for runner change + harvest + aggregator) | $0 |
| 48GB on-demand pod for re-harvest | ~$0.42 |
| Eval (CPU) | $0 |
| **Total** | **~$0.42** |

## Anti-goals

- No tuning the schedule weights to fit the train fold. Fixed schedule
  weights based on K2 ablation a priori (e.g., `w_commit(t)` = 1.5 for
  t∈{2,3}, 1.0 for t∈{0,1,4}).
- No mixing with PRM-MLP from T1.B (different paradigm). Pure
  vote-counting.
- No N>100 in initial spike. Scale only on WIN.

## Files (when this fires)

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled
- `e4/runner.py` — partial-pred trace extension
- `phase2/spikes/temporal-sc-commit-lora/aggregate_temporal_sc.py` — voter

## Status notes

**Why not auto-dispatched in this session:** runner trace-dump extension
is a code change requiring testing locally. Locking PRE_REG now so a
future session has fixed thresholds.

**Trigger condition:** ready to fire as soon as the runner extension
ships and a 48GB pod is provisioned.
