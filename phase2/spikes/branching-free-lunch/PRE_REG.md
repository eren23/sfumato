# Pre-registration — Branching free-lunch (S0 + S1)

**Phase:** Phase 2/3 spike chain
**Spike ID:** S0 (branch batching) + S1 (ESC quorum=3 early-exit)
**Author:** main session
**Date:** 2026-05-04
**Plan source:** `phase2/proposals/kernel-survey/SUMMARY.md` §3 top pick

## Hypothesis

**S0:** Replacing the sequential `for b in range(n_branches)` loop in
`runner.py:cmaj` (lines 203–232), `cmajc` (233–261), and `cmerge`
(262–288) with a single batched `(B, L+gen)` LLaDA forward call yields
**≥1.3× wall-clock speedup** on `cmaj b=5` at N=20 with **|Δaccuracy|
≤1pp** vs the published baseline `cmaj=79.5% [73.3, 84.9]` on N=200.

**S1:** Adding ESC quorum=3 early-exit on top of S0 — when 3 of 5
branches' partial extracted answers agree at a sub-block boundary, set
the remaining branches' transfer-mask to all-False so they no longer
commit — yields **combined ≥1.5× wall-clock speedup** vs the same
sequential baseline, with **|Δaccuracy| ≤1pp**.

## Conditions tested

- **Baseline:** `CONDITION=cmaj BRANCHES=5 K_STEPS=64 TEMP=0.7 SEED=0
  LORA_PATH=eren23/sfumato-llada-prefix-robust-v3 N_PROBLEMS=20`
  (sequential, current code).
- **S0 treatment:** same, with new batched code path
  (`denoise_block_batched`).
- **S0+S1 treatment:** same, with batched code + ESC `step_callback`
  that votes per sub-block and prunes non-quorum rows.
- **Confirmation:** S0+S1 at N=200 for the headline accuracy CI; cmajc
  at N=200 for the cmajc-v3=82.5% headline.

## Success / kill criteria

| Metric | WIN | LOSS | INCONCLUSIVE |
|---|---|---|---|
| S0 wall-clock speedup at N=20 | ≥1.3× | <1.15× | [1.15×, 1.3×) |
| S0+S1 wall-clock speedup at N=20 | ≥1.5× | <1.3× | [1.3×, 1.5×) |
| `cmaj` N=200 accuracy | ∈ [78.0%, 81.0%] | outside [77.5%, 81.5%] | — |
| `cmajc-v3` N=200 accuracy | ∈ [80.0%, 85.0%] | outside [80.0%, 85.0%] | — |
| ESC false-prune rate | ≤2pp accuracy degradation | ≥4pp | — |

INCONCLUSIVE on speedup → defer S12 (Hydragen, more invasive) to Phase-3.
LOSS on accuracy → revert; investigate seed remapping / RNG semantics.

## Measurement plan

1. **Pre-spike snapshot:** `git rev-parse HEAD` recorded; existing
   `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl` is the baseline
   accuracy (79.5% [73.3, 84.9]) with no per-problem wall-clock column —
   so we run a paired baseline at N=20 explicitly to get the wall-clock
   distribution.
2. **Wall-clock instrumentation:** add `wallclock_ms` column to the
   per-problem row dict in `runner.py:run_condition` (before the JSONL
   write at the end of the loop).
3. **Paired runs:** identical (problems, seeds) under (a) sequential
   baseline (b) S0 batched (c) S0+S1 batched-with-ESC. All three on the
   same RTX 4090 spot pod, same session, no other workloads.
4. **Statistical test:** paired Wilcoxon on per-problem wall-clock
   between (a) and (b), and between (a) and (c). Report median speedup
   + 95% CI.
5. **Accuracy:** binom-CI (`scripts/binom_ci.py`) on majority-correctness
   for each treatment at N=200. Report compared to the published
   baseline CI.
6. **ESC trigger logging:** under S0+S1, log `esc_trigger_subblock` and
   `branches_pruned` per problem to enable false-prune-rate analysis.

## Files modified

**Source:**
- `e4/diff_llada.py` — new methods `denoise_block_batched` +
  internal `_generate_batched`. Original `denoise_block` / `_generate`
  unchanged → backcompat fixture stays bit-identical.
- `e4/runner.py` — `cmaj`, `cmajc`, `cmerge` switch to
  `denoise_block_batched`. Other conditions untouched.
- `e4/runner.py:run_condition` log row — add `wallclock_ms`.

**Tests / fixtures:**
- `phase2/inference_viz/test_backcompat.py` — must still pass (locks
  default-callback bit-identicality on the *non-batched* path).

## Compute envelope

- N=20 paired baseline + S0 + S0+S1: ~3 × 12 min × 5 branches ≈ 3 hr
  on RTX 4090 → **~$0.60** at $0.20/hr spot.
- N=200 cmaj (S0+S1): ~30 min batched ≈ **~$0.10**.
- N=200 cmajc-v3 (S0+S1): ~30 min batched ≈ **~$0.10**.
- Total: **≤$5** including spin-up overhead.

## Rollback / safety

- New method is opt-in at the runner level. Reverting cmaj/cmajc/cmerge
  to call `denoise_block` (legacy) is a one-line revert.
- If accuracy fails CI: revert, file `FAIL.md` documenting the
  divergence (likely RNG semantics or per-row mask leakage).

## Falsification artifacts

- `phase2/spikes/branching-free-lunch/wallclock_n20_baseline.jsonl`
- `phase2/spikes/branching-free-lunch/wallclock_n20_s0.jsonl`
- `phase2/spikes/branching-free-lunch/wallclock_n20_s0_s1.jsonl`
- `phase2/spikes/branching-free-lunch/RESULT.md`
- `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200_S0S1.jsonl`
- `e4/results/raw_cmajc_k64_seed0_b5_v3LoRA_N200_S0S1.jsonl`

## Notes

- ESC trigger is computed on the **partial** decoded text at each
  sub-block boundary. Re-uses `e4/grade.py:extract_answer`. When the
  partial doesn't yet contain a matchable answer, that branch
  contributes no vote and ESC waits.
- ABL_B closure (`phase2/spikes/abl_b_RESULT.md`, commit `27da1c9`):
  commit-LoRA-v3 reinforces format markers but does not flip extracted
  digits — supports the low false-prune-rate prior for cmajc.
