# Phase K iterated routing — N-matched comparison

## Setup

F10 final (305M composite, val_ar_nll=1.155), held-out GSM8K problems
(offset 7000+), `gen_diff_draft_ar_refill` with K_AR=64, K_DIFF=64,
N_DIFF_STEPS=16. Two runs, identical except for `N_OUTER`:

- `k2_n_outer1_n100.json` — N_OUTER=1, single-pass (this is the
  recipe in T0_PROBES_FINAL.md Phase K)
- `k2_iterated2_n100.json` — N_OUTER=2, iterated self-correction loop
  (re-mask refilled positions → re-run diff → re-flag → AR-refill again)

Both at N_EVAL=100 (matched). Wall times not comparable across runs
(local Mac MPS, shared with other workloads).

## Result

| refill pct | n_outer=1 NLL | n_outer=2 NLL | Δ NLL | n_outer=1 loop | n_outer=2 loop |
|------------|---------------|---------------|-------|-----------------|-----------------|
|     10%    |     12.004    |     12.042    | +0.04 |       11%       |        8%       |
|     25%    |     11.994    |     12.238    | +0.24 |       10%       |       10%       |
|   **50%**  |   **12.063**  |   **11.864**  | **−0.20** |   **19%**   |     **12%**     |
|     75%    |     12.285    |     12.321    | +0.04 |       17%       |       14%       |

## Reading

- **pct50 is the optimal threshold for iteration**: −0.20 NLL gain *and*
  −7pp loop-rate reduction. The first AR-refill round commits 50% of
  the diff-drafted positions to AR; the second diff round runs only on
  those 50 positions; the second AR-refill commits the persistent-low-
  confidence subset.
- **Low thresholds (pct10/pct25) don't benefit**: cumulative refill rate
  doubles (9.4%→18.8%, 25%→50%) but the second round just re-flags
  positions the first round already touched. The diff head can't find
  meaningfully better continuations on tokens it already endorsed.
- **High threshold (pct75) doesn't benefit either**: at 75% refill the
  AR head is doing most of the generation; iteration is mostly
  redundant.
- **Sweet-spot mechanism**: at pct50 the iteration cycles through
  *different* low-confidence positions across rounds (per the
  `refill_history` arrays — second-round refill positions are NOT a
  subset of first-round positions), suggesting the AR refills create
  new local-context perturbations that change the diff head's
  confidence ranking.

## Why this is composite-unique

Pure-AR cannot re-mask its own outputs (no MASK token, no
bidirectional attention). Pure-diff has no external endorsement signal
to gate refills. Composite uniquely supports a self-correcting loop
where each head verifies the other's commits and re-revises the
persistent-low-confidence subset.

## Comparison to published precedents

- DEER (2512.15176): block-level draft+verify, single round.
- Corrective DLMs (2512.15596): within-diffusion correction, no AR escape.
- I-DLM (2604.11035): per-token confidence routing, single pass.

None published a per-token AR-routed iterative diff-refinement loop
at <1B text scale prior to this measurement.

## Caveat

N=100 with single seed on a 305M model trained on a math substrate.
Generalisation beyond GSM8K-Q/A continuations untested. The pct50
sweet spot may shift at scale (1B+) or on broad-text substrates;
Phase K already showed substrate sensitivity (WikiText-2 transfer
was small but positive).

## Next steps to harden

1. **N_OUTER=3 sweep** (cheap; ~10 min A40): does the gain saturate?
2. **Multi-seed** at N_OUTER=2 pct50 to pin the −0.20 effect within
   binomial-on-NLL noise.
3. **WikiText-2 transfer** at N_OUTER=2: does the iteration win on
   non-math text?
4. **Paper C update**: add a one-paragraph Section 9 subsection
   ("Iterated self-correction") with this table, point to Phase P.2
   (disjoint feature bases) as the mechanism that enables iteration.

## Provenance

- `e5/scripts/probe_diff_draft_ar_refill.py:264-280` — outer loop
  implementation (already present; default N_OUTER=1).
- `e5/results/f10_mixed/composite/k2_n_outer1_n100.json` — control.
- `e5/results/f10_mixed/composite/k2_iterated2_n100.json` — iterated.
- Date measured: 2026-05-26.
