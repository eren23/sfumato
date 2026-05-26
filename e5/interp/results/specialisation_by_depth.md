# Phase P.2 expanded — mode specialisation by depth

**Date:** 2026-05-27
**Inputs:** 10 AR-mode SAEs (block_6..15_ar) + 10 diff-mode SAEs
(block_6..15_diff) + ln_f_ar + ln_f_diff. All TopK k=64, d_features=16384,
trained on F10 (305M, val_ar_nll=1.155) FineWeb-Edu activations.

## Headline table

| layer | mean cos(AR→diff) | median | max | % features with diff sibling ≥ 0.7 |
|------:|------------------:|-------:|----:|-----------------------------------:|
| block_6  | 0.162 | 0.128 | 0.924 | 0.2% |
| block_7  | 0.159 | 0.128 | 0.905 | 0.1% |
| block_8  | 0.161 | 0.128 | 0.914 | 0.1% |
| block_9  | 0.162 | 0.127 | 0.924 | 0.1% |
| block_10 | 0.161 | 0.127 | 0.932 | 0.1% |
| block_11 | 0.159 | 0.126 | 0.913 | 0.1% |
| block_12 | 0.159 | 0.126 | 0.951 | 0.1% |
| block_13 | 0.158 | 0.126 | 0.845 | 0.1% |
| block_14 | 0.158 | 0.127 | 0.845 | 0.1% |
| block_15 | 0.157 | 0.127 | 0.904 | 0.2% |
| **ln_f** | **0.169** | 0.129 | 0.904 | **0.4%** |

## Reading

1. **Specialisation is uniform-by-depth.** Block-level mean cosine sits
   in the tight band [0.157, 0.162] across blocks 6–15 (range 0.005,
   coefficient of variation ~1%). The "modes specialise" finding is
   not a single-layer phenomenon — it holds throughout the divergence
   zone.

2. **Pre-head (ln_f) is slightly LESS specialised than internal
   blocks.** Mean cos at ln_f = 0.169, mean cos at all blocks ≤ 0.162.
   This is opposite to what you'd expect if specialisation arose from
   the explicit `head_diff_proj` layer alone — that would make ln_f
   the *most* specialised hookpoint. The block-level values being
   slightly lower than ln_f says the backbone itself learns specialised
   circuits, and the pre-head residual is a slight aggregation back
   toward shared structure.

3. **The original Paper C P.2 finding (0.169 on ln_f) was an
   underestimate** of how disjoint the circuits actually are at the
   block level (0.157–0.162). The paper claim should be tightened
   from "modes specialise at the pre-head residual" to "modes
   specialise uniformly across all backbone layers in the divergence
   zone."

4. **Share≥0.7 (the "highly aligned" tail) is essentially zero
   everywhere.** Block-level: 0.1–0.2% of AR features have a diff
   sibling at cosine ≥ 0.7. ln_f: 0.4%. The bridge-feature population
   is small at every depth, but slightly larger at the pre-head
   residual (consistent with point 2 — some shared structure leaks
   back at the very last layer).

## Paper-claim revision (recommended)

**Current abstract (Paper C, after the pivot edit):**
> Sparse-autoencoder analysis of the F10 (305M) backbone shows the
> two heads operate on near-disjoint feature bases: mean cross-head
> cosine 0.169 on 16,384 TopK SAE features at the pre-head residual,
> 0 index overlap on top-8 features across 12 prompts, ...

**Suggested tighten (next paper revision):**
> Sparse-autoencoder analysis of the F10 (305M) backbone shows the
> two heads operate on near-disjoint feature bases: mean cross-head
> cosine in [0.157, 0.162] across all ten blocks of the divergence
> zone (and 0.169 at the pre-head residual), with the bridge-feature
> tail (cosine ≥ 0.7) accounting for under 0.5% of features at any
> depth. Specialisation is uniform across the backbone, not a
> pre-head artefact of the head_diff_proj projection.

## Provenance

- Driver: `e5/interp/cross_head_overlap_by_depth.py`
- Raw output: `e5/interp/results/cross_head_overlap_by_depth.json`
- SAE inventory: 22 files in `e5/interp/saes/` (10 AR × 10 layers +
  10 diff × 10 layers + ln_f_ar + ln_f_diff).
- SAEs durably stored at `huggingface.co/eren23/sfumato-composite-ckpts/tree/main/interp/saes`.

## What this unlocks for Paper D

If a follow-up paper goes the mechanistic-interp route (per the
situation report's recommended PoV pivot 5c), this table is the new
headline figure. Adding error bars across seeds + steering on
top-cos features at each depth would deliver a complete causal story.
