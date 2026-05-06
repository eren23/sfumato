# Pre-registration — MATH-500 K-step sweep at cmajc-k3

**Date:** 2026-05-06 | **Spike:** Phase-3 bench-polish on top of T2.A
WIN. Cheap (~$0.55) ablation testing whether the K2 inverted-U peak
position is robust to compute-budget changes.

## Hypothesis

T2.A locked cmajc k=3 = 0.58 as the MATH-500 numeric peak at K_STEPS=64.
This spike sweeps K_STEPS ∈ {32, 64, 128, 256} at fixed cmajc-k=3 to
test whether:

1. **The optimal K_STEPS is in [64, 128].** If 32 hurts and 256 doesn't
   help, K=64 is the right compute budget for the K2 ablation conclusion.
2. **More compute (256) doesn't trivially close the cmajc-vs-c2c gap.**
   If it does, the K2 finding becomes "commit-LoRA helps because we're
   under-sampling at K=64" rather than "commit-LoRA carries domain-
   general schedule signal."

## Substrate

- Same MATH-500 numeric subset (`e4/data/math500_numeric_indices.json`),
  N=50 idx 0..49.
- Fixed: cmajc, COMMIT_N_BLOCKS=3, BRANCHES=5, BATCHED=0, TEMP=0.7, SEED=0.
- Sweep: K_STEPS ∈ {32, 64, 128, 256}.

## Decision rules

| Outcome | Verdict |
|---|---|
| K=128 acc ≥ K=64 acc + 2pp AND K=256 acc ≤ K=128 acc + 1pp | **WIN** — K=64 was slightly under-budget; K=128 is the true optimal; T2.A k=3=0.58 is a near-best ceiling |
| K=64 acc is the highest of all 4, ±1pp on neighbors | **WIN-FLAT** — K=64 is the right budget; K2 inverted-U claim solid |
| K=256 acc > K=64 + 5pp | **PARTIAL** — paper §3.5 needs caveat that more compute closes more gap; commit-LoRA story softer |
| K=32 acc ≥ K=64 acc | **NULL** — K-step doesn't matter at this scale; commit-LoRA effect is K-invariant |

## Cost

| Item | $ |
|---|---:|
| 1 pod × 4 runs × ~25 min BATCHED=0 | ~$0.55 |
| **Total** | **~$0.55** |

## Anti-goals

- No multi-seed in this sweep. Same seed=0 throughout. Multi-seed only
  if WIN at single seed.
- No K_STEPS > 256. Diminishing returns; 256 already 4× base compute.
- No reuse of T2.A's K=64 number (re-run for paired comparability).

## Files

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled
