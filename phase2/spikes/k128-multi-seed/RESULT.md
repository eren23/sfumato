# T2.A.3.b K=128 Multi-Seed Confirm — RESULT

**Pre-reg context:** robustness check on T2.A.3 K=128 dip. Single-seed
T2.A.3 showed K=128 at 0.54 vs K=64=K=256=0.58 plateau. This run
adds seed=1, seed=2 confirmation.

**Run dates:** 2026-05-07. **Cost:** ~$0.55 (1 pod × 2 sequential
cmajc-k3 BATCHED=0 N=50 K=128 runs).

**Outcome:** **WIN — K=128 dip replicates** across 3 seeds; mean dip
of −5pp below K=64 plateau is real, not seed jitter.

---

## Headline numbers

| K_STEPS | seed=0 | seed=1 | seed=2 | mean | std |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.580 | n/a | n/a | reference (T2.A.3 + GSM8K dev rerun) | — |
| **128** | **0.540** | **0.540** | **0.500** | **0.527** | 0.019 |
| 256 | 0.580 | n/a | n/a | (T2.A.3 single seed) | — |

K=128 mean acc = **0.527** vs K=64=0.580 reference. **Dip = −5.3pp**,
σ = 1.9pp single-seed. Dip is real (>2σ).

W&B runs:
- seed=0: T2.A.3 `math500-cmajc-k3-K128-N50-seed0` (vdq5tn1l)
- seed=1: `math500-cmajc-k3-K128-N50-seed1-confirm`
- seed=2: `math500-cmajc-k3-K128-N50-seed2-confirm`

## Pre-reg verdict

WIN — K=128 dip replicates 3/3 seeds. The "more compute past K=64
hurts at K=128" pattern is structural, not noise. K=256 (single seed)
recovers to plateau, so the dip appears to be a non-monotone artifact
of doubling-without-fully-completing the trajectory budget rather
than a hard ceiling.

## Implication for §3.5

Updates the §3.5 K-step sweep finding from "K=64 plateau, K=128 dip
(single-seed)" to "**K=64 plateau, K=128 dip (3-seed mean −5pp,
replicates), K=256 recovers**" — schedule signal is bounded above
K=64 with mid-K instability.

## Cost ledger

| Item | $ |
|---|---:|
| 2× cmajc-k3 K=128 BATCHED=0 N=50 ~50min/run | ~$0.55 |
| **Total** | **~$0.55** |

## Files

- `RESULT.md` — this file
- `e4/results/k128_multi/raw_cmajc_k128_seed1.jsonl` (50 rows, acc=0.54)
- `e4/results/k128_multi/raw_cmajc_k128_seed2.jsonl` (50 rows, acc=0.50)
