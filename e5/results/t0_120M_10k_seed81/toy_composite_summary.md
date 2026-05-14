# T0 — toy composite AR+diffusion training, seed=81

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **2.0%** | 110.2M |
| **B3 paired_separate** | **0.0%** | 102.6M (paired) |
| **Δ (composite − B3)** | **+2.00pp** | |

**Verdict: `within_noise_positive`** — Positive within noise. Multi-seed (+$200) before scaling.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 4.0% |
| composite | paired (self) | 2.0% |
| B1 ar_only | ar | 4.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 0.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 340s | 2.378 | 5.887 |
| ar_only (B1) | 339s | 1.944 | n/a |
| diff_only (B2) | 338s | n/a | 6.628 |
| B3 ar_sub | 196s | 2.075 | n/a |
| B3 diff_sub | 239s | n/a | 6.523 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
