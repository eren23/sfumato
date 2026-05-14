# T0 — toy composite AR+diffusion training, seed=3

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **2.0%** | 51.3M |
| **B3 paired_separate** | **0.0%** | 60.3M (paired) |
| **Δ (composite − B3)** | **+2.00pp** | |

**Verdict: `within_noise_positive`** — Positive within noise. Multi-seed (+$200) before scaling.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 2.0% |
| composite | paired (self) | 2.0% |
| B1 ar_only | ar | 0.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 0.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 58s | 3.234 | 6.505 |
| ar_only (B1) | 58s | 3.064 | n/a |
| diff_only (B2) | 66s | n/a | 6.497 |
| B3 ar_sub | 43s | 3.220 | n/a |
| B3 diff_sub | 50s | n/a | 6.434 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
