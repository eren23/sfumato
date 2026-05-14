# T0 — toy composite AR+diffusion training, seed=59

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **4.0%** | 161.7M |
| **B3 paired_separate** | **0.0%** | 144.2M (paired) |
| **Δ (composite − B3)** | **+4.00pp** | |

**Verdict: `within_noise_positive`** — Positive within noise. Multi-seed (+$200) before scaling.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 2.0% |
| composite | paired (self) | 4.0% |
| B1 ar_only | ar | 2.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 0.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 237s | 3.515 | 5.988 |
| ar_only (B1) | 236s | 3.382 | n/a |
| diff_only (B2) | 234s | n/a | 6.469 |
| B3 ar_sub | 149s | 3.121 | n/a |
| B3 diff_sub | 159s | n/a | 6.433 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
