# T0 — toy composite AR+diffusion training, seed=40

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **4.0%** | 254.2M |
| **B3 paired_separate** | **2.0%** | 220.4M (paired) |
| **Δ (composite − B3)** | **+2.00pp** | |

**Verdict: `within_noise_positive`** — Positive within noise. Multi-seed (+$200) before scaling.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 0.0% |
| composite | paired (self) | 4.0% |
| B1 ar_only | ar | 4.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 2.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 350s | 3.746 | 6.201 |
| ar_only (B1) | 349s | 3.463 | n/a |
| diff_only (B2) | 347s | n/a | 6.440 |
| B3 ar_sub | 178s | 3.283 | n/a |
| B3 diff_sub | 178s | n/a | 6.342 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
