# T0 — toy composite AR+diffusion training, seed=98

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **4.0%** | 110.2M |
| **B3 paired_separate** | **2.0%** | 102.6M (paired) |
| **Δ (composite − B3)** | **+2.00pp** | |

**Verdict: `within_noise_positive`** — Positive within noise. Multi-seed (+$200) before scaling.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 4.0% |
| composite | paired (self) | 4.0% |
| B1 ar_only | ar | 2.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 2.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 173s | 3.242 | 6.352 |
| ar_only (B1) | 175s | 3.454 | n/a |
| diff_only (B2) | 170s | n/a | 6.682 |
| B3 ar_sub | 112s | 3.387 | n/a |
| B3 diff_sub | 96s | n/a | 6.578 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
