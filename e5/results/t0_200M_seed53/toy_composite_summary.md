# T0 — toy composite AR+diffusion training, seed=53

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **6.0%** | 161.7M |
| **B3 paired_separate** | **0.0%** | 144.2M (paired) |
| **Δ (composite − B3)** | **+6.00pp** | |

**Verdict: `strong_positive`** — ≥+5pp lift. Justify scale-up to BD3-LM-variant ($500-1500).

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 0.0% |
| composite | paired (self) | 6.0% |
| B1 ar_only | ar | 2.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 0.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 142s | 3.563 | 6.005 |
| ar_only (B1) | 140s | 3.390 | n/a |
| diff_only (B2) | 140s | n/a | 6.582 |
| B3 ar_sub | 74s | 3.177 | n/a |
| B3 diff_sub | 72s | n/a | 6.293 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
