# T0 — toy composite AR+diffusion training, seed=54

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **0.0%** | 161.7M |
| **B3 paired_separate** | **2.0%** | 144.2M (paired) |
| **Δ (composite − B3)** | **-2.00pp** | |

**Verdict: `inconclusive`** — Within ±3pp. Reread Transfusion/BD3 ablations for scale.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 4.0% |
| composite | paired (self) | 0.0% |
| B1 ar_only | ar | 0.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 2.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 238s | 3.619 | 6.005 |
| ar_only (B1) | 238s | 3.545 | n/a |
| diff_only (B2) | 236s | n/a | 6.521 |
| B3 ar_sub | 145s | 3.261 | n/a |
| B3 diff_sub | 151s | n/a | 6.614 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
