# T0 — toy composite AR+diffusion training, seed=4

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **0.0%** | 51.3M |
| **B3 paired_separate** | **2.0%** | 60.3M (paired) |
| **Δ (composite − B3)** | **-2.00pp** | |

**Verdict: `inconclusive`** — Within ±3pp. Reread Transfusion/BD3 ablations for scale.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 2.0% |
| composite | paired (self) | 0.0% |
| B1 ar_only | ar | 0.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 2.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 109s | 2.945 | 6.161 |
| ar_only (B1) | 70s | 2.753 | n/a |
| diff_only (B2) | 68s | n/a | 6.331 |
| B3 ar_sub | 44s | 2.887 | n/a |
| B3 diff_sub | 45s | n/a | 6.390 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
