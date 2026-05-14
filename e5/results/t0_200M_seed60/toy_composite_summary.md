# T0 — toy composite AR+diffusion training, seed=60

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **2.0%** | 161.7M |
| **B3 paired_separate** | **4.0%** | 144.2M (paired) |
| **Δ (composite − B3)** | **-2.00pp** | |

**Verdict: `inconclusive`** — Within ±3pp. Reread Transfusion/BD3 ablations for scale.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 2.0% |
| composite | paired (self) | 2.0% |
| B1 ar_only | ar | 4.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 4.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 238s | 3.496 | 5.818 |
| ar_only (B1) | 237s | 3.370 | n/a |
| diff_only (B2) | 236s | n/a | 6.416 |
| B3 ar_sub | 152s | 3.140 | n/a |
| B3 diff_sub | 161s | n/a | 6.326 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
