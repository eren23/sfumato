# T0 — toy composite AR+diffusion training, seed=80

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **0.0%** | 110.2M |
| **B3 paired_separate** | **4.0%** | 102.6M (paired) |
| **Δ (composite − B3)** | **-4.00pp** | |

**Verdict: `empirical_negative`** — Composite < B3 − 3pp. Toy-scale negative for the vision; redesign or scale-up.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 0.0% |
| composite | paired (self) | 0.0% |
| B1 ar_only | ar | 2.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 4.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 606s | 2.168 | 5.874 |
| ar_only (B1) | 601s | 2.221 | n/a |
| diff_only (B2) | 597s | n/a | 6.529 |
| B3 ar_sub | 452s | 2.238 | n/a |
| B3 diff_sub | 477s | n/a | 6.407 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
