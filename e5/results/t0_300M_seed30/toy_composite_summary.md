# T0 — toy composite AR+diffusion training, seed=30

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **0.0%** | 341.6M |
| **B3 paired_separate** | **4.0%** | 284.9M (paired) |
| **Δ (composite − B3)** | **-4.00pp** | |

**Verdict: `empirical_negative`** — Composite < B3 − 3pp. Toy-scale negative for the vision; redesign or scale-up.

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | 4.0% |
| composite | paired (self) | 0.0% |
| B1 ar_only | ar | 0.0% |
| B2 diff_only | diff | 0.0% |
| B3 paired_separate | paired (cross-model) | 4.0% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | 271s | 4.140 | 6.280 |
| ar_only (B1) | 270s | 3.791 | n/a |
| diff_only (B2) | 270s | n/a | 6.409 |
| B3 ar_sub | 125s | 3.550 | n/a |
| B3 diff_sub | 124s | n/a | 6.528 |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
