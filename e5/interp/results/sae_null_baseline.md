# SAE cross-head cosine — null baseline (the control the headline needed)

**Date:** 2026-05-29
**Why:** The mode-specialisation headline (mean best-cosine AR→diff = 0.169)
is only interpretable against a null. TopK SAEs have no canonical basis,
so two SAEs need not align even on identical data. We measure the chance
floor and a same-mode different-seed baseline.

## Result

| comparison | mean best-cosine |
|---|---:|
| AR-seed0 → AR-seed1 (same mode, different seed) | **0.286** |
| AR → diff (cross-mode; the headline) | **0.169** |
| AR → random unit dictionary (chance floor) | **0.124** |

(`mean best-cosine` = for each of the 16,384 decoder columns of the
first SAE, the max cosine to any column of the second; averaged.
d_model = 1024.)

## Reading

- Cross-mode (0.169) sits **below** the same-mode baseline (0.286) and
  only **+0.045** above the chance floor (0.124).
- Same-mode is **+0.162** above chance.
- Going same-mode → cross-mode loses **~72%** of the above-chance
  feature alignment: `1 − 0.045/0.162 ≈ 0.72`.
- **Conclusion:** the two heads share substantially less feature
  structure than two independently-trained SAEs of the *same* head do.
  This is the rigorous statement of "specialisation" — it survives the
  null. But cross-mode is **not** at the chance floor, so a small shared
  "bridge" subspace exists; "near-disjoint"/"orthogonal" would
  overstate it. The defensible word is **substantial specialisation**.

## Conservative caveat

The seed-1 AR SAE was trained for **2,000 steps** vs **8,000** for the
original `ln_f_ar` / `ln_f_diff`. An undertrained SAE has noisier, less
converged features, which tends to *lower* same-mode cosine — so 0.286
is a **conservative (low) estimate** of true same-mode alignment. A
matched-length (8,000-step) retrain would, if anything, widen the
specialisation gap. Flagged as the clean follow-up in the paper's
limitations.

## Reproduce

```bash
# train the seed-1 control SAE (SAE_SEED knob added to train_saes.py)
HOOKPOINT=ln_f.ar SAE_SEED=1 STEPS=2000 DEVICE=mps \
  TOKENS_PATH=$HOME/.cache/sfumato_e5/fineweb_gpt2_200000000.npy \
  OUT_DIR=e5/interp/saes/ln_f_ar_seed1 WANDB_MODE=disabled \
  python3 -u -m e5.interp.train_saes

# then compute the three cosines (load ln_f_ar, ln_f_ar_seed1, ln_f_diff
# decoders, F.normalize rows, mean of row-wise max cross-cosine).
```

Artefacts: `e5/interp/saes/ln_f_ar/sae.pt` (seed0, 8k steps),
`e5/interp/saes/ln_f_ar_seed1/sae.pt` (seed1, 2k steps),
`e5/interp/saes/ln_f_diff/sae.pt` (8k steps). The seed1 `.pt` is local
(not pushed); the cosine numbers above are the durable artefact.
