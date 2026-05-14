# T0 / Phase E — TMLR upgrade verdict

## Overall: INCOMPLETE — pending: ['incomplete']

- E3a: composite_ms64_32 = 4.00±0.57% vs ar_only 1.60±0.36%, Δ = +2.40pp → strong_workshop
- E3b: composite wins at 3/3 scales (>0.2 NLL) → TMLR
- E3c: incomplete
- 
OVERALL: INCOMPLETE — pending: ['incomplete']

## E3a: probe-5 mode-switching (multi-seed)

n_seeds = 5

**composite**:
  - ar_only: mean=2.80%, SEM=0.91%, n=5, per-seed=[2.0, 0.0, 4.0, 6.0, 2.0]
  - mode_switch_96_32: mean=2.80%, SEM=1.07%, n=5, per-seed=[0.0, 0.0, 4.0, 4.0, 6.0]
  - mode_switch_64_32: mean=4.00%, SEM=0.57%, n=5, per-seed=[6.0, 2.0, 4.0, 4.0, 4.0]
  - paired_64_64: mean=2.40%, SEM=0.67%, n=5, per-seed=[2.0, 0.0, 4.0, 2.0, 4.0]

**ar_only**:
  - ar_only: mean=2.00%, SEM=0.57%, n=5, per-seed=[4.0, 0.0, 2.0, 2.0, 2.0]
  - mode_switch_96_32: mean=1.60%, SEM=0.36%, n=5, per-seed=[2.0, 0.0, 2.0, 2.0, 2.0]
  - mode_switch_64_32: mean=1.60%, SEM=0.36%, n=5, per-seed=[0.0, 2.0, 2.0, 2.0, 2.0]
  - paired_64_64: mean=1.60%, SEM=0.67%, n=5, per-seed=[2.0, 0.0, 2.0, 4.0, 0.0]

## E3b: D2 multi-scale compute-matched control

| Scale | composite-3k NLL | pure-diff-6k NLL | composite lead (pd6k − comp_3k) | n |
|---|---|---|---|---|
| 120M | 5.68 | 6.06 ± 0.017 | +0.38 | 3 |
| 300M | 5.35 | 6.09 ± 0.05 | +0.74 | 3 |
| 60M | 5.74 | 6.091 ± 0.031 | +0.351 | 3 |
