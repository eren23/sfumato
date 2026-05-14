# T0 / Phase E — TMLR upgrade verdict

## Overall: TMLR with caveats / strong workshop

- E3a: composite_ms64_32 = 4.00±0.57% vs ar_only 1.60±0.36%, Δ = +2.40pp → strong_workshop
- E3b: composite wins at 3/3 scales (>0.2 NLL) → TMLR
- E3c: deltas=[-0.165, 0.052, 0.18], monotone=True, crossover=True → TMLR
- 
OVERALL: TMLR with caveats / strong workshop

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

## E3c: D3 crossover refinement

| Data size | composite | ar_only | Δ (composite − ar) | n |
|---|---|---|---|---|
| 800p | 3.53 | 3.695 | -0.165 | 3 |
| 1200p | 3.194 | 3.142 | +0.052 | 3 |
| 1500p | 3.09 | 2.91 | +0.180 | 3 |
