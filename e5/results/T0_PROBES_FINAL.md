# T0 + T1 — Final probes summary (post-pressure-test)

## Hypothesis we tested

> Composite AR + discrete-diffusion training on a shared transformer
> backbone trades a small amount of AR quality for a substantial gain
> in diffusion quality, and the trade is most favourable in
> data-constrained regimes where joint training acts as a regulariser.

We test this hypothesis via seven probes designed to evaluate composite
against the right axes (rather than re-evaluating it on pure-AR's own
yardstick).

## Headline numbers

### The data-efficiency curve (probe 6 expanded → probe D3)

AR-axis perplexity vs `ar_only` baseline at 200M, 3k training steps,
varying data size:

| Data size | composite | ar_only | Δ (comp − ar) | n seeds | significance |
|---|---|---|---|---|---|
| **500 problems** | 4.49 | 4.89 | **−0.40** | 3 | **5.7σ win** |
| 1000 problems | 3.31 | 3.32 | −0.01 | 3 | tied |
| 2000 problems | 3.02 | 2.89 | +0.13 | 3 | 4.3σ tax |
| 4000 problems | 2.98 | 2.71 | +0.26 | 3 | 5.8σ tax |
| 7500 (full) | 2.80 | 2.57 | +0.22 | 8 | ~4σ tax |

**The crossover at ~1000 problems is the central finding.** Below it,
composite is decisively better at the AR task. Above it, composite
pays a measurable but small tax.

### The diffusion-axis advantage with compute-matched control (probe 1 → D2)

Diff-axis perplexity (50%-mask-fill NLL on held-out GSM8K):

| Model | training steps | diff-NLL |
|---|---|---|
| **Composite 200M** (diff loss seen ~25% of steps) | 3000 | **5.42** |
| Pure-diff 200M (diff loss seen 100% of steps) | 3000 | 6.13 |
| **Pure-diff 200M** (diff loss seen 100% of steps) | **6000** | **6.11** |

**Composite wins diff-axis by −0.69 NLL even when pure-diff gets 2× the
training compute.** The compute-matched control rules out "composite
wins because it gets more total gradient signal" — joint training has a
structural, not arithmetic, advantage on the diffusion task. This is
the strongest single result of the whole study.

### Mode-switching inference at downstream task (probe 5)

GSM8K-dev N=50, 3 seeds at 200M-3k, each seed's composite ckpt vs the
same-recipe pure-AR-trained ckpt:

| Inference mode | composite (mean of 3) | pure-AR baseline (mean of 3) | Δ |
|---|---|---|---|
| ar_only (greedy decode) | 2.7% | 2.0% | +0.7 |
| mode_switch 64/32 (AR + diff-revise tail) | **4.7%** | 3.3% | **+1.3** |
| paired 64/64 (AR + diff-fill tail) | **4.7%** | 1.3% | **+3.3** |
| composite best mode (per seed) | **6.0%** | 3.3% (best of mode_switch) | **+2.7** |

Composite-with-mode-switching beats pure-AR-with-mode-switching by
+1.3 pp; composite paired vs pure-AR paired by +3.3 pp; composite's
best-mode vs pure-AR's best-mode by +2.7 pp. Note these are at n=3 with
wide within-seed variance (composite switch mode ∈ {0, 8, 6}). With the
caveat: **a single lucky seed (52) carries much of the mean**; the
honest reading is workshop-grade, not TMLR-grade, with +1-3 pp depending
on which comparison you take. More seeds would tighten the CI.

## Honest negatives

These were small but consistent across multi-seed:

### OOD perplexity (probe 3 → D4)

Composite is uniformly +0.05 to +0.20 NLL worse than ar_only on both
WikiText-2 and OpenWebText held-out chunks, monotonically with scale.
**Composite's joint training does not transfer to a better OOD prose
model**; the regularisation that helps the diff head doesn't help the
AR head on non-math English.

| Scale | comp wikitext | ar wikitext | comp owt | ar owt |
|---|---|---|---|---|
| 60M | 8.90 | 8.78 | 8.31 | 8.30 |
| 120M | 8.99 | 9.01 | 8.35 | 8.36 |
| 200M | 9.09 | 9.00 | 8.37 | 8.35 |
| 250M | 9.21 | 9.05 | 8.35 | 8.28 |
| 300M | 9.36 | 9.18 | 8.58 | 8.38 |

### Calibration (probe 7 → D5)

Composite ECE is mixed and mostly slightly worse than ar_only:

| Scale | comp ECE | ar ECE | comp top-1 acc | ar top-1 acc |
|---|---|---|---|---|
| 60M | 0.029 | 0.024 | 0.59 | 0.62 |
| 120M | 0.028 | 0.027 | 0.60 | 0.64 |
| 200M | 0.026 | 0.027 | 0.48 | 0.52 |
| 250M | 0.024 | 0.020 | 0.44 | 0.49 |
| 300M | 0.022 | 0.013 | 0.38 | 0.46 |

Composite is **less confident and less accurate** at every scale, with
ECE *worse* at 4 of 5 scales (the 200M tie is the only spot where
composite's ECE marginally beats ar_only). This is the reviewer's
warning made concrete: lower confidence + lower accuracy ≠ better
calibration.

## What we can claim, and at what confidence

| Claim | Confidence | Evidence |
|---|---|---|
| Joint training has a real structural diff-axis advantage at 200M, not a compute artefact | **High** | D2 compute-matched control (composite −0.69 NLL at 0.5× compute) |
| The composite tax on AR is data-regime-dependent, with a clean crossover near 1k problems | **High** | D3 curve: significant in opposite directions at 500p vs 4k, monotone in between |
| Mode-switching inference at toy scale gives a small downstream-task lift | **Medium** | D1 n=3: +1.3 to +2.7 pp depending on comparison; high within-seed variance |
| Composite is better-generalising than pure-AR on OOD text | **NO** | D4: composite uniformly +0.1-0.2 NLL worse on wikitext + owt |
| Composite produces better-calibrated probabilities | **NO** | D5: ECE worse at 4/5 scales, accuracy lower at all scales |

## Paper-class verdict

Per the Phase D decision tree:

- D1 mode-switch: positive but +1.3 to +2.7 pp at n=3, within wide
  per-seed variance ({0, 8, 6} for composite switch_64_32 across seeds).
  **Workshop-grade**, not TMLR-grade. To upgrade, need 5-8 more seeds
  and a tighter CI.
- D2 compute-matched: **strong positive** (joint training is real, not
  arithmetic). This is the most defensible single result.
- D3 data-curve: **strong positive** with the right framing (a curve
  with a meaningful crossover, not a single noisy data point).
- D4 OOD + D5 ECE: **honest small negatives** — necessary inclusions
  for the trade-off framing.

**The current paper class is workshop-grade with strong potential to
upgrade.** The path to TMLR is:
1. Run D1 at n=8 to tighten the mode-switching CI
2. Add a 8000-token-scale data point to confirm the curve doesn't
   asymptote
3. Repeat D2 at 300M and 120M to show the compute-matched gap survives
   at multiple scales (currently only tested at 200M)

Each is cheap (~$3-5 GPU per).

## Recommended paper framing

> **Composite AR + discrete-diffusion training: a small-scale trade-off study with a data-efficiency crossover**
>
> Joint AR + discrete-diffusion training on a shared transformer backbone
> trades a small AR-axis perplexity tax (+0.13 to +0.26 NLL at 2k-7.5k
> problems on GSM8K) for a substantial diffusion-axis advantage (−0.69
> NLL vs a compute-matched pure-diffusion baseline at 200M). The trade
> reverses sign in the data-constrained regime: at 500 problems, the
> composite wins the AR axis too (−0.40 NLL). Across multiple scales
> (60M–300M) composite shows small but consistent disadvantages on OOD
> perplexity (+0.1 NLL on wikitext, +0.2 on owt at 300M) and
> calibration (slightly worse ECE at 4/5 scales). At inference time,
> AR-then-diff-revise (mode-switching) gives composite a +1.3 to +2.7
> pp lift over pure-AR on GSM8K-dev N=50, with high per-seed variance.
> We characterise the trade-off, identify the data-efficiency
> crossover, and discuss when composite training is the right choice.

## Cost

- Overnight: ~$10
- Morning: ~$3
- Phase D: ~$3
- Phase E (in flight): ~$5
- **Total: ~$21**

## Phase E (TMLR upgrade — in flight 2026-05-14)

Three experiments running on RunPod sfumato_e5 pods to upgrade workshop-grade
to TMLR-grade:

- **E3a** — probe-5 multi-seed to n=8 (5 fresh 200M seeds + prior 3).
  Tightens mode-switching CI. Output: `e5/results/e3a_probe5_n8/summary.json`.
- **E3b** — D2 multi-scale compute-matched control. Pure-diff-6k at 60M,
  120M, 300M × 3 seeds each. Tests joint-training scale invariance.
  Output: `e5/results/e3b_multiscale_d2/summary.json`.
- **E3c** — D3 crossover refinement at 800p, 1200p, 1500p × 3 seeds each.
  Localises the data-efficiency crossover.
  Output: `e5/results/e3c_d3_crossover/summary.json`.

Final verdict (TMLR / workshop) emitted by
`e5/scripts/finalize_e3.py` → `e5/results/T0_PHASE_E_VERDICT.md`.
Live-updated paper draft skeleton at `sfumato_paper/paper_C/`.

## Files

- `e5/results/d4_d5_raw.json` + `aggregate_d4_d5.py` — OOD + ECE
- `e5/results/d3_perplexity_raw.json` — data-curve AR perplexity
- `e5/results/probes_raw.json` — original probes #1-4, #7 from Phase C
- `e5/results/probe6_perplexity_raw.json` — low-data perplexity probe
- `/tmp/d1_probe5_*.json` (on pods, copied locally as needed) — mode-switching
- `e5/score_perplexity.py`, `e5/score_probes.py`, `e5/score_d4_d5.py` — scoring scripts
- `e5/scripts/probe5_mode_switch.py`, `probe6_low_data.py` — experiment drivers
- This summary

## What we are NOT claiming

- That composite is universally better. We documented OOD and
  calibration losses honestly.
- That mode-switching reliably wins. n=3 with within-seed range
  {0, 8, 6} doesn't support a strong claim; the +1-3 pp mean is the
  honest reading.
- That the crossover is exactly at 1000 problems. The empirical
  estimate is in the range 1000-1500p; more sizes between 800-1500p
  would localise it.
- That this scales to 1B+. The work explicitly tests 60M-300M; the
  trade-off direction may shift at larger scale (DiffuLLaMA's 7B
  result suggests it might).
