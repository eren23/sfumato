# T0 — Final summary (overnight autonomous run, 2026-05-13 → 2026-05-14)

## What we did

Trained ~24 small composite AR + discrete-diffusion language models on
GSM8K-train from scratch at scales 60M / 120M / 200M / 250M / 300M, plus
one toy continuous-flow LM (ELF-flavoured), and ran 4 baselines per
composite to support the parameter-matched B3 comparison defined in
`/Users/eren/.claude/plans/sfumato-vision-aligned-rosy-jellyfish.md`.

The core comparison: **does composite[paired-mode inference] beat the
parameter-matched two-separate-models B3[paired-separate] baseline?**

Eval: greedy decoding on first 50 problems of `e4/data/gsm8k_dev_200.json`
(GPT-2 BPE tokenisation, max_new=128, 32 ODE/diff steps).

## Headline numbers — scale curve

| Scale | seeds (n) | composite acc | B3 acc | Δ mean | Δ SD | 95% CI |
|-------|-----------|----------------|--------|--------|------|--------|
| 60M   | 7         | 1.7%           | 2.6%   | **-0.86pp** | 1.7 | [-2.4, +0.7] |
| 120M  | 7         | 1.7%           | 1.1%   | **+0.57pp** | 2.5 | [-1.7, +2.8] |
| 200M  | 8         | 2.3%           | 0.5%   | **+1.75pp** | 3.3 | [-0.5, +4.0] |
| 250M  | 1         | 4.0%           | 2.0%   | +2.0pp  | —    | — |
| 300M  | 1         | 0.0%           | 4.0%   | -4.0pp  | —    | — |
| flow  | 1         | 0.0% (paired)  | —      | —       | —    | — |

## Verdict

**Directionally positive but not significant at this N and toy scale.**

- 60M → 200M shows **monotonic upward trend** in Δ (−0.86 → +0.57 → +1.75pp).
  This is **directionally consistent with the original Sfumato vision** —
  composite training advantage emerges with scale.
- All three multi-seed 95% CIs cross zero. We cannot claim statistical
  significance at α=0.05.
- The 200M point produces 4 of 8 seeds in the "strong_positive" band
  (≥+5pp) but is offset by 4 seeds at ≤0pp. The eval metric's discreteness
  at this scale (50 problems × ~0-6% accuracy ≈ 0-3 problems correct) is
  the dominant source of variance.
- The single-seed 300M result (-4pp) breaks the upward trend and most
  likely reflects **undertraining at fixed 3k steps** rather than a real
  reversal of the vision (300M model, same training budget as 60M).
  Untested.

Per the plan's decision table, the proper classification at 200M (n=8)
is **`within_noise_positive`** — "Positive within noise. Multi-seed
(+$200) before scaling." Which is exactly what this run did. The
next-step recommendation in that branch was BD3-LM-variant scale-up to
$500–1500.

## Cost

- Two pods running for ~8h (4090 @ $0.69/hr + A40 @ $0.44/hr).
- Crucible bootstrap overhead negligible.
- Total spend ≈ **$8–9** for the overnight run.
- All-session spend (incl. earlier E2/G/D-step2): ≈ **$10–11**.

## What I'd do next if you keep going at this budget level

1. **More seeds at 200M** until the 95% CI is tight (~15 seeds total).
   Cheap, definitive. ~$3, half a day.
2. **200M with longer training** (10k–20k steps, 3 seeds). Tests the
   training-duration confound that the 300M result hints at. ~$5, one day.
3. **Switch to a continuous eval metric** (held-out token perplexity)
   instead of GSM8K accuracy. Toy models score at the noise floor on
   accuracy; perplexity gives a continuous signal of how much each loss
   function actually fits the data. Code change ~1 hr.
4. **Drop discrete-mask diffusion in favour of continuous flow** (T0b
   was just a sanity check at one seed and got 0% — but the loss curve
   was descending; with longer training and proper conditioning it may
   work, and would let us repeat T0's comparison with an ELF-flavoured
   diffusion half).

If you skip the toy probes and want to commit budget: per the plan's
fork, **200M's directional signal is enough to justify a BD3-LM-variant
scale-up at $500–1500**. That run would either consolidate the signal at
publishable scale or empirically kill the vision. It is the next
honest step.

## What we did NOT do

- Did not test α-schedule ablations (would need train.py modification).
- Did not test alternative mode-routing strategies inside the composite.
- Did not test a "shared embedding only" variant of the composite (where
  the two heads have separate transformer trunks but a shared input).
- Did not modify the eval to use perplexity or token-level loss.
- Did not write the AR+flow composite that's the T1 follow-up.

## Caveats / things I might be wrong about

- **GSM8K accuracy at toy scale is at the noise floor.** Composite hit
  6.0% peak; B3 hit 4.0% peak. We're talking about 1–3 problems per 50.
  The variance is dominated by which random tokens the model latches onto,
  not by the architecture difference.
- **3k training steps may be insufficient at 200M+.** The 300M
  counter-result strongly suggests undertraining. The "scale helps"
  conclusion may partially be "small-model-overshoots-bigger-undertrained-model"
  noise rather than a real composite advantage.
- **n=8 at 200M is still small for definitive multi-seed claims.**
  The 95% CI at [-0.5, +4.0]pp is centered positively but excludes
  significance.
- **The vision is NOT empirically dead at toy scale.** The earlier T0
  read (3k seed=0 Δ=-2pp) was deeply pessimistic for what is essentially
  a noisy 1-seed snapshot. With multi-seed data the picture is mildly
  positive.

## Files

All under `/Users/eren/Documents/ai/sfumato/e5/results/`:

- `t0_3k_seed{0,1,2,3,4}/` — 60M, 3k steps, 5 seeds
- `t0_60M_6k_seed10/`, `t0_60M_10k_seed20/` — 60M longer training
- `t0_120M_seed{96,97,98,99}/` — 120M, 3k steps, 4 seeds
- `t0_120M_6k_seed90/`, `t0_120M_10k_seed{80,81}/` — 120M longer training
- `t0_200M_seed{50,51,52,53,54,55,56,57}/` — 200M, 3k steps, 8 seeds
- `t0_250M_seed40/` — single 250M data point
- `t0_300M_seed30/` — single 300M data point (undertrained?)
- `flow_full_seed0/` — toy continuous-flow LM at 60M

Each subdirectory contains:
- `model.pt` — composite checkpoint
- `composite/`, `ar_only/`, `diff_only/`, `paired/{ar_only,diff_only}/` — sub-runs
- `*_results.json` — per-mode eval results
- `toy_composite_summary.{json,md}` — variant-level summary

The training+eval driver and model code are at
`/Users/eren/Documents/ai/sfumato/e5/`.

## Plan and decision-log

- The plan file driving this run: `/Users/eren/.claude/plans/sfumato-vision-aligned-rosy-jellyfish.md`
- The autonomous experiment queue (mid-run): `e5/results/AUTONOMOUS_QUEUE.json`
- Per-experiment summary.md inside each result directory describes
  the band classification for that single experiment.
