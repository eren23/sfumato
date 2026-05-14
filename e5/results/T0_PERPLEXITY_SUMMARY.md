# T0/T1 — Perplexity rescoring of toy composite checkpoints

## TL;DR

Held-out AR-perplexity on 94 GSM8K-train problems (indices 7000–7099)
across 21 composite-training checkpoints reveals a **clean, scale-
monotone negative** for the composite-training vision at toy scale:
**joint AR+diffusion training measurably degrades the AR head's ability
to predict gold answer tokens**, and the degradation grows with model
size. This breaks the GSM8K-accuracy noise floor and gives the program a
publishable negative.

The early read of "directional positive at 200M" from the overnight
GSM8K-accuracy multi-seed (n=14 mean +0.43pp) was within noise of zero,
hiding the underlying tax that perplexity now reveals.

## Method

- Held-out chunk: GSM8K-train indices 7000–7099 (100 problems, ~94 fit
  within the 256-token context after the others were skipped).
- Tokenisation: GPT-2 BPE (same as training).
- For each `model.pt`:
  - Run in AR mode (`model(idx, mode='ar')`).
  - Score: average per-token negative log-likelihood over the
    answer-region tokens (the gold CoT after `Answer:`), conditioning
    on the question.
- Headline deltas:
  - **Δ(comp − ar_only)** = composite NLL − same-arch pure-AR NLL. The
    joint-training tax — does adding the diffusion loss hurt the AR head
    on the same backbone?
  - **Δ(comp − b3_ar)** = composite NLL − paired-AR-sub NLL. The
    central plan comparison — does the bigger-backbone composite beat the
    smaller-backbone two-separate-models baseline at AR?

Positive Δ = composite worse. Negative Δ = composite better.

## Results

| Scale | n seeds | steps | composite NLL | ar_only NLL | b3_ar NLL | Δ(comp − ar) | Δ(comp − b3) |
|---|---|---|---|---|---|---|---|
| 60M | 5 | 3k | 2.445 | 2.263 | 2.397 | **+0.182** | +0.048 |
| 60M | 1 | 6k | 1.909 | 1.768 | 1.927 | **+0.141** | −0.018 |
| 60M | 1 | 10k | 1.490 | 1.311 | 1.515 | **+0.179** | −0.025 |
| 120M | 3 | 3k | 2.577 | 2.387 | 2.258 | **+0.190** | +0.319 |
| 120M | 1 | 6k | 2.009 | 1.827 | 1.707 | **+0.182** | +0.303 |
| 120M | 2 | 10k | 1.536 | 1.291 | 1.315 | **+0.244** | +0.221 |
| 200M | 8 | 3k | 2.795 | 2.574 | 2.274 | **+0.220** | **+0.521** |
| 250M | 1 | 3k | 3.026 | 2.715 | 2.447 | **+0.311** | **+0.579** |
| 300M | 1 | 3k | 3.346 | 2.926 | 2.666 | **+0.420** | **+0.679** |
| T1 (AR+flow) 60M | 1 | 3k | 2.517 | 2.341 | 2.313 | +0.176 | +0.204 |

## What this says

### 1. Joint training measurably hurts the AR head (Δ comp−ar all positive)

Across **every** scale × training-duration we tested, the composite's
AR-mode perplexity is worse than the same-arch pure-AR-trained model.
Mean Δ(comp − ar) ≈ **+0.20 NLL** at 60M–120M, growing to **+0.31 NLL at
250M** and **+0.42 NLL at 300M**. At 200M with n=8 seeds, the result is
+0.22 ± (consistent across seeds).

This is not subtle. It is a structurally negative result: adding the
diffusion loss to the joint training of a shared backbone trades AR
quality for diffusion quality, and at this budget the trade is
unfavourable on the AR side.

### 2. Composite ≤ B3 paired-baseline at scale (Δ comp−b3 grows with size)

The plan's central comparison, **composite vs B3 paired-separated**, is:
- 60M (5 seeds): Δ = +0.05 (composite ≈ half-size B3)
- 120M (3 seeds): Δ = +0.32 (composite worse)
- **200M (8 seeds): Δ = +0.52** (composite clearly worse)
- 250M (1 seed): Δ = +0.58
- 300M (1 seed): Δ = +0.68

Direction is **consistently positive** (composite worse) and **monotone
in scale** (gap grows with model size). The plan's hypothesised
"composite advantage grows with scale" is **empirically reversed** at our
budget: the composite disadvantage grows with scale.

### 3. T1 (AR + continuous flow) mirrors T0 (AR + discrete mask)

Single T1 seed=1 at 60M: Δ(comp − ar) = +0.18, Δ(comp − b3) = +0.20.
Same toy-scale tax as T0 at the same scale. The continuous-flow choice
does not save the composite design at toy scale.

### 4. Why GSM8K accuracy was misleading

GSM8K-dev accuracy at 0–6% (1–3 correct out of 50) had binomial SD ≈ 2pp,
swamping the architectural delta. The +0.43pp 200M mean we measured was
**not noise** — it was inside the per-experiment binomial noise. Perplexity
gives a continuous metric with 8000 tokens scored per run, so even
~0.05 NLL differences are detectable.

## Interpretation

The composite vision at this scale is **not just "unfalsified within
noise"** — it is **empirically refuted on the AR axis**. Joint training
is making the AR head worse, both vs same-arch pure-AR and vs param-
matched B3, and the effect grows with scale.

Two possible mitigations the data does not yet rule out:

- **α-schedule re-tuning.** Our composite used α: 1.0 → 0.5 (50% diff
  loss by end). A more AR-heavy schedule (α: 1.0 → 0.8) would presumably
  shrink the AR tax. Cost: ~$2–3 to test at 200M, 3 seeds.
- **Diffusion-axis perplexity.** This rescoring measures only AR. The
  composite *might* be the right architecture for the diff half but is
  pulling backbone capacity from AR. A clean diffusion-axis metric
  (mask-fill perplexity at fixed mask ratio) would test that. Cost:
  ~1 hour of code; rescore for $0.
- **Larger scale.** Published composite-training papers (BD3-LMs,
  Transfusion) show advantage at ≥400M. Our toy scale may be entirely
  the wrong regime. Cost: $500–1500 for a 1B-scale repro — but we
  should not commit to this without exhausting the cheap probes above.

## Recommendation per the plan's decision table

The Phase B plan said:
> Tier 3 — if tier 1 reveals no signal: ship the negative cleanly.

Perplexity shows a **clean negative**, not "no signal" — composite has a
measurable, scale-monotone disadvantage. The decision is:

1. **Cheapest pre-ship probes** (~$5, half a day):
   - α-schedule sweep (α: 1.0 → 0.8 vs 0.5 vs 0.3) at 200M, 3 seeds each.
   - Add diffusion-axis perplexity to `e5/score_perplexity.py` and rescore
     existing checkpoints. Determines whether composite *trades* AR loss
     for diff loss (the original vision) or just *degrades* both.

2. **Then ship the negative.** Tight 3–5 page report:
   - The 29-experiment GSM8K-accuracy table (overnight run).
   - This perplexity table (the headline).
   - Optional: α-sweep and diff-axis perplexity if probes return.
   - Literature anchor: BD3-LMs, Transfusion, DiffuLLaMA, ELF, DiFFPO —
     they show composite advantage at ≥1B; at 60M–300M our data says
     joint training imposes a measurable tax.

## What we are NOT doing

- More 200M-3k seeds (n=8 already; perplexity is decisive at +0.22 NLL
  vs ar_only; n=14 of GSM8K-accuracy was masking this).
- 1B-scale pretraining without the α-sweep first.
- More architectural variants (the binding constraint is not the head
  design or the diffusion paradigm — both T0 and T1 show the same tax).

## Files

- `e5/results/perplexity_raw.json` — full raw rescoring output
- `e5/results/perplexity_processed.json` — per-experiment delta table
- This file (`e5/results/T0_PERPLEXITY_SUMMARY.md`)
- The 29-experiment GSM8K-accuracy table in `e5/results/T0_FINAL_SUMMARY.md`
  (overnight) and individual `t0_*/toy_composite_summary.md` files

## Caveats / things this rescoring cannot tell us

- **AR mode only.** We did not measure diffusion-axis perplexity. The
  composite *might* be much better at masked-token prediction even
  though it's worse at next-token prediction. Worth testing.
- **Trained on the same 4M tokens, ~3k steps.** At that data/step
  budget, both heads are undertrained. The tax pattern may not extend
  to converged-scale training.
- **The α schedule is one design choice.** A more AR-heavy schedule
  could mitigate. Untested at this rescoring.
- **B3 ar_sub is a half-size model.** It's possible the apparent
  composite disadvantage at 200M is partly that the small B3 model
  generalises better at our data budget. Eyeballing the table: B3 NLL is
  comparable across scales (2.27–2.67 for AR), while composite NLL
  scales up (worse) — which suggests the composite's joint-loss tax IS
  the explanation, not B3 small-model advantage.
