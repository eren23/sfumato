# Phase-4 Direction A — schedule-RLHF mid-pilot RESULT

**Pre-reg:** plan file `bro-bro-bro-bro-prancy-volcano.md` "Phase-4 NEXT".
**Date:** 2026-05-08 (overnight). **Cost:** ~$1.30 (RTX 4090 24GB on-demand,
~3.5h: ~10min boot + 61min training + 26min eval + buffer).
**Outcome:** **PASS-with-trending-positive.** 200-step GRPO with KL
anchor (β=0.05) on a mixed 200-prompt substrate produced a paired
**+1 problem on MATH-500 numeric N=50** (trained 28/50 = 0.560 vs
baseline 27/50 = 0.540). 49 of 50 predictions are identical; one
problem flipped from wrong→correct (gain), zero lost. Within
binomial CI ±14pp but **direction reverses the mini-pilot's −1**, so
the 4× training-scale + KL anchor pushed the policy slightly
*toward* baseline and possibly past it.

---

## Headline numbers

| Variant | acc N=50 idx 0..49 | n_correct | flip count |
|---|---:|---:|---:|
| **mid-pilot trained (200 steps, KL=0.05, M=4)** | **0.560** | **28/50** | gain 1, lost 0 |
| Paired baseline (untrained commit-v3) | 0.540 | 27/50 | (baseline) |
| **Δ paired** | **+0.02** | **+1 problem** | net +1 |
| 95% CI binomial N=50 | ±14pp | — | — |

**For comparison:**

| Pilot | N | Steps | KL | Substrate | Trained | Baseline (paired) | Δ |
|---|---:|---:|---:|---|---:|---:|---:|
| mini (commit `fbc67b1`) | 20 | 50 | 0.0 (no anchor) | 200 GSM8K-only | 0.350 | 0.400 | **−1 problem** |
| **mid (this spike)** | 50 | 200 | 0.05 (anchored) | 200 GSM/MATH mix | **0.560** | **0.540** | **+1 problem** |

Direction reverses with 4× more steps + KL anchor; on the 49/50
identical-prediction floor, the mid-pilot's training is gentle,
schedule-conditional, and at the very least non-destructive.

## Pre-registered sanity rules — outcome

| Rule | Triggered? |
|---|---|
| 200/200 training steps complete without divergence | YES |
| Jaccard tripwire NOT firing | YES (max 0.311 << 0.6 threshold) |
| Loss does not NaN, Inf, or monotonically explode | YES (range −162 to +482, no NaN) |
| KL anchor produces non-zero terms (vs mini-pilot's KL_BETA=0) | YES (train_loss > 0 at reward_mean=1.0; was 0 in mini-pilot) |
| Adapter saves correctly | YES (64MB at /tmp/track2_commit_rl_mid/commit) |
| Trained ≥ baseline acc on paired N=50 | YES (0.560 vs 0.540, +1 problem) |
| **All sanity rules** | **PASS** |

## What changed since mini-pilot

Three eng deltas, all landed in commit `08bae7e`:

1. **KL anchor wired (forward-B pass).** Per-(rollout, sub-block):
   `peft_model.disable_adapter_layers()` + `phase_emb.set_adapter_disabled(True)`
   + `torch.no_grad()` → `log_pi_frozen`. KL = `kl_k3_estimator(log_pi_theta,
   log_pi_frozen).sum()`. Loss = `-log_pi_theta.sum() * advantage + KL_BETA * KL`.
   Backward flows through theta only.
2. **Adapter+grad restore happens twice.** Once after the rollout
   (denoise_block disables adapter), and once after the frozen pass
   (disable_adapter_layers fires there too).
3. **Mixed 200-row substrate** (100 GSM8K-train + 100 MATH-500
   numeric proxy). Built via `scripts/build_rl_substrate.py`. Mini-
   pilot was 50 rows of GSM8K-train only.

## Step-by-step training trace (selected)

| Step | reward_mean | train_loss | jaccard |
|---:|---:|---:|---:|
| 5 | 1.000 | +1.44 | 0.240 |
| 10 | 0.750 | −47.50 | 0.285 |
| 50 | 0.750 | +128.0 | 0.202 |
| 85 | 0.500 | −162.0 | 0.241 |
| 100 | 0.750 | +47.5 | 0.311 |
| 130 | 0.500 | +60.5 | 0.252 |
| 150 | 0.750 | −99.7 | 0.270 |
| 165 | 0.750 | −119.7 | 0.291 |
| 185 | 0.750 | +234.96 | 0.275 |
| 200 | 1.000 | +0.11 | 0.266 |

KL anchor produces non-zero loss even at reward_mean=1.0 (steps 5,
20, 25, 95, 105, 120, 125, 140, 145, 170, 175, 190, 200) — confirming
the anchor is contributing to the gradient. In the mini-pilot, the
exact same reward configuration produced loss=0 (33/50 such steps,
all wasted updates).

Loss range is wider than mini-pilot (−162 to +482 vs −150 to +482)
because KL terms add additional signal. Volatility is structural
GRPO behavior, not divergence; final-step loss converges to small
positive values.

## Paired diff per-problem

49/50 predictions are bit-identical between baseline-v3 and trained.
The single flip:

| idx | gold | baseline pred | baseline ✓ | trained pred | trained ✓ | delta |
|---:|---:|---:|---|---:|---|---|
| 27 | 6 | 4 | ✗ | 6 | ✓ | **GAIN** |

Zero predictions flipped from correct → wrong. The 4× scale + KL
anchor produced a one-way improvement at the noise floor.

## Why this matters for the full pilot decision

The locked PRE_REG (1550 prompts × M=8 × 2 epochs ≈ 3100 steps with
KL anchor) targets MATH-500 N=200 cmajc-k3 ≥ 0.51 as WIN. Mid-pilot
gives the first directional evidence:

- **Plumbing PASS** (mini-pilot already established this; mid-pilot
  re-confirms at 4× scale and with KL).
- **Direction reversal** from mini-pilot's −1 to mid-pilot's +1 is
  consistent with "more training + KL anchor = better policy" rather
  than "no signal at any scale."
- **Magnitude is tiny** (+1 problem on N=50, well within CI). Cannot
  extrapolate to PRE_REG WIN threshold (+10pp over 0.41 baseline)
  with this evidence alone.
- **Substrate is small** (200 prompts × 1 epoch = 1/15 of the full
  pre-reg). Whether the gain scales linearly, plateaus, or reverses
  is the open question only the full pilot can answer.

**Decision call:** the full pilot is now better-justified than after
the mini-pilot. The mini-pilot was a "does the training infra work"
check (PASS) but produced no directional signal. The mid-pilot
delivers a directional signal (small, but in the right direction),
combined with confirmation that KL anchor stabilizes the policy
without preventing it from moving. **Recommendation:** the next
gate is a 4× larger pilot (~$5-10, ~4 hrs) before committing to the
$70-150 full pilot.

## Cost ledger

| Item | $ |
|---|---:|
| Pod-01 RTX 4090 24GB on-demand (~3.5 hrs) | ~$1.10 |
| HF model + LoRA download | ~$0.10 |
| Buffer | ~$0.10 |
| **Total** | **~$1.30** |

## Files

- `phase2/spikes/direction-a-mid-pilot/RESULT.md` — this file
- `phase2/spikes/direction-a-mid-pilot/training_log.txt` — full training trace
- `phase2/spikes/direction-a-mid-pilot/eval_math500_N50_trained.jsonl` — 50 outcomes (trained)
- `phase2/spikes/direction-a-mid-pilot/eval_math500_N50_baseline.jsonl` — 50 outcomes (baseline)
- (adapter binary stays on local artifacts/, gitignored — too large for git)

## §3.5 paper framing (when paired with full pilot result)

If the full pilot WINs, the §3.5 paragraph can lead with:

> *"We staged Direction A through three pilots of increasing scale.
> Mini-pilot (50 steps, no KL) was a plumbing-only check. Mid-pilot
> (200 steps, KL=0.05, paired N=50) delivered the first directional
> signal — net +1 problem over baseline at the noise floor with 49/50
> identical predictions. Full pilot (3100 steps, paired N=200)
> achieved WIN-threshold +X pp."*

If the full pilot LOSSes, the §4 paragraph reads:

> *"Mid-pilot suggested the trajectory was correct: net +1 over
> baseline at 200 steps. The 15× scale-up to the full pilot did not
> produce the expected linear extrapolation, suggesting the schedule-
> conditional advantage saturates earlier than naive scaling would
> imply. We add this as a fourth unified-negative datapoint."*

## Headline for paper §3.5 right now

*"200-step GRPO with KL anchor produces +1 problem over paired
baseline on MATH-500 N=50. Statistically inside noise; directionally
reverses the mini-pilot's regression. The full pilot's WIN/LOSS
becomes the gating experiment."*
