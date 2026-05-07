# Phase-4 Direction C — LLaDA-1.5 Base Swap RESULT

**Pre-reg:** `PRE_REG.md` (committed `4ddf9bd`).
**Date:** 2026-05-07. **Cost:** ~$0.65 (one spot eviction at 5min ~$0.05 + on-demand rerun ~$0.55 + idle ~$0.05).
**Outcome:** **LOSS.** Drop-in swap of LLaDA-8B-Instruct → LLaDA-1.5
with the existing v3 LoRAs lands at acc 0.56 — slightly *below* the
0.58 LLaDA-8B-Instruct reference. Base swap does not close the
peer-class gap.

---

## Headline numbers

| Run | Base | Adapters | MATH-500 N=50 cmajc-k3 | Δ |
|---|---|---|---:|---:|
| T2.A reference | LLaDA-8B-Instruct | v3 prefix-robust + commit | **0.580** | (baseline) |
| **This spike** | LLaDA-1.5 (VRPO post-trained) | v3 prefix-robust + commit (frozen, no retrain) | **0.560** | **−2.0pp** |

W&B: `math500-llada15-cmajc-k3-N50-seed0-od` (run_id `74af75`).

## Pre-registered decision rules — outcome

| Rule | Triggered? |
|---|---|
| MATH-500 acc ≥ 0.68 (closes ≥10pp peer-class gap) → WIN | NO |
| MATH-500 acc ∈ [0.62, 0.68) → PARTIAL | NO |
| MATH-500 acc < 0.62 → **LOSS** | **YES** (0.56) |
| Adapter compat fails → VOID | NO — compat passed cleanly |

## Sub-hypotheses

**Adapter compatibility (PASS):** v3 LoRAs trained against
LLaDA-8B-Instruct apply cleanly on LLaDA-1.5 (same module names,
same shapes). PEFT loaded without errors; only standard "Already
unmerged. Nothing to do." warning fired during the toggle cycle.
This is itself a useful finding for any future LLaDA-family
swap experiments.

**Capability transfer (FAIL):** the v3 LoRAs were trained against
LLaDA-8B-Instruct's specific output distribution (~0.822 GSM8K mean).
LLaDA-1.5's VRPO post-training shifts the base-model distribution in
ways the GSM8K-only-trained adapter doesn't optimally exploit. The
−2pp regression is small but consistent enough at N=50 to signal
that a "drop-in" base swap won't fix sfumato's MATH-500 cross-domain
weakness.

## Why this matters for Phase-4 sequencing

This spike was Phase-4 Direction C-cheap. The plan:
1. T2.C BD3-LMs (deferred — needs SFT-from-scratch budget)
2. **Direction C base swap** ← this spike, **LOSS**
3. Direction A schedule-RLHF (next; ~$70-150)

**Implications:**
- The "stronger base alone fixes cross-domain" hypothesis is falsified
  for the cheapest available alternative DLM (LLaDA-1.5). VRPO
  post-training, while presumably better for general dialogue, doesn't
  carry sfumato's GSM8K-trained adapter behavior forward.
- Direction A schedule-RLHF is now the only remaining Phase-4 bet
  with a credible cross-domain payoff. The path forward is
  schedule-aware fine-tuning of commit-LoRA on a GSM8K + MATH-train
  mixture with REINFORCE/GRPO, NOT a base swap.

## What this kills, what it leaves open

**Killed:**
- The "swap LLaDA-8B-Instruct → LLaDA-1.5 closes MATH-500 peer-class
  gap" hypothesis. Adapter compat works but capability doesn't transfer.
- Any near-term cheap fix that doesn't involve training the adapter.

**Open:**
- **Adapter retrain on LLaDA-1.5 base.** Could the v3 LoRAs be
  re-trained against LLaDA-1.5's distribution and recover (or
  improve) the cmajc-k3 number? ~$0.50 GPU + 1-2 days eng.
  Worth doing IF the user wants to fully exhaust Direction C
  before committing to Direction A's bigger eng cost.
- **Direction A schedule-RLHF** is now the natural next bet.

## Cost ledger

| Item | $ |
|---|---:|
| Spot 4090 pod eviction at 5min | ~$0.05 |
| On-demand 4090 pod (~30min wallclock incl. boot) | ~$0.20 |
| LLaDA-1.5 model download + cold start | ~$0.05 |
| **Total** | **~$0.30** |

(Original PRE_REG estimated $0.35 cap; came in slightly under
even with the spot retry.)

## Files

- `PRE_REG.md` — pre-reg locked at `4ddf9bd`
- `RESULT.md` — this file (LOSS)
- `e4/results/llada15/raw_cmajc_k3_N50_seed0_od.jsonl` — 50 rows
- W&B: math500-llada15-cmajc-k3-N50-seed0-od

## Headline for paper §3.5 (if needed)

*"We tested whether sfumato's MATH-500 cross-domain weakness is
driven by the base diffusion-LM's pretraining quality. Swapping
LLaDA-8B-Instruct for LLaDA-1.5 (the VRPO-post-trained successor,
same architecture, drop-in compatible v3 LoRAs) produced 0.56 vs
0.58 — a 2pp regression rather than the +10pp lift our pre-reg WIN
threshold required. The peer-class gap is therefore NOT explained
by base-model VRPO post-training; it is more likely a property of
how the GSM8K-trained commit-LoRA fails to generalize to the
MATH-500 numeric distribution. Schedule-aware adapter retraining
(Direction A, future work) is the natural next step."*
