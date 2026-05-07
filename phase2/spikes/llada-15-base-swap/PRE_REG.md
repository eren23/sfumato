# Pre-registration — LLaDA-1.5 Base Swap (Phase-4 Direction C-cheap)

**Date:** 2026-05-07. **Spike:** Phase-4 Direction C — base model swap.
**Status:** Ready to dispatch (replaces the original "BD3-LMs cross-substrate
as cheap gate" plan now that BD3-LMs is confirmed perplexity-only).

## Hypothesis

`GSAI-ML/LLaDA-1.5` (8B, VRPO post-trained, released Oct 2025) is a
drop-in architectural replacement for `GSAI-ML/LLaDA-8B-Instruct`
(the current sfumato base). Swapping the base model alone — keeping
the existing prefix-robust LoRA + commit-LoRA on top — closes some
fraction of sfumato's MATH-500 cross-domain peer-class gap (current:
0.58 vs Qwen 2.5 7B 0.80 = −22pp).

**Two sub-hypotheses tested:**
1. **Adapter compatibility:** the v3 LoRAs trained against
   LLaDA-8B-Instruct still apply cleanly on LLaDA-1.5 (same
   architecture, same module names). If TRUE, no adapter retraining
   needed.
2. **Capability transfer:** with LLaDA-1.5 as base + frozen v3 LoRAs,
   MATH-500 numeric N=50 cmajc-k3 closes ≥10pp of the peer-class gap
   (i.e., reaches ≥0.68 vs current 0.58).

## Substrate

- **Base:** `GSAI-ML/LLaDA-1.5` (swap from `GSAI-ML/LLaDA-8B-Instruct`)
- **Adapters:** Existing `eren23/sfumato-llada-prefix-robust-v3` +
  `eren23/sfumato-llada-commit-v3` (no retraining unless compat fails)
- **Config:** cmajc, K_STEPS=64, BRANCHES=5, BATCHED=0, TEMP=0.7,
  SEED=0, COMMIT_N_BLOCKS=3, MERGE_ADAPTER=1
- **Eval:** MATH-500 numeric N=50 (idx 0..49 of `math500_numeric_indices.json`)
  for direct comparison with T2.A baseline (0.58)
- **Pod:** 24GB 4090 should fit at BATCHED=0; 48GB if shadow forward
  needed for compat verification

## Method

1. **Compatibility smoke (CPU-only, free):**
   - Add `GSAI-ML/LLaDA-1.5` to `e4/diff_llada.py` model registry.
   - Run `MOCK_MODELS=0 N_PROBLEMS=2 BATCHED=0 BRANCHES=1
     CONDITION=c2 DIFF_MODEL=GSAI-ML/LLaDA-1.5` — verify model loads,
     no shape mismatches when LoRA is applied.
   - If LoRA loads with errors, capture the diff (likely module name
     drift between -Instruct and 1.5) and either patch with
     target_modules override or fall back to retrain on LLaDA-1.5.

2. **MATH-500 cmajc-k3 N=50 dispatch:**
   - On a 24GB 4090 spot pod (~$0.20/hr).
   - ~25 min run BATCHED=0, ~$0.10 total.
   - Compare directly with T2.A `math500-cmajc-k3-N50-seed0` = 0.58.

3. **Optional: GSM8K dev-200 cmajc-k3 N=200 BATCHED=1**
   - ~30 min, ~$0.10. Verify in-domain doesn't regress.

## Decision rules

| Outcome | Verdict |
|---|---|
| MATH-500 acc ≥ 0.68 (closes ≥10pp toward peer-class) AND GSM8K dev-200 ≥ 0.80 | **WIN** — base swap fixes cross-domain. §3.5 framing: "schedule-toggle composes with VRPO-post-trained bases" |
| MATH-500 acc ∈ [0.62, 0.68) | **PARTIAL** — modest cross-domain lift, doesn't close peer-class gap |
| MATH-500 acc < 0.62 OR GSM8K dev-200 regresses >2pp | **LOSS** — base swap doesn't solve cross-domain; lean harder on Direction A schedule-RLHF |
| Adapter compat fails AND we don't retrain | **VOID** — re-scope as "needs adapter retrain on new base" follow-up |

## Cost

| Item | $ |
|---|---:|
| Compat smoke (CPU local) | $0 |
| MATH-500 cmajc-k3 N=50 BATCHED=0 (24GB 4090 ~25min) | ~$0.10 |
| GSM8K dev-200 cmajc-k3 BATCHED=1 (24GB 4090 ~30min) | ~$0.15 |
| Buffer | ~$0.10 |
| **Total** | **~$0.35** |

**Drastically cheaper than T2.C BD3-LMs** (which needs SFT first,
~$50). LLaDA-1.5 is the right Direction C bet.

## Anti-goals

- No commit-LoRA retraining unless compat smoke fails. The whole
  point is testing whether the SAME adapters transfer.
- No mid-spike threshold amendments. Any pivot becomes a separate
  follow-up spike.
- No dispatch on a 48GB pod unless 24GB OOMs. Spot 4090 is the right
  cost target for this size of test.

## Files

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled after dispatch
- `e4/diff_llada.py` — line ~524 model_id; small registry edit
- `e4/results/llada15/raw_cmajc_k3_N50_seed0.jsonl` — output
- `e4/results/llada15/raw_cmajc_k3_N200_GSM8K_seed0.jsonl` — output

## Why this matters for Phase-4 sequencing

Per plan: T2.C → Direction C → Phase-4 RLHF.

This spike **replaces T2.C as the cheap fundamental gate** because:
- BD3-LMs requires SFT-from-scratch (perplexity-only base, no math)
- LLaDA-1.5 is a true drop-in swap (same arch, post-trained)
- Cost is $0.35 vs $50 for BD3-LMs at minimal scope

If LLaDA-1.5 closes the peer-class gap → Phase-4 framing becomes
"base swap + schedule-toggle compose." If LLaDA-1.5 doesn't move the
needle → Phase-4 RLHF (Direction A) is the only remaining bet.

The original T2.C BD3-LMs spike PRE_REG (`998b0b0`) stays scaffolded
but deferred until either (a) someone publishes a math-instruction-
tuned BD3-LM, or (b) the user explicitly approves the SFT-from-scratch
budget.
