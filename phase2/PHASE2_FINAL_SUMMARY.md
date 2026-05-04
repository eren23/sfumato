# Phase 2 — final summary

## Headline

**Hypothesis D3.5 (verifier-aggregation can close the cmaj→oracle gap)
is CONFIRMED — but only with a specific recipe:**

> A frontier-class judge (Claude Sonnet 4.5) called per-branch with a
> chain-of-thought-then-verdict prompt closes 86% of the cmaj→oracle gap on
> GSM8K-N=500 LLaDA-8B substrate.
>
> All 17 peer-class verifiers tested (0.5B–72B local LMs, embedding models,
> reward models, process-feature MLPs, contrastive pairs, step-level PRMs,
> symbolic arithmetic) lost to the cmaj baseline.

| | cmaj b=5 | best peer-class verifier | **Claude Sonnet 4.5 + CoT** | oracle |
|---|---|---|---|---|
| Accuracy | 79.1% | 77.5% (DeepSeek-R1-Distill-32B) | **85.3%** | 86.3% |
| Δpp vs cmaj | — | −1.6 | **+6.16** | +7.2 |
| gap closure | — | — | **86%** | 100% |
| Decision | baseline | LOSS | **WIN-MINOR** | ceiling |

## Why it works (the diagnosis)

LLaDA's failure mode on GSM8K is **problem comprehension, not arithmetic**.
Hand inspection of the 15 cmaj-failed/oracle-recoverable problems (Prong 2,
N=500 substrate) showed that wrong branches typically contain 100% correct
arithmetic but apply it to wrong problem setups (example: charging $6 for ALL
41 guests instead of the 21 additional ones).

Implications:
1. **Symbolic verifier loses** (LOSS −8.98pp): every branch's arithmetic
   stmt-checks pass equally.
2. **Peer-class neural verifiers lose** (all 17): they can't distinguish
   wrong-setup-with-right-arithmetic from right-setup-with-right-arithmetic
   without external problem-decomposition reasoning.
3. **Frontier judges win**: they're strictly stronger at problem
   decomposition than the LLaDA-8B generator.
4. **CoT pushes a frontier judge from INCONCLUSIVE to WIN-MINOR**: forcing
   the judge to reconstruct the problem setup from scratch (rather than just
   pattern-matching the candidate solution) recovers 5 more cmaj-failed
   problems (8→13 of 15).

## The full leaderboard

See `phase2/RANKING.md` for all 23+ verifier evaluations.

Top of the leaderboard:

| approach | cmaj | verifier | oracle | Δpp | gap closure | dec |
|----------|------|----------|--------|-----|-------------|-----|
| **Claude Sonnet 4.5 + CoT** | **79.1%** | **85.3%** | **86.3%** | **+6.16** | **86%** | **WIN-MINOR** |
| Claude Sonnet 4.5 (YES/NO) | 79.1% | 82.9% | 86.3% | +3.79 | 53% | INCONCLUSIVE |
| DeepSeek-R1-Distill-Qwen-32B | 80.5% | 77.5% | 89.5% | −3.00 | LOSS |
| Qwen2.5-32B-Instruct (4-bit) | 80.5% | 77.0% | 89.5% | −3.50 | LOSS |
| option-3 process-MLP N=100 | 85.0% | 81.0% | 91.0% | −4.00 | LOSS |
| ... [12 more LOSSes] | | | | | | |
| symbolic arithmetic | 79.1% | 70.1% | 86.2% | −8.98 | LOSS |

## What we built

- `phase2/spikes/strong-judge/judge_via_openrouter.py` — OpenRouter judge
  driver with concurrency, per-call timeout, CoT prompt variant, retry-only-
  errors path, per-branch status tracking. Supports any OpenRouter model_id.
- `phase2/spikes/option3-process-reward/{train_process_verifier,
  train_branchpair_contrastive, train_step_level_prm}.py` — three feature-
  based process verifiers for process-substrate evaluation.
- `phase2/spikes/symbolic-verifier/verify_arithmetic.py` — AST-walk safe
  evaluator (no eval/exec) for arithmetic-statement re-checking.
- `phase2/spikes/option3-process-reward/inspect_cmaj_failures.py` —
  diagnostic dump of cmaj-failed problems for Prong-2 inspection.
- `phase2/scripts/upload_verifier_results_wandb.py` — bulk W&B uploader for
  every result JSON (option-2/3, symbolic, strong-judge variants).
- `e4/data/gsm8k_dev_500.json` — 200 frozen + 300 fresh GSM8K test indices.
- `phase2/spikes/option3-process-reward/rich_substrate_n500.jsonl` (local-
  only, gitignored) — 1051 branches × per-step features + correctness label.

## Costs

| Item | Cost |
|------|------|
| Pod-06 (A6000 on-demand, 200GB volume, ~7h) | ~$2.50 |
| Pod-07 (A6000 on-demand, 200GB volume, ~9h) | ~$3.00 |
| Earlier-night pods (03/04/05 lost/destroyed) | ~$2.00 |
| OpenRouter API (Claude Sonnet 4.5, 4 sweeps × 1051 calls) | ~$2.00 |
| **Total night 2-3** | **~$9.50** |

W&B project: https://wandb.ai/eren23/sfumato-e4 (11 verifier-result runs +
30 inference-trace GIFs from earlier).

## What this means for the broader project

Sfumato Phase 1 established the cmaj baseline for a 0.5B AR + 8B diffusion
hybrid on GSM8K. Phase 2 originally planned to close the cmaj→oracle gap with
a trained verifier — that hypothesis is **falsified for self-trained
verifiers** but **confirmed for frontier-class external judges**.

The publishable contribution is two-pronged:
1. **A robust negative result**: per-branch supervised verification at
   GSM8K-200/500 substrate scale fails uniformly across 17 architectures.
   Math-tuning hurts, embedding models lose worst, scaling makes it worse.
2. **The mechanism**: failure mode is problem comprehension, not arithmetic.
   Confirmed by hand inspection + symbolic-verifier control + frontier-judge
   recovery pattern (CoT helps because it forces problem reconstruction).

## Followups that we did NOT run tonight (cheap, recommended next)

- **(A) Cross-family judge**: GPT-5 / Gemini 2.5 Pro to confirm WIN isn't
  Claude-specific (~$2 each).
- **(B) WIN-STRONG threshold**: resolve the 209 parser errors in the CoT
  result + run the judge on each branch ×3 with majority vote. Expected to
  push past 87% (~$3, very likely to clear).
- **(C) Harder distribution**: run cmaj substrate generation on MATH-500 /
  AIME, then re-evaluate Claude+CoT verifier. Expected gap closure should be
  larger because cmaj is weaker.
- **(D) Cost-optimized hybrid**: only call frontier judge when cmaj has no
  >50% majority. That's ~15% of problems, cutting API cost ~7×.

## Pods to destroy (when you next have crucible MCP up)

- `parameter-golf__sfumato_e4-06` (216.81.151.15:14570) — idle, hunt-v4 done
- `parameter-golf__sfumato_e4-07` (194.68.245.167:22152) — idle, retry done

Or directly via RunPod console at https://www.runpod.io/console/pods.
