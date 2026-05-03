# Proposal D3 — Diversity-as-Objective Fine-Tuning for DDLMs

**Author:** Workstream B subagent | **Date:** 2026-05-02 | **Status:** proposal + spike pre-reg

## Motivation

Phase 1 produced an unexpected falsifier of the "Track-1 LoRA collapses
sampling diversity" worry: post-Track-1 v2, the 5/5-branch-agreement rate
*dropped* from 52.4% (base) to 47.5% — LoRA *expanded* diversity rather than
collapsed it (`e2/RESULTS_TRACK1.md`, "Diversity-expansion finding"). And yet
cmaj b=5 accuracy moved only +1.5pp (v2) to −2.0pp (v3). The mechanism question:
**is branch diversity, by itself, the bottleneck on cmaj accuracy?** If yes,
training a DDLM with an explicit branch-disagreement reward should push cmaj
upward. If no — i.e. if the marginal extra branches go to wrong-answer modes —
then the v3 capacity tradeoff is the real story and diversity is not a free
lever.

Most published DDLM fine-tuning recipes (`d1`, `d2`, `diffu-GRPO`, `wd1`,
`coupled-GRPO`, `GIFT`) optimize either single-shot accuracy or token-importance
coverage; **none directly optimize an inter-branch disagreement objective**.
GIFT (2509.20863) re-weights tokens by per-token entropy but at the *single
trajectory* level. d1's diffu-GRPO uses verifiable correctness, not ensemble
diversity. The closest published work is the "self-consistency reward" line
(d1, RLVR survey, AGRPO) which aggregates votes — but the reward is still
*per-branch correctness*, not the diversity of the branch set.

## Mathematical formulation

Let $\pi_\theta$ be the LLaDA sampler with adapter $\theta$. For a problem $x$,
draw $b$ branches $\{y_i\}_{i=1}^b \sim \pi_\theta(\cdot \mid x)$. Let
$\hat a_i = \text{ans}(y_i)$ be the extracted numeric answer of branch $i$, and
$y^\star$ the gold answer. Define:

- **Cmaj reward** $R_{\text{cmaj}}(x, \{y_i\}) = \mathbb{1}[\text{mode}(\{\hat a_i\}) = y^\star]$
- **Branch-disagreement bonus** $D(\{y_i\}) = 1 - p_{\text{maj}}$ where
  $p_{\text{maj}} = \max_a \frac{1}{b}|\{i : \hat a_i = a\}|$
- **Composite reward** $R = R_{\text{cmaj}} + \lambda \cdot R_{\text{cmaj}} \cdot D$

The product term means: diversity is only rewarded *conditional on cmaj being
right*. This avoids the failure mode "model learns to scatter answers
uniformly to maximize $D$ at the cost of correctness."

Training: GRPO-style (à la diffu-GRPO from `d1`) where the group is the $b$
branches, advantage is per-branch $A_i = R - \bar R$ where $\bar R$ is the
within-group mean, and the loss is the diffusion ELBO weighted by $A_i$ over
LoRA params.

Hyperparameters to sweep: $\lambda \in \{0.0, 0.25, 0.5, 1.0\}$ (with
$\lambda = 0$ being the d1/diffu-GRPO baseline), $b \in \{5, 7\}$,
temperature $\tau \in \{0.7, 1.0\}$.

## Quantitative success criteria

- **Win:** cmaj b=5 on GSM8K-test ≥ **82.5pp** (Phase-1 v2 ceiling 81.5pp
  + 1.0pp). At least one $\lambda > 0$ value beats $\lambda=0$ baseline by
  ≥ **+1.5pp** with non-overlapping 95% Clopper-Pearson CIs at N=200.
- **Mechanism check:** the winning $\lambda$ must show a *narrower* 5/5-agreement
  spike than $\lambda=0$ — i.e. the diversity bonus actually changed the
  branch distribution, not just shifted accuracy via some other path.
- **Compute floor:** $\lambda = 0$ baseline must reproduce diffu-GRPO's
  reported +9.9pp gain on GSM8K over base LLaDA (sanity-check the harness).

## Predictable failure modes

1. **Branch-redundancy ceiling:** if base LLaDA's 5 branches already cover the
   right-answer mode 79% of the time (we know it does on test), and the
   remaining 21% are problems where *no* branch lands on the right answer,
   then no amount of diversity bonus can help — there's no correct mode to
   amplify. **Kill signal:** spike result (below) shows
   `agree-rate × cmaj-acc` covariance ≤ 0.1 across temperature.
2. **Reward hacking via tail-pruning:** model learns to suppress numerical
   tokens entirely on hard problems so the answer-extractor returns "" for
   most branches, mechanically inflating disagreement. **Mitigation:** require
   $\geq b/2$ branches to extract a parseable answer; otherwise the reward
   is zero.
3. **Capacity tradeoff replay:** v3 already showed a capacity-vs-diversity
   tradeoff at 7/7 LoRA modules. Adding a reward term may push the model
   back into v1-style mode collapse on out-of-distribution prefixes.
   **Mitigation:** include the v3 prefix-suite (C2hint, C2empty, C3p) as
   eval-only sentinels; abort if any drops > 5pp from base.

## Compute cost estimate

- **GRPO training run, 1 epoch on `eren23/sfumato-prefix-robust-gsm8k`
  subset (5k rows), 1×4090 spot:** ~3 h × $0.34/h ≈ **$1.02 per λ value**.
  Four λ values = **$4.08**.
- **Eval (cmaj b=5, k=64, N=200, four adapters):** ~1 h × $0.34/h ≈
  **$0.34 per adapter** = $1.36.
- **Total estimated full experiment: ~$5.50.** Above the $5 spike cap, so
  the *graduating experiment* (Phase 2 follow-up) — not the spike.
- **Spike (defined separately, see Pre-registration):** $2.

## Dependencies

- None on Workstream C (spike does not need step-callbacks).
- Reuses `e4/runner.py` cmaj path (`CONDITION=cmaj`), `scripts/branch_agreement.py`
  for diversity score computation, `scripts/binom_ci.py` for whiskers.
- Graduating experiment needs the v3 LoRA target-module list documented in
  `e2/RESULTS_TRACK1.md` (`["q_proj","k_proj","v_proj","attn_out","ff_proj","up_proj","ff_out"]`)
  to avoid re-tripping the v1/v2 capacity bug.

## Literature evidence

| Paper | URL | One-line takeaway |
|---|---|---|
| Nie et al. 2502.09992 (LLaDA) | https://huggingface.co/papers/2502.09992 | Base model; SFT on 4.5M pairs, no diversity term. |
| Zhao et al. 2504.12216 (d1) | https://huggingface.co/papers/2504.12216 | diffu-GRPO baseline; correctness reward only, no inter-branch term. GitHub: dllm-reasoning/d1, 440 stars. |
| Zhu et al. 2510.11052 (LRD) | https://huggingface.co/papers/2510.11052 | Belief-state refinement to fix DDLM "premature commitment"; orthogonal mechanism, not RL. |
| Berrayana et al. 2510.15244 | https://huggingface.co/papers/2510.15244 | Hybrid AR+DDLM; their +27pp on DART-5 is *ensemble* benefit, not diversity-tuned. |
| GIFT (2509.20863) | https://arxiv.org/abs/2509.20863 | Per-token entropy importance weighting in SFT; per-trajectory not per-ensemble. |
| AGRPO / GDPO (2510.08554) | https://arxiv.org/abs/2510.08554 | RL for DDLM beating diffu-GRPO; reward is correctness, no diversity term. |
| CDLM (2511.19269) | https://arxiv.org/abs/2511.19269 | Consistency between adjacent denoising states — opposite direction (less diversity). |
| Conditional [MASK] (2411.06438) | https://aclanthology.org/2025.emnlp-main.450/ | Diffusion-EAGS entropy-adaptive Gibbs; adjacent direction but for sampling, not training. |
| Self-Speculative (2510.04147) | https://huggingface.co/papers/2510.04147 | Self-drafting for speedup, leveraging DDLM parallel-prediction; not diversity-related. |

Cross-reference count: **9 papers**, all from 2025-2026, none directly
implementing an inter-branch disagreement reward term in DDLM training.

## Pre-registration (spike)

**Spike name:** `temperature-diversity-falsifier`

**Hypothesis under test:** Branch diversity, as a function of sampling
temperature $\tau$, is a causally meaningful predictor of cmaj b=5 accuracy
on GSM8K-dev. Specifically: there exists a $\tau$ where mean branch-agreement
rate $\bar p_{\text{maj}}$ is meaningfully lower than at $\tau=0.7$ (Phase-1
default) **without** dropping per-branch accuracy by more than the diversity
gain on cmaj.

**Procedure:** Run the existing v3 adapter (`eren23/sfumato-prefix-robust-gsm8k-v3`)
on **N=20 GSM8K-dev problems** at four temperatures $\tau \in \{0.5, 0.7, 1.0, 1.3\}$,
$b=5$ branches each, $k=64$ steps. Use `CONDITION=cmaj`. Measure:
1. Per-branch single-shot accuracy $\bar a_1(\tau)$
2. Cmaj accuracy $a_b(\tau)$
3. Mean branch-agreement rate $\bar p_{\text{maj}}(\tau)$
4. Fraction of problems where the right answer appears in $\geq 1$ branch
   (oracle ceiling)

**Win criterion:** at some $\tau \neq 0.7$, $a_b(\tau) - a_b(0.7) \geq +1.5pp$
**AND** $\bar p_{\text{maj}}(\tau) - \bar p_{\text{maj}}(0.7) \leq -0.05$.
This would prove that diversity-via-temperature is a free lever, justifying
the more expensive diversity-reward training run.

**Loss criterion (kill the proposal):** if $a_b(\tau)$ is monotone-decreasing
in $\tau$ across $\{0.5, 0.7, 1.0, 1.3\}$ AND the oracle ceiling is also
flat or decreasing — meaning extra diversity goes to *wrong* modes — the
diversity-as-objective hypothesis is falsified at the cheapest possible cost.
We would publish this as a negative result and not graduate the proposal.

**Inconclusive criterion:** N=20 CIs (Clopper-Pearson) overlap on $a_b$
across all $\tau$. Then we'd note insufficient power and recommend the spike
graduate to N=200 only if D3 still ranks #1 in `RANKING.md`.

**Cost cap:** $2.00 (well under the $5 spike budget).
**Pre-registration commit timestamp:** 2026-05-02 ~14:50 UTC (this file).
