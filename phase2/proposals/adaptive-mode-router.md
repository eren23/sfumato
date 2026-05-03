# Proposal D1 — Adaptive Mode Router (E1) with Bandit + Branch-Agreement Reward

**Author:** Workstream B subagent | **Date:** 2026-05-02 | **Status:** proposal (no spike — needs WS-C trace data first)

## Motivation

Phase 1's three-axis decomposition (prefix-damage / planner-trust / sampling-
diversity) and Berrayana's hybrid architecture both demonstrate that
**different reasoning steps benefit from different mechanisms** — but no
existing system *learns the per-step routing decision*. Berrayana use a fixed
"plan = DDLM, execute = AR" policy. Self-Speculative Decoding (`2510.04147`)
uses fixed verification trees. Expert-Choice Routing in DLMs (`2604.01622`)
routes *experts within a single DDLM*, not across mode types.

The structural-separation finding from Phase 1 (commit-LoRA v2 vs v3, 4/7 vs
7/7 modules) showed adapters can coexist without interference; the open
question is the *routing policy*. We propose a small bandit policy that, at
each LLaDA sub-block boundary, decides among:

1. `continue_llada` — keep diffusing the current sub-block
2. `switch_to_ar(n)` — hand off the next $n$ tokens to the Qwen AR-extender
3. `branch_cmaj(b)` — fork into $b$ parallel branches and majority-vote
4. `commit` — finalize current sub-block and stop

The router's training signal is the **branch-agreement intrinsic reward**:
when a counterfactual branching at this position would have produced higher
agreement, the router should have routed to `branch_cmaj`. This is a
self-supervised bootstrap signal that needs no human labels.

The closest published work, BaRP (`2510.07429`), is a contextual bandit for
*model selection* between LLMs — same architectural pattern, different
action space. We are not aware of a published bandit router over
*generation-mode* actions inside a single hybrid model.

## Mathematical formulation

State $s_t$ at sub-block boundary $t$ (consuming Workstream C's trace JSONL,
schema pinned in `phase2/STATUS.md` lines 51-63):

$$
s_t = (e_t, c_t, l_t, m_t)
$$

where $e_t \in \mathbb{R}^{|V|}$ is the per-token entropy histogram of the
just-committed sub-block (from `entropy[]` field), $c_t \in \{0,1\}$ is
`commit_lora_active`, $l_t \in \mathbb{R}$ is `logit_shift_norm`, $m_t \in
\{\text{llada}, \text{ar\_extend}, \text{cmaj\_branch}\}$ is the prior
sub-block's mechanism (from `mechanism` field). Featurized to
$\phi(s_t) \in \mathbb{R}^{32}$ via a tiny MLP.

Action $a_t \in \mathcal{A} = \{\text{continue}, \text{ar}(n), \text{cmaj}(b), \text{commit}\}$,
discretized to $|\mathcal{A}| = 7$ (continue; ar with $n \in \{8, 16, 32\}$;
cmaj with $b \in \{3, 5\}$; commit).

Policy $\pi_w(a \mid s) = \text{softmax}(W \phi(s))$.

**Reward** $r_t$: at the end of the trajectory, given final answer $\hat y$
and gold $y^\star$, $r_T = \mathbb{1}[\hat y = y^\star]$. Per-step reward is
the **counterfactual branch-agreement bonus**:

$$
r_t = \alpha \cdot \mathbb{1}[a_t = \text{cmaj}(b)] \cdot (p_{\text{maj}}^{(t)} - p_{\text{maj}}^{(t-1)})
$$

where $p_{\text{maj}}^{(t)}$ is the post-action majority share. Total
return $G = \sum_t r_t + r_T$, optimized by REINFORCE with a learned baseline.

Bandit-feedback caveat (à la BaRP): in deployment we observe only the
chosen action's outcome. Training uses Workstream C's offline traces where
multiple actions were tried by hand — this gives partial off-policy data
to bootstrap before any new generation cost.

## Quantitative success criteria

- **Primary:** on GSM8K-dev N=200, the routed policy achieves
  ≥ **+2.0pp accuracy** over the best fixed-mechanism baseline at
  **matched FLOPs** (measured via `e4/flops.py`). The fixed baselines are
  all-LLaDA, AR-only-Qwen-1.5B, cmaj b=5 with v3.
- **Mechanism interpretability:** the trained policy's mode-mix on hard
  problems (those where base cmaj b=5 misses) skews towards `cmaj` and
  `ar_extend`; on easy problems it stays in `continue_llada`. Quantified as
  $H(\text{mode-mix} \mid \text{difficulty-bin})$ ≥ 1.0 nats.
- **Sample efficiency:** policy converges within 2,000 traces
  (≈ 10 problems × 200 sub-blocks).

## Predictable failure modes

1. **Branch-agreement reward is too sparse to bootstrap.** Most sub-blocks
   on a 5-block trajectory don't change the final answer; the credit-
   assignment signal is weak. **Kill signal:** baseline-subtracted policy
   gradient variance > 4× the mean over first 1,000 steps.
2. **The router learns "always cmaj"** because cmaj uniformly improves over
   single-shot. **Mitigation:** include a FLOPs penalty term
   $-\beta \cdot \text{flops}(a_t)$ in the reward.
3. **Off-policy bias from Workstream C's manual traces.** The human user's
   action choices aren't $\epsilon$-greedy; they're highly biased toward
   "interesting" demonstrations. **Mitigation:** use importance-sampling
   correction with a uniform-action prior, or use the manual traces only
   for the value-function pretraining, with on-policy collection for the
   policy net.
4. **Mode-collapse on novel problem distributions.** The router trained on
   GSM8K-dev may simply over-fit the dev set's structural cues. **Mitigation:**
   hold out 50 problems for IID test, plus run on MATH-500 for
   distribution-shift check.

## Compute cost estimate

- **Trace harvest:** 50 problems × 5 manual sessions each ≈ 250 traces from
  Workstream C's app — **free** (uses C's existing budget).
- **Synthetic on-policy traces (rollouts during training):** 2,000 sub-block
  decisions × ~1s/decision GPU ≈ 1 GPU-h × $0.34 ≈ **$0.34**.
- **Policy training (REINFORCE on 32-dim MLP, 5,000 steps, CPU-friendly):**
  ~30 min, **negligible** GPU.
- **Final eval (N=200 GSM8K-test, with FLOPs accounting):** ~2 GPU-h ≈ **$0.68**.
- **Total estimated full experiment: ~$1.50.** Cheap! But blocked on
  Workstream C delivering trace JSONL data.

## Dependencies

- **Hard:** Workstream C's trace JSONL schema (already pinned in
  `phase2/STATUS.md` Workstream C → "Trace JSONL schema") plus at least
  one batch of real traces (3 example traces are part of C's done criteria).
  We cite the schema verbatim:
  ```jsonc
  {"step_idx":0, "sub_block":0, "mechanism":"llada|ar_extend|cmaj_branch",
   "tokens_committed":[], "entropy":[], "commit_lora_active":false,
   "logit_shift_norm":null, "manual_intervention":null}
  ```
  The featurizer $\phi(s)$ in §"Mathematical formulation" reads exactly
  these fields. If any field gets renamed during C's app build, D1 must
  re-version.
- **Soft:** `e4/diff_llada.py` `_generate()` refactor with `step_callback`
  (also Workstream C's deliverable) — needed for inference-time policy
  invocation, not for training-time trace replay.

## Literature evidence

| Paper | URL | One-line takeaway |
|---|---|---|
| BaRP (2510.07429) | https://huggingface.co/papers/2510.07429 | Contextual bandit over LLM choice; same pattern, different action space. +12.46% over offline routers. |
| Expert-Choice DLM (2604.01622) | https://huggingface.co/papers/2604.01622 | Routes *within* a DDLM at expert level; shows DLMs have routable computation. GitHub: zhangshuibai/EC-DLM. |
| BEST-Route (2506.22716) | https://arxiv.org/html/2506.22716v1 | Routes between model + sample-count; shows test-time-compute knob is learnable. |
| Berrayana et al. 2510.15244 | https://huggingface.co/papers/2510.15244 | Fixed AR/DDLM router; D1 generalizes their hand-coded policy to learned. |
| Self-Spec (2510.04147) | https://huggingface.co/papers/2510.04147 | Fixed verification policy in DDLM; same boundary layer as D1's `commit` action. |
| FailFast (2512.20573) | https://arxiv.org/html/2512.20573v1 | Dynamically chooses how many tokens to draft based on confidence — confidence-conditioned action selection, sister problem. |
| RWS d1/diffu-GRPO (2504.12216) | https://huggingface.co/papers/2504.12216 | Provides the GRPO-style optimizer pattern transferable to D1's policy gradient. |
| RLVR survey (2505.19590) | https://arxiv.org/pdf/2505.19590 | Catalog of self-consistency-as-reward; D1's branch-agreement bonus fits the RLVR taxonomy. |

Cross-reference count: **8 papers**. The novel contribution is the
*per-sub-block, mode-typed action space* + *branch-agreement intrinsic
reward*. Closest cousin (BaRP) operates over models, not modes within a
single model.
