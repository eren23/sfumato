# Proposal D2 — Latent-Coupled Joint LoRA Training (Berrayana Extension)

**Author:** Workstream B subagent | **Date:** 2026-05-02 | **Status:** proposal (no spike)

## Motivation

Berrayana et al. (`2510.15244`, "Planner and Executor") report that shifting
DDLM→ARM communication from text-space to latent-space yields massive gains:
**27.0% → 54.0% on DART-5** and **0.0% → 14.0% on AIME24**, with a learned
projector mapping DDLM latents into the ARM embedding space — *with both
backbones frozen*. Their Section-5 ablation acknowledges that they "do not
co-train the backbones with the projector" and explicitly flag this as
future work.

If +27pp is the *floor* with frozen backbones and a projector-only training
budget, the open question is: **does jointly LoRA-adapting the AR side, the
DDLM side, and the projector reduce the capacity demand on the projector
itself?** A smaller, lower-rank projector that hits comparable accuracy would
imply the AR/DDLM latent spaces are closer than the frozen-backbone variant
suggests — a structural finding about the hybrid space. Conversely, if joint
LoRA hurts (or is no-op vs. projector-only at higher rank), the latent gap
is real and projector-only is the optimal architecture.

The "Dual-Objective Language Models" paper (ICLR 2026, `2512.14549`) shows
that one model *can* be trained on both AR and masked-diffusion objectives
simultaneously. That's the strongest independent evidence the gradient
landscape supports joint optimization.

## Mathematical formulation

Following Berrayana §3.2: a DDLM planner $P_\phi$ produces hidden states
$h_P \in \mathbb{R}^{T_P \times d_P}$; a learned projector $g_\psi:
\mathbb{R}^{d_P} \to \mathbb{R}^{d_E}$ maps these into the ARM executor
$E_\eta$'s embedding space. The ARM autoregresses the answer
$y = E_\eta(\text{prompt}, g_\psi(h_P))$.

Berrayana fix $\phi, \eta$ frozen and train only $\psi$ (projector params).
We propose:

$$
\theta = (\phi, \eta, \psi) \to (\phi + \Delta\phi_{\text{LoRA}}, \eta + \Delta\eta_{\text{LoRA}}, \psi)
$$

with rank-$r_P$ LoRA on the DDLM ($\phi$), rank-$r_E$ LoRA on the AR ($\eta$),
plus the full projector $\psi$. Joint loss:

$$
\mathcal{L}(\theta) = -\log p_{E_{\eta + \Delta\eta}}(y^\star \mid \text{prompt}, g_\psi(P_{\phi + \Delta\phi}(x))) + \beta \cdot \|\Delta\phi\|_F^2 + \beta \cdot \|\Delta\eta\|_F^2
$$

Train on the same DART-5 + AIME24 + GSM8K-train mix Berrayana use. Ablation
plane: $(r_P, r_E, \dim(\psi)) \in \{(0,0,256), (8,8,128), (8,8,64), (16,16,32)\}$.
$(0,0,256)$ is the Berrayana baseline; the rest test "smaller projector
+ tiny LoRA on each side" against it.

## Quantitative success criteria

- **Primary:** at least one $(r_P, r_E, \dim(\psi))$ combination with
  $\dim(\psi) \leq 128$ matches Berrayana's frozen-backbone DART-5 = 54.0%
  within 95% CIs (i.e. ≥ 50.0% absolute), proving capacity is partly
  transferable from projector to backbone-LoRA.
- **Secondary:** $(8, 8, 64)$ beats $(0, 0, 256)$ on DART-5 by ≥ +3pp at
  matched compute (joint training time vs. projector-only training time both
  capped to 8 GPU-hours). Failure here = "the latent gap really is
  projector-shaped, joint training adds nothing."
- **Stretch:** AIME24 ≥ 18% (Berrayana's 14% + 4pp), the headline number
  worth a paper.

## Predictable failure modes

1. **Joint training collapses the DDLM into the AR mode.** With AR's gradient
   pulling on $\phi$ via the projector, the DDLM may stop generating useful
   diverse plans. **Kill signal:** branch-agreement rate (à la Phase 1 falsifier)
   on the DDLM planner-only output drops below 0.6 of base LLaDA's value.
2. **Projector-rank capacity is the actual bottleneck (not backbone capacity).**
   Then joint LoRA is a waste — should reduce $\dim(\psi)$ further with a
   *bigger* projector net but no LoRAs. **Kill signal:** $(0,0,128)$ already
   matches $(0,0,256)$ within 1pp.
3. **Latent representation drift.** If $\phi$ moves, the projector $\psi$ has
   to track a moving target; could introduce instabilities documented in
   joint VAE-decoder training. **Mitigation:** EMA-stabilized $\phi$ (à la
   d1's exploration), or train $\phi$ for 0.5 epoch then freeze.
4. **Berrayana's gains don't transfer to GSM8K** (their headline is DART-5
   and AIME). GSM8K is easier and the cmaj=79% base ceiling on LLaDA-8B
   may already be close to the structural limit. **Mitigation:** evaluate
   on Berrayana's own benchmarks first; only port to GSM8K if DART-5 lifts.

## Compute cost estimate

- **Berrayana's setup uses 1×4090 + 1×A100 for projector training.** Their
  GitHub repo isn't linked from the HF paper page yet (checked 2026-05-02);
  must reproduce the projector arch from §3.2 of the PDF.
- **Projector-only baseline reproduction (1 condition):** ~6 GPU-hours
  on 1×A100 spot ≈ **$2.40**.
- **Joint LoRA × 4 ablation cells:** 4 × 8 GPU-hours × $0.40/h ≈ **$12.80**.
- **Eval on DART-5 + AIME24 + GSM8K cmaj b=5 (5 adapters × 3 benchmarks):**
  ~5 × 1.5 h × $0.40/h ≈ **$3.00**.
- **Total estimated full experiment: ~$18.20.** Eats most of the Phase 2
  budget. **Way over $5 spike cap** — this stays plan-only, executes only
  if RANKING.md picks it as the graduating direction and budget allows.

## Dependencies

- **External:** Berrayana code (request from authors via HF discussion at
  `huggingface.co/papers/2510.15244` if not on GitHub by go-time).
- **Internal:** none on Workstreams A or C.
- **Blocking risk:** the LCH JVP shim issue (`scripts/lch_feasibility.py`,
  PEFT `lora_magnitude_vector` ModuleDict) would also block any centroid-based
  diagnostic of the projector — but we don't need centroids for D2 itself.
  Flag it for the overflow queue, not D2 critical path.

## Literature evidence

| Paper | URL | One-line takeaway |
|---|---|---|
| Berrayana et al. 2510.15244 | https://huggingface.co/papers/2510.15244 | Frozen-backbone projector hits +27pp DART-5; explicitly flags joint training as future work. |
| Nie et al. 2502.09992 (LLaDA) | https://huggingface.co/papers/2502.09992 | Backbone DDLM; base model for our $\phi$. |
| Zhu et al. 2510.11052 (LRD) | https://huggingface.co/papers/2510.11052 | Two-stage latent refinement for DDLMs; orthogonal but shows DDLM latents have exploitable structure. |
| Kang et al. 2510.04573 (LaDiR) | https://huggingface.co/papers/2510.04573 | VAE+latent diffusion for AR; *symmetric* to Berrayana's setup (AR-side latent), useful negative control. |
| "Think First, Diffuse Fast" (2603.13243) | https://arxiv.org/html/2603.13243 | Plan-conditioning DDLM with AR plans; +11.6pp on GSM8K. Closest published cousin to D2 in text-space. |
| Dual-Objective LMs (2512.14549) | https://arxiv.org/abs/2512.14549 | One model trained on both AR and masked-diff objectives — proves joint gradient landscape exists. |
| d1 (2504.12216) | https://huggingface.co/papers/2504.12216 | LoRA + GRPO on LLaDA works without instability; SFT-then-RL recipe transfers. |
| LAD: LoRA-Adapted Diffusion (EMNLP 2025) | https://aclanthology.org/2025.emnlp-demos.8.pdf | LoRA on diffusion models is well-tooled; precedent for $\Delta\phi_{\text{LoRA}}$. |

Cross-reference count: **8 papers**. None co-train DDLM-LoRA + AR-LoRA +
projector jointly; D2 is novel in that exact configuration.
