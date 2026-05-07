# R1: Prior-Art Validation — Schedule-Aware RL Fine-Tuning of Diffusion-LM Adapters

## TL;DR

**Direction A is substantially novel.** No prior work trains adapters via RL conditioned on *discovered* inference-time denoising schedules, nor combines schedule-conditional rewards with phase-embedding adapter synthesis. The closest work (Learning Unmasking Policies, 2512.09106) trains *sampling strategies* end-to-end with RL but treats the schedule as a passive input, not a learned signal discovered from performance data. VRPO (2505.19223) solves only variance reduction in DPO, not schedule conditioning. The core sfumato framing—discovering inverted-U masking patterns at inference, then training an adapter to *exploit* that schedule via phase_emb[t]—appears unmatched in the literature.

---

## Closest Prior Work (Top 5)

| Paper / ID | Year | What They Did | How Direction A Differs |
|---|---|---|---|
| **Learning Unmasking Policies** (2512.09106) | 2025 | RL-trained single-layer transformer policy maps token confidences → unmasking logits. Lightweight GRPO on sampling decisions. | Learns *optimal sampling strategy* end-to-end; schedule is implicit. Does NOT train full LoRA adapter on base model. No phase_emb synthesis. No schedule conditioning of rewards. |
| **LLaDA 1.5 / VRPO** (2505.19223) | 2025 | Variance-reduced DPO on full LLaDA-8B. ELBO-based preference optimization. | Full model fine-tuning, not adapter-based. No schedule conditioning. Reward is binary preference, not trajectory-conditioned. Does not discover or exploit inference schedule patterns. |
| **dTRPO** (2603.18806) | 2026 | Trajectory-reduced DPO for dLLMs. Two-stage offline preference optimization. Updates only MLP + output projection (parameter-efficient). | Offline DPO framework, not RL with schedule discovery. No learned schedule-conditional rewards. Does not use phase_emb or timestep-conditional synthesis. |
| **TC-LoRA** (2510.09561) | 2025 | Hypernetwork generates LoRA weights conditioned on (timestep, control signal) for vision diffusion *image generation*. | Timestep-conditional but for *generative control*, not language. Does not train on *discovered* schedules from performance. No phase_emb synthesis. Vision-only. |
| **TimeStep Master** (2503.07416) | 2025 | Mixture-of-experts LoRA: different LoRA experts tuned at different noise levels (fostering stage). Assembles experts via core-context collaboration. | Does not combine discovered schedules with RL rewards. Vision-only. No phase_emb. Schedule is predetermined (noise levels), not learned from inference patterns. |

---

## Direct Hits or Near-Misses

### High-Alert: 2512.09106 (Learning Unmasking Policies)
This is the **closest threat to novelty**. It:
- Trains a lightweight policy with RL (GRPO) over diffusion sampling
- Explicitly conditions on timestep `t ∈ [T]`
- Learns adaptive unmasking strategies (scheduling problem)
- Evaluated on GSM8K (same test set as Direction A)

**Critical difference:** The policy is a *side-car* sampling controller (< 0.01% params) that learns *what to unmask*, not a LoRA adapter (*14M params*) on the base model trained on actual performance rewards. The schedule is an input to the policy, not a signal discovered and *folded back into* training supervision. The learned behaviors are about *which tokens to reveal*, not *how to adapt the base model's denoising process itself*.

### Moderate: dTRPO (2603.18806)
Parameter-efficient fine-tuning of dLLMs via offline DPO. Updates MLP + projection layers selectively. **No schedule conditioning of rewards.** Treats denoising as a *trajectory* but optimizes *preference pairs* (y+ vs y-), not schedule-aware trajectory decomposition.

### Moderate: TC-LoRA (2510.09561) & TimeStep Master (2503.07416)
Both use timestep-modulated LoRA synthesis. Neither:
- Discovers schedules from *inference performance*
- Uses RL with schedule-conditional rewards
- Targets language models (vision-only)
- Implements phase_emb additive synthesis on LoRA-A

---

## Direct Evidence of Novelty

**Schedule-discovery-to-training loop appears absent from literature:**
- Searched arXiv 2024–2026 for: "learned schedule" + "adapter" + "RL" / "diffusion LM"
- Found: inference-time schedule discovery (e.g., TPDM, DNO), full-model DPO (VRPO, dTRPO), policy learning for sampling (Unmasking Policies)
- **Not found:** A system that *discovers* a schedule at inference (e.g., inverted-U masking pattern), then trains a learned adapter to be conditioned on that schedule during RL fine-tuning

**Phase_emb[t] additive bias on LoRA-A is novel architecture:**
- TC-LoRA does hypernetwork-based timestep conditioning, but for image generation and full LoRA synthesis
- No prior work synthesizes an *additive phase embedding* biased into LoRA-A specifically

---

## Strongest Novelty Claim (for Direction A)

> **Schedule-Conditioned Adapter Alignment:** We are the first to couple *inference-time discovery of optimal denoising schedules* with *RL fine-tuning of adapters conditioned on those schedules*, using phase embeddings to implement per-timestep sub-block LoRA activation, yielding a principled bridge between observational (K2: inverted-U masking phases) and prescriptive (Direction A: training to exploit that structure) diffusion-LM post-training.

---

## Risks / Things That Could Moot Direction A

1. **If VRPO already dominates math reasoning:** VRPO (2505.19223) achieves +4.7pp on GSM8K with full-model DPO. If LLaDA-1.5-base + VRPO already saturates the benchmark, Direction A's premise (schedule-aware RL on adapter) may fail to deliver gains. Check: does VRPO's full-model DPO subsume the marginal benefit of schedule-aware training?

2. **If Unmasking Policies' sampling strategy outpaces adapter-level gains:** If the lightweight GRPO policy (2512.09106) achieves near-optimal masking decisions, training a 14M adapter may add redundancy rather than complementary signal. Risk: adapter learns to *mimic* good sampling, not to *enable* it structurally.

3. **If schedule discovery doesn't generalize across domains:** The inverted-U pattern (K2) was discovered on GSM8K. If sub-block commit-toggle is artifact of GSM8K reasoning, not MATH or AIME, then Direction A's generalization claim breaks. Pre-reg mitigates this: explicit MATH-500 / AIME targets.

---

## Recommended §3.5 Framing (If Direction A WINs)

> We introduce **schedule-conditioned RL fine-tuning** as a new post-training paradigm for masked diffusion language models. Unlike prior work that either discovers optimal schedules at inference time (Unmasking Policies) or aligns full models via preference optimization (VRPO/dTRPO), we decouple these phases: first, we observe that efficient denoising exhibits phase-specific masking patterns (K2 finding); second, we train a compact FFN-LoRA adapter whose activation is modulated by a learned phase embedding conditioned on the discovered schedule, jointly optimized under RL rewards that are themselves schedule-aware (binary correctness, KL regularization to base). This yields a principled, parameter-efficient alternative to full-model alignment and a new axis of control for diffusion-LM post-training: the *timing and structure* of adaptation, not just its magnitude.

---

## Appendix: Search Scope

- **VRPO (2505.19223):** Full paper + HTML. Confirmed: full-model DPO, no adapter, no schedule conditioning.
- **dTRPO (2603.18806):** Full paper + HTML. Confirmed: selective layer updates, offline DPO, no schedule-conditioned rewards.
- **Learning Unmasking Policies (2512.09106):** Full paper + HTML. Confirmed: RL on sampling policy, timestep input, no base-model adapter training.
- **TC-LoRA (2510.09561):** HTML. Confirmed: hypernetwork-based LoRA synthesis for *image generation* control, not language models.
- **TimeStep Master (2503.07416):** Full paper + HTML. Confirmed: mixture-of-expert LoRA at noise levels, vision-only, no discovered schedule.
- **d1 (2504.12216):** PDF. Confirms: full-model + SFT path; no schedule-aware adapter framing.
- **Inference-time schedule discovery (TPDM, DNO, etc.):** Multiple papers. None train adapters to exploit discovered schedules.
- **dTRPO, VRPO, Mask-GRPO:** All reviewed; none couple schedule discovery with adapter RL.

**Conclusion:** Zero false positives. No prior art found that matches the sfumato Direction A framing.

