# R3: Cross-Substrate Primitive Transfer in Diffusion-LMs

**Topic:** Phase-4 T2.C BD3-LMs cross-substrate generality validation.
**Date:** 2026-05-07. **Author:** Phase-4 research-validation subagent R3.

## TL;DR

Cross-DLM-family transfer of **inference-time primitives** (K-step
schedules, sampling heuristics, confidence thresholds) is **largely
unexplored** — no published papers test the same primitive on multiple
distinct DLM families. Sfumato's K2 schedule-toggle transfer claim
(LLaDA-8B-Instruct → BD3-LMs on MATH-500) is **novel**, though the
field bar for "general mask-DLM primitive" requires ≥3 families. With
T2.C alone you have a strong incremental claim; with one cheaper
3rd family added (Dream-7B or DiffuLLaMA-1B) you defend a generality
narrative.

## Cross-family primitive transfer table (top 5)

| Paper | ArXiv / Year | Primitive | Families tested | Cross-family replication |
|---|---|---|---|---|
| Sitan Chen et al. — Optimal Inference Schedules for Masked Diffusion | 2511.04647 / 2025 | Optimal unmasking schedule (cosine, log-linear) | Theory; single instantiation | Theory only — no empirical cross-family validation |
| DiffuLLaMA — HKU-NLP | 2410.17891 / ICLR-25 | AR→DLM adaptation; zero-shot CoT transfer | LLaMA-2 (127M-7B converted); LLaDA implicit | Qualitative parity; no controlled K-sweep |
| BD3-LMs (Arriola et al.) | 2503.09573 / ICLR-25 | Block-wise diffusion decoding; varying block sizes 4/8/16 | BD3-LMs only | Within-family block-size; no cross-family transfer |
| Fast-dLLM | 2512.02892 / 2025 | Progress-aware confidence schedules | Diffusion LMs (limited family count) | Within-family; no cross-family replication study |
| Cross-LoRA | 2508.05232 / 2025 | LoRA transfer via subspace alignment | Heterogeneous AR LLMs | ✓ Demonstrated; **explicitly degrades with architectural distance** |

**Key finding:** Only Cross-LoRA attempts adapter cross-family transfer
— and reports degradation with architectural distance. This is an
omen: sfumato already saw it on LLaDA-8B-Instruct → LLaDA-1.5 (commit
`3e034f5`, Direction C LOSS −2pp).

## Concrete bar for §3 generality claim

**Field standard (ICLR/NeurIPS 2026):**
- 1 family + 2 domains: incremental ("Works on LLaDA across GSM8K and MATH-500")
- 2 families + 1 domain: novel but underpowered for "general" claim
- **3+ families + ≥2 domains + matching difficulty:** defensible "general law of mask diffusion"

**Sfumato's current footprint:**
- LLaDA-8B-Instruct (committed): K2 inverted-U; MATH-500 cmajc-k3 N=200 paired = 0.41 vs c2c-k0 = 0.325 = +8.5pp lift
- BD3-LMs (T2.C proposed): re-run K-sweep, target cmajc ≥0.30 + visible inverted-U in k∈{2,3}
- **With T2.C alone**: bimodal "works on 2/2 families tested" → strong conference paper claim, not "general law"

## Cost-of-evidence ranking (cheapest → most expensive new substrate)

| Target family | Base | Est. compute | Wall-clock | Key risk |
|---|---|---:|---:|---|
| **DiffuLLaMA-1B** | GPT-2 1B (AR→DLM) | ~$15-25 | 3-7 days | Smaller model; K-sweep may saturate at lower k or no inverted-U |
| **Dream-7B** | Qwen2.5 7B (diffusion-adapted) | ~$30-50 | 7-14 days | May already be fine-tuned; LoRA compat unproven |
| **BD3-LMs (T2.C)** | 400M-1B (block 4 default) | ~$50-80 | 14-21 days | **OWT pretrain → MATH SFT bottleneck**; sub-block semantics may differ post-SFT |
| MMaDA (multimodal) | Proprietary / unclear release | Unknown | Unknown | Licensing / repro risk; not recommended |

## Risks / things that could moot T2.C generality claim

1. **Publication race**: SEDD-v2 / concurrent preprint may publish "schedule-aware inference works on 2+ families" before sfumato ships T2.C. Likely window: Jun-Aug 2026.
2. **Mechanistic divergence post-SFT**: BD3-LMs after MATH SFT may have different sub-block dependencies (peak shifts to k=1 or k=4 or vanishes). T2.C result becomes confounded with SFT recipe rather than primitive transfer.
3. **Architecture-specific saturation**: Dream-7B / DiffuLLaMA may show flat or monotonic K-curves (no inverted-U) — would suggest the K2 phenomenon is LLaDA-specific (training objective, masking schedule, tokenizer-related).
4. **LoRA transfer degradation**: Cross-LoRA's finding that adapters degrade with arch distance — sfumato already saw LLaDA-8B-Instruct → LLaDA-1.5 degrade (−2pp). BD3 architectural distance is much greater.
5. **Inference-cost vs gain wash**: even if K2 replicates on BD3, wall-clock speed may not beat AR LLMs on MATH (which favor sequential reasoning). Practical impact uncertain.

## Recommended next step

**Option A (high-bar, ~$50-80, 3-4 weeks):** T2.C BD3-LMs full sweep. If WIN, immediately parallelize Dream-7B as 3rd-family secondary substrate. If BD3 fails, abort and pivot to DiffuLLaMA-1B sanity check.

**Option B (low-bar, ~$25-35, 2 weeks):** Skip BD3, directly test DiffuLLaMA-1B + Dream-7B in parallel. Both inverted-U → 3-family claim. Either fails → weaker but still publishable. **~Half the cost of T2.C alone.**

**Option C (minimum, ~$10-15, 1 week):** DiffuLLaMA-1B only. Suffices for "2/2 families sampled" but limits narrative scope. 2026 venues likely expect ≥3 families for strong generality.

**Recommended:** **Option A** if budget permits — BD3 success unlocks the strongest published narrative. **Option B** if budget tight — DiffuLLaMA + Dream gives 3-family claim at lower cost. **Defer T2.C** if Direction A WINs first (Direction A's framing strengthens to "schedule-aware RL on a single family that beats peer-class" without needing cross-substrate).

## Recommended §3.5 framing if T2.C WINs

> *"We demonstrate the K2 schedule-toggle finding generalizes across
> diffusion-LM families: the same +8.5pp cross-domain lift holds when
> we train a fresh 14M FFN-LoRA on top of BD3-LMs and re-run the
> K-sweep. The schedule-toggle is a property of mask-diffusion
> generation, not LLaDA-specific calibration. To our knowledge this
> is the first published cross-DLM-family transfer of an
> inference-time primitive."*

## References

- BD3-LMs: arXiv 2503.09573 (ICLR-25)
- Optimal Schedules: arXiv 2511.04647
- LLaDA: arXiv 2502.09992
- DiffuLLaMA: arXiv 2410.17891 (ICLR-25)
- Cross-LoRA: arXiv 2508.05232
- Fast-dLLM: arXiv 2512.02892

## 100-word verdict

Cross-substrate K2 transfer is **novel as framed** — no prior work tests the same inference-time schedule discovery on multiple DLM families with the same K-sweep protocol. Sfumato's existing LLaDA-8B-Instruct → LLaDA-1.5 LOSS (Direction C) signals that adapter transfer across architectural variants degrades; cross-FAMILY transfer (LLaDA → BD3) is a much harder bar but a much stronger claim. Recommend Option B (DiffuLLaMA-1B + Dream-7B in parallel, ~$25-35) over Option A (BD3 alone, ~$50-80) for cost/coverage tradeoff. Defer entirely if Direction A produces a §3.5 marquee result on LLaDA alone.
