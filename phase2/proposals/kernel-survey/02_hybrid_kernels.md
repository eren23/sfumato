# 02 — Hybrid / Fused GPU Kernels for AR ↔ Diffusion Systems

*Survey for the sfumato repo. No code changes. Read together with `/Users/eren/Documents/AI/sfumato/PLAN.md`, `/Users/eren/Documents/AI/sfumato/e4/runner.py`, `/Users/eren/Documents/AI/sfumato/e4/diff_llada.py`, `/Users/eren/Documents/AI/sfumato/e4/ar_qwen.py`. Cited material is dated through 2026-05.*

## 0. What sfumato actually does at the kernel level

Confirmed from the e4/ source:

- **AR backbone — `e4/ar_qwen.py`.** Qwen2.5-{0.5B,1.5B}-Instruct via `transformers.AutoModelForCausalLM.generate(...)`. Strict causal attention, standard HF KV cache. Used for plan (≤32 tokens), finalize, extend (`extend_cot`), and merge over diffusion candidates (`cmerge`). No custom kernels; whatever HF + flash-attn-2 ships is what runs.
- **Diffusion module — `e4/diff_llada.py`.** LLaDA-8B-Instruct loaded via `AutoModel.from_pretrained(..., trust_remote_code=True)`. The reference modeling code calls a **dense bidirectional self-attention** every diffusion round on the *full* `prompt + gen_length` sequence (`x.shape == (1, L_prompt + 128)`). No KV cache: every one of the `steps` denoising rounds (default 32–128) re-runs the entire prefill. Sub-block schedule = 4 sub-blocks × 32 tokens, `steps_per_block = steps // 4`.
- **Hybrid glue.** `runner.run_condition` orchestrates `c1 / c2 / c2c / c3 / c3p / c4 / cmaj / cmajc / cmerge / crev`. Each Qwen ↔ LLaDA hop is a **separate forward graph** with **independent KV state** (Qwen has cache, LLaDA does not). The plan tokens are textually concatenated into LLaDA's prompt — they go through LLaDA's prefill again, *not* via cache transfer.
- **Mask-shape inventory.** AR side: causal, dense. LLaDA side: full bidirectional over `prompt | gen` (LLaDA's published modeling actually masks nothing inside the gen window; cross-block ordering is enforced sampler-side by which positions are remasked to `[MASK]=126336`). `cmaj`/`cmajc`/`cmerge` are independent forwards (no cross-branch attention). The semi-AR schedule is a *sampler* construct, not a kernel one.

So sfumato today is paying full price for: (a) LLaDA prefill at every diffusion step (no `dKV-Cache` / `Fast-dLLM` reuse), (b) re-prefilling the AR plan as text into LLaDA, (c) running each `cmaj` branch as a separate batch=1 forward instead of batched. The kernels below are the published prior art that closes those gaps.

---

## 1. Hybrid-attention-mask kernels (table)

Each row: paper / repo, what is fused, claimed speedup, sfumato applicability per condition.

| # | Kernel / system | Paper / repo | Mask shape supported | Reported speedup | Applicability to sfumato |
|---|---|---|---|---|---|
| 1 | **FlexAttention** | PyTorch 2.5 blog [1], `torch.nn.attention.flex_attention` [2] | Arbitrary `mask_mod` + `score_mod` lowered to a Triton FlashAttention kernel via `torch.compile`. Block-causal, prefix-LM, intra-block-bidi all expressible in ≤10 lines. | "performance competitive with handwritten" FlashAttention; ≈15 % over a hand-rolled custom mask in dFactory [12] | **Direct fit.** sfumato's "AR-prefix causal then bidi-window over gen" is exactly the BD3-LM mask pattern that FlexAttention demos in the dFactory tutorial. Replaces LLaDA's dense full-attention with a sparser kernel that ignores the masked-out diffusion-future. |
| 2 | **FlashAttention-2 (custom-mask)** | `2307.08691` Dao 2023 [3] | Causal + ALiBi + sliding-window. **Does NOT support per-batch arbitrary block-diffusion masks.** | 2× over FA1 on H100/A100 | Already used implicitly by Qwen via HF. Not directly usable for the LLaDA bidi-then-causal hybrid mask without forking the kernel. |
| 3 | **FlashAttention-3 / Hopper** | `2407.08608` Shah 2024 [4] | Same set as FA2 + WGMMA + warp-spec. Custom masks still require kernel edits. | 1.5–2× over FA2 on H100 (FP16/FP8) | Sfumato runs on RTX 4090s today — FA3's Hopper-only path doesn't apply. Useful only if/when the hybrid moves to H100/B200 inference. |
| 4 | **FlashMask** | `2410.01359` Wang 2024 [5] (ICLR 2025) | "Column-wise sparse" mask family — encodes any mask whose pattern is a union of contiguous-row spans per column. Includes prefix-LM, block-diagonal, document-causal. | 1.65–3.22× end-to-end speedup vs. dense FA | Strong fit for the **`cmerge` finalizer**: AR finalizer reads `N` candidate diffusion CoTs; FlashMask's "document-causal" is exactly the right mask for that concatenation. Drop-in if PaddleNLP path is ported. |
| 5 | **Block-Sparse FlashAttention** | mit-han-lab/Block-Sparse-Attention [6] | Block-granularity sparse + token-granularity sparse + streaming. Forward+backward both shipped Oct 2024. | up to 5× over dense FA on long-context | Only useful if sfumato grows to long contexts (>4 k). At gen_length=128 the block-sparse overhead probably eats the win. Park for long-form. |
| 6 | **BD3-LM custom kernel (vectorised dual-forward)** | `2503.09573` + `kuleshov-group/bd3lms` [7] | Block-causal cross-block + bidirectional intra-block. Implementation concatenates the two views into a single attention call to amortise memory bandwidth. | "20–25 % training speed-up over running two forward passes" [7] | **Closest published precedent for sfumato's exact mask.** sfumato isn't training, but BD3-LM's inference path can be lifted directly to LLaDA if we replace LLaDA's dense forward with a BD3-LM-style block-causal attention — saves the wasted compute on already-committed sub-blocks. |
| 7 | **ACDiT Skip-Causal Attention Mask (SCAM) on FlexAttention** | `2412.07720` + thunlp/ACDiT [8] | Each noised block attends to itself + all *prior clean* blocks. Implemented in PyTorch 2.5 FlexAttention. | Not benchmarked in isolation; ACDiT trains on ImageNet at ACDiT-XL scale | **Same mask shape as sfumato's c4 / cmerge handoffs** when the AR-extend output is "clean" and the next diffusion round is "noised". Reference implementation is short — the SCAM `mask_mod` is ~5 lines. |
| 8 | **Transfusion mixed mask** | `2408.11039` [9] | Causal over text spans, full bidi *within* image spans, span boundaries given by token type. | n/a (not optimised, just correctness) | Identical shape to sfumato's `c3` (causal Qwen prefix → bidi LLaDA gen → causal Qwen finalize). Transfusion ships an HF-style implementation; the mask fn is reusable. **Not a kernel win** (no fused implementation), but it is a reference for *how to spell* the mask in a single graph if sfumato ever fuses the three legs. |
| 9 | **Show-o Omni-Attention** | `2408.12528` + showlab/Show-o [10] | Causal for text, full for image, switched per-token by modality flag. | Demonstrated, not benchmarked | Same idea as #8 but with a per-token gate already wired to a single transformer. If sfumato were ever to merge AR and diffusion into one graph, Show-o's omni-attention is the reference template. |
| 10 | **HMAR block-sparse IO-aware kernel** | `2506.04421` + NVlabs/HMAR [11] | Markovian block-sparse: scale-`s` attends only to scale-`s−1`. Custom Triton kernel. | "≥2.5× training, ≥1.75× inference, ≥3× lower memory" vs. VAR | Image-domain. Pattern is similar to "diffuse only conditioned on the immediately previous block, not the full prefix." If sfumato adopts a Markovian sub-block schedule it ports directly. |
| 11 | **dFactory block_diff_mask FlexAttention recipe** | `inclusionai.github.io/dFactory` [12] | Composes Block-Diagonal (intra-block bidi) ⊕ Offset-Block-Causal (clean→noised) ⊕ Block-Causal (clean→clean). | ~15 % over a hand-rolled mask | Most surgical win on sfumato right now: replaces LLaDA's `model(x)` dense call with a FlexAttention call that *skips* the `O(L_prompt²)` prompt-self-attention recompute on every step. |

**Bottom line.** The sfumato hybrid mask is FlexAttention-shaped. BD3-LMs and dFactory have already published the exact recipe; the only work is wiring it into LLaDA's `trust_remote_code` modeling.

---

## 2. KV-cache hot-swap survey (≥3 entries, including null results)

The hard problem: AR Qwen has a `(num_layers, 2, B, H, T, d)` cache, LLaDA has *none*, and they share zero parameters. What's been published?

| # | System | Mechanism | Verdict for sfumato |
|---|---|---|---|
| 1 | **dKV-Cache (NeurIPS 2025)** | `2505.15781` + horseee/dKV-Cache [13] | Caches K/V for *already-decoded* tokens in a diffusion LM, conditioned on a delayed-write rule (only commit a cache entry once the token is decoded; mask-state tokens stay re-computed). 2–10× LLaDA speedup. | **Direct fit.** sfumato's `cmaj`/`cmajc` runs `BRANCHES=5` LLaDA forwards from the same prompt — dKV-Cache lets the prompt prefix be cached once and reused across branches and across diffusion steps. Single biggest free lunch on the LLaDA side. |
| 2 | **Fast-dLLM (NVIDIA, ICLR 2026)** | `2505.22618` + NVlabs/Fast-dLLM [14] | Block-wise approximate KV cache for bidirectional diffusion LLMs + confidence-aware parallel decoding. Up to **27.6× throughput** on LLaDA / Dream; **11× on GSM8K-512**. | **Highest expected speedup, lowest risk** — the LLaDA reference code is in scope, NVIDIA shipped a working repo, GSM8K is exactly sfumato's eval. Free lunch for `c2`/`c2c`/`cmaj`/`cmajc` and the LLaDA legs of `c3`/`c4`. |
| 3 | **D2F (Discrete Diffusion Forcing)** | `2508.09192` [15] | Trains LLaDA-style models with *block-causal* attention so finished blocks' K/V can be cached, while later blocks still diffuse. "Faster than AR" claimed. | Requires retraining LLaDA. Out-of-scope without compute budget. Useful as a *target* architecture if sfumato ever gets training compute (track E1 in PLAN.md). |
| 4 | **AR↔Diffusion KV bridge** | **No published kernel exists** | A fused kernel that takes Qwen's per-layer K/V tensor and projects it into LLaDA's K/V space (or vice-versa) without re-prefilling text — has not been published anywhere I can find. The two architectures don't share dims (Qwen2.5-1.5B is 28L×16H, LLaDA-8B is 32L×32H), so a direct copy is impossible; some learned projection would be needed. | **Open problem.** This is exactly the kind of "obvious-in-retrospect" missing piece that justifies a free-lunch agent. Even a *naïve* solution — embed Qwen's last-hidden-state into LLaDA's prompt via a learned linear adapter, plus dKV-Cache for the LLaDA prefix — would save the full plan re-prefill on every `c3` step. |
| 5 | **Ca²-VDM (ICML 2025)** | causal video-diffusion cache sharing [16] | Strict-causal attention so cache of conditional frames is precomputable and reused across AR steps. Video-domain. | Wrong modality. The *idea* — "force the diffusion model into a causal mask so an AR-style cache works" — is the same as D2F #3. |
| 6 | **HMAR no-cache strategy** | `2506.04421` [11] | Reformulates next-scale prediction as Markovian, *eliminating* the need for KV-cache (each scale only conditions on its immediate predecessor). | Negative-space option for sfumato: instead of *adding* a KV cache, *remove* the long-prefix dependency. Requires retraining; out of scope today. |

**Null result:** there is **no fused kernel that hot-swaps a Qwen-shape KV cache into a LLaDA-shape KV cache.** Everyone who has published a dual-mode system (Transfusion, Show-o, BAGEL, MANZANO, JanusFlow) does it by *unifying the model* into a single transformer at training time, not by bridging two pre-trained models at inference. **This is the sfumato-specific gap.**

---

## 3. Phase-aware paged attention (vLLM / SGLang state of the art)

| Stack | LLaDA-style mask diffusion support? | Hybrid AR↔diffusion support? | Notes |
|---|---|---|---|
| **vLLM** | Experimental as of April 2026 [17][18]. The `vllm-omni` fork has an open RFC [19] for "per-role attention backends" so self/cross/joint/cross-modal attention can each pick a backend; this is **prerequisite plumbing** for hybrid AR+diffusion serving but not a working solution. Custom mask issue #5228 is open since 2024 [20]. | No. | LLaDA inference still rides on `transformers` + flash-attn-2; vLLM's PagedAttention can't yet model bidirectional-then-remask schedules. |
| **SGLang** | **Day-0 LLaDA-2.0 support shipped Dec 2025** [21]. Their dLLM framework batches multiple Diffusion Blocks in a single chunked-prefill call and reuses RadixAttention's prefix tree for shared prompts. RFC `#12766` [22] discusses block-diffusion specifically; roadmap issue `#14199` [23] tracks 2025-Q4 / 2026-Q1 work. | Partial: text-AR + diffusion-image is there for unified models (Show-o, Janus-Pro). Hybrid AR-text + diffusion-text (sfumato's exact case) is **not** wired up. | RadixAttention [24][25] already gives sfumato the "shared prompt prefix across `BRANCHES=5` diffusion runs" win for free *if* sfumato is served via SGLang. **Free-lunch candidate for Agent 4/5**: serve sfumato through SGLang and let RadixAttention dedupe the question prefix across the 5 cmaj branches. |
| **xFormers `memory_efficient_attention` w/ `BlockDiagonalCausalLocalAttentionMask`** | Supports arbitrary attn_bias and a small set of block-causal masks. | No native hybrid. | Strictly weaker than FlexAttention now that the latter is in core PyTorch. |
| **DeepSpeed-Inference** | Causal only. | No. | Not relevant. |

**Explicit free-lunch claim for Agent 4/5:** **vLLM does not yet support LLaDA-style mask diffusion with custom block-causal masks. SGLang supports LLaDA but does not yet wire sfumato's specific AR-plan → diffusion → AR-finalize pipeline.** Either a (a) thin SGLang plugin that exposes `cmaj` / `c3` as composable phases or (b) an `vllm-omni` per-role backend config that points the LLaDA leg at FlexAttention is *low-effort, high-leverage*. The plumbing in vllm-omni is being built right now and will accept external contributions.

---

## 4. Multimodal cross-pollination (techniques that *would* port to sfumato but haven't been tried)

These are all "single transformer, dual loss, hybrid mask" systems whose mask + cache tricks would carry over but which sfumato has not borrowed because sfumato keeps Qwen and LLaDA as *separate* pretrained models.

| Technique | Source | Why it would help sfumato | Status today |
|---|---|---|---|
| **Omni-Attention per-token modality gate** | Show-o [10] | Replace the "two-models-bolted-together" architecture with one Qwen-sized model whose attention mask is per-token gated (causal for plan/finalize spans, bidi for CoT span). Eliminates the AR↔Diffusion KV-bridge problem entirely (gap #4 above). | Untried in sfumato. |
| **Mixture-of-Transformer-Experts (MoT) with shared self-attention** | BAGEL [26] | Two transformer experts share self-attention across the same token sequence — analogous to Qwen+LLaDA sharing KV. Long-context interaction between understanding and generation comes "for free" through the shared attention. | Untried. Closest analog of "what sfumato wants to be." |
| **Hybrid tokenizer with continuous + discrete adapters** | MANZANO [27] | Single shared encoder, two adapters (continuous-for-understanding, discrete-for-generation). For sfumato: a single encoder whose output feeds both the AR head (for plan/finalize) and the diffusion head (for CoT) — KV cache is inherently shared. | Untried; requires retraining. |
| **Causal-then-bidi mask (Transfusion)** | `2408.11039` [9] | Reference "spelling" of sfumato's `c3` mask in a single graph. Cheap to copy as a `mask_mod`. | Mask is published; sfumato hasn't lifted it. |
| **Skip-Causal Attention Mask (SCAM)** | ACDiT [8] | Each diffusion sub-block sees all *clean* prior context but attends inside itself bidirectionally. Maps 1:1 to sfumato's "after AR-extend, the extension is clean; the next diffusion sub-block is noised." | Mask published, FlexAttention-ready. |
| **JanusFlow flow-matching head sharing** | `2411.07975` | The flow-matching head and the AR head share a backbone. Suggests sfumato's commit-LoRA could be re-cast as a flow-matching head on top of Qwen, removing LLaDA entirely for the final block. | Architectural experiment, not a kernel one. |
| **Chameleon early-fusion tokenization** | `2405.09818` | All modalities flat-tokenised; one causal mask suffices. The trade-off is no bidi diffusion, only AR. Useful as the *negative control* — "what do you give up by going pure AR?" | Already implicitly tested as sfumato's `c1` baseline. |

---

## 5. Gaps — hybrid kernels that don't yet exist but obviously should

These are the ones to feed into the free-lunch agents (Agent 4 / Agent 5 from the phase 2 ranking).

1. **Cross-architecture KV bridge (Qwen→LLaDA).** A learned linear adapter that projects Qwen's per-layer K/V into LLaDA's K/V space, plus dKV-Cache for the LLaDA prefix. Saves the full plan re-prefill on every `c3` step. **No published kernel exists** (§2 row 4).
2. **Branch-batched diffusion attention.** sfumato's `cmaj`/`cmajc` runs `BRANCHES=5` LLaDA forwards as five separate batch=1 calls. A fused kernel that runs them as batch=5 with a *shared* prompt prefix (à la RadixAttention) and per-branch independent suffix would amortise the prompt prefill 5×. Implementable today as a SGLang plugin; nobody has shipped it for LLaDA specifically.
3. **Phase-switching paged attention.** A single attention backend that *changes mask shape* mid-sequence: causal for tokens 0..N (Qwen plan), bidi for tokens N..N+128 (LLaDA CoT), causal for tokens N+128..end (Qwen finalize). FlexAttention can express it; no inference engine wires it as a single graph.
4. **Sub-block-aware commit-LoRA fused kernel.** sfumato's commit-LoRA flips PEFT adapters at the sub-block boundary (`diff_llada.py:_enable_commit`). PEFT's per-Linear walk costs >0; a fused kernel that gates LoRA application by sub-block index would remove that overhead and make `commit_n_blocks ∈ {1,2,3,4}` essentially free to sweep.
5. **Mask-diffusion-aware speculative decoding.** AR speculative decoding (Medusa, Eagle) is well-studied. *Diffusion* speculative decoding — propose `k` tokens via a small LLaDA, verify via a large LLaDA in one bidi pass — is unbuilt for mask diffusion (only continuous-diffusion analogs exist).
6. **Cross-branch consensus attention for `cmaj`.** Instead of running `BRANCHES` independent forwards and majority-voting on the answer, a kernel that lets the `b`-th branch attend to the *committed* tokens of the other branches at each diffusion step. Closest analog: parallel-decoding with shared cache, but with bidi attention. Would let `cmaj` converge in fewer steps.
7. **Adaptive-budget attention scheduler.** The `K_STEPS` knob in sfumato is fixed per-condition. A kernel that exposes a per-position "halt on confidence ≥ τ" decision (cf. Fast-dLLM's confidence-aware parallel decoding [14]) and stops attending to that position from then on. Would compound with #2 across branches.
8. **Trust-remote-code-free LLaDA forward.** LLaDA's `AutoModel.from_pretrained(..., trust_remote_code=True)` blocks vLLM/SGLang adoption (both engines avoid arbitrary remote code). A FlexAttention-rewritten `LLaDAModel` in 200 LoC, feature-equivalent to the official one but using only `torch.nn.attention.flex_attention`, would unblock vLLM/SGLang serving.

---

## 6. Applicability to sfumato — per-condition table

Speedup numbers below are *upper bounds* derived from the cited literature; engineering days are calendar days for one engineer who already knows the codebase. "Free lunch" = no algorithmic change, kernel swap only.

| Condition | Hot path today | Best applicable kernel(s) | Expected speedup | Eng days | Risk |
|---|---|---|---|---|---|
| **`c1`** (pure Qwen AR) | HF generate, FA2 implicit | nothing — already optimal at this scale | 1.0× | 0 | — |
| **`c2`** (pure LLaDA) | LLaDA dense bidi, no cache, full re-prefill every step | **Fast-dLLM** [14] for KV cache + parallel decoding | up to 11× on GSM8K-512, ≈3–5× at sfumato's gen_length=128 | 2–3 | low (NVIDIA repo, drop-in for LLaDA) |
| **`c2c`** (LLaDA + commit-LoRA last-N blocks) | same as c2 + PEFT adapter flip per sub-block | Fast-dLLM **+** sub-block-aware LoRA gating (gap #4) | 3–5× from cache, plus ~5–10 % from removing PEFT walk | 3–5 | low / medium |
| **`c3`** (Qwen plan → LLaDA CoT → Qwen finalize) | Qwen forwards at full price, LLaDA prefill at full price (plan re-tokenised) | dKV-Cache [13] for LLaDA prefix reuse across diffusion steps; cross-arch KV bridge (gap #1) for plan reuse | 2–3× from dKV-Cache alone; potential 5–6× with KV bridge | 2 (dKV-Cache) / 10–15 (KV bridge) | low / high |
| **`c3p`** (c3 minus finalize) | as c3 minus the third leg | Fast-dLLM on the diffusion leg | 3–4× | 2–3 | low |
| **`c4`** (c3 + AR-extend + 2nd diffuse) | second LLaDA prefill includes the entire concatenation `q + plan + draft + extension` | dKV-Cache + ACDiT-SCAM mask via FlexAttention so 2nd diffuse skips re-attending to clean tokens | 4–5× on the second diffusion leg | 5–7 | medium |
| **`cmaj`** (5 LLaDA branches, vote) | 5× independent prefill of identical prompt | **SGLang RadixAttention** [24] (free if served via SGLang) **+** branch-batched diffusion attention (gap #2) | 4–5× from prefix sharing alone; up to 8× with branch-batching | 3 (SGLang serve) / 8–10 (branch-batch kernel) | low / medium |
| **`cmajc`** (cmaj + commit-LoRA per branch) | as cmaj + 5× PEFT adapter flips | RadixAttention + sub-block-aware LoRA gating | 4–5× | 5–7 | medium |
| **`cmerge`** (5 LLaDA branches → Qwen merge) | 5 LLaDA prefills + Qwen reads concatenated 5×CoT | RadixAttention on the diffusion side; **FlashMask** [5] document-causal mask on the Qwen finalizer | 4–5× on diffusion leg, 1.5–2× on the merge leg | 5–8 | medium |

**Aggregate.** If sfumato lifts only **Fast-dLLM + RadixAttention** (≈5–7 eng-days total, low risk, both have open-source repos), the diffusion-heavy conditions (`c2`, `c2c`, `cmaj`, `cmajc`, `cmerge`) get a 3–8× wallclock speedup with **zero algorithmic change**. That is the floor. The ceiling — full FlexAttention rewrite + cross-arch KV bridge + branch-batched cmaj — is closer to 15× but takes a month and ships novel kernels.

---

## Citations

1. PyTorch team, "FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention," PyTorch blog, Aug 2024. <https://pytorch.org/blog/flexattention/>
2. PyTorch core, `torch.nn.attention.flex_attention`. <https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html>
3. Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691.
4. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision," arXiv:2407.08608.
5. Wang et al., "FlashMask: Efficient and Rich Mask Extension of FlashAttention," arXiv:2410.01359 (ICLR 2025). <https://arxiv.org/html/2410.01359v1>
6. MIT-HAN-Lab, "Block-Sparse-Attention," GitHub repo. <https://github.com/mit-han-lab/Block-Sparse-Attention>
7. Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models," arXiv:2503.09573 (ICLR 2025 Oral); repo `kuleshov-group/bd3lms`. <https://github.com/kuleshov-group/bd3lms>
8. Hu et al., "ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer," arXiv:2412.07720; repo `thunlp/ACDiT`. <https://github.com/thunlp/ACDiT>
9. Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model," arXiv:2408.11039.
10. Xie et al., "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation," arXiv:2408.12528; repo `showlab/Show-o`. <https://github.com/showlab/Show-o>
11. Kumbong et al., "HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation," arXiv:2506.04421 (CVPR 2025); repo `NVlabs/HMAR`. <https://github.com/NVlabs/HMAR>
12. dFactory documentation, "Block Diffusion" tutorial with FlexAttention `block_diff_mask`. <https://inclusionai.github.io/dFactory/algo/block_diffusion.html>
13. Ma et al., "dKV-Cache: The Cache for Diffusion Language Models," arXiv:2505.15781 (NeurIPS 2025); repo `horseee/dKV-Cache`. <https://github.com/horseee/dkv-cache>
14. Wu et al., "Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding," arXiv:2505.22618 (ICLR 2026); repo `NVlabs/Fast-dLLM`. <https://github.com/NVlabs/Fast-dLLM>
15. Yang et al., "Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing (D2F)," arXiv:2508.09192.
16. Chen et al., "Ca²-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing," ICML 2025. <https://icml.cc/virtual/2025/poster/44902>
17. vLLM-Omni docs, "Adding a Diffusion Model." <https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_diffusion_model/>
18. Spheron Blog, "Deploy Diffusion Language Models on GPU Cloud," 2026. <https://www.spheron.network/blog/deploy-diffusion-language-models-dllm-gpu-cloud-2026/>
19. vLLM-Omni RFC #2632, "Per-Role Attention Backend Configuration for Diffusion Models." <https://github.com/vllm-project/vllm-omni/issues/2632>
20. vLLM issue #5228, "Custom attention masks." <https://github.com/vllm-project/vllm/issues/5228>
21. LMSYS blog, "Power Up Diffusion LLMs: Day-0 Support for LLaDA 2.0," Dec 2025. <https://www.lmsys.org/blog/2025-12-19-diffusion-llm/>
22. SGLang issue #12766, "RFC: Block Diffusion Large Language Model (dLLM) Framework In SGLang." <https://github.com/sgl-project/sglang/issues/12766>
23. SGLang issue #14199, "Roadmap: Diffusion LLMs (2025 Q4 & 2026 Q1)." <https://github.com/sgl-project/sglang/issues/14199>
24. LMSYS blog, "Fast and Expressive LLM Inference with RadixAttention and SGLang," Jan 2024. <https://www.lmsys.org/blog/2024-01-17-sglang/>
25. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," arXiv:2312.07104.
26. ByteDance Seed, "BAGEL: Emerging Properties in Unified Multimodal Pretraining," arXiv:2505.14683; repo `ByteDance-Seed/Bagel`. <https://github.com/ByteDance-Seed/Bagel>
27. Apple, "MANZANO: A Simple and Scalable Unified Multimodal Model with a Hybrid Vision Tokenizer," arXiv:2509.16197.
28. Liu et al., "MarDini: Masked Autoregressive Diffusion for Video Generation at Scale," arXiv:2410.20280.
29. Chen et al., "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling," arXiv:2501.17811; repo `deepseek-ai/Janus`.
30. Wu et al., "Janus / JanusFlow," arXiv:2410.13848 / 2411.07975.
