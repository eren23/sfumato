# Diffusion-LM Kernel Survey — for sfumato `e4/diff_llada.py`

**Audience:** sfumato kernel work-stream + human PI.
**Author:** research subagent | **Date:** 2026-05-04
**Scope:** GPU kernels and inference-time compute-saving techniques specifically for
discrete / mask-diffusion LMs (LLaDA, Dream, MDLM, BD3-LM, SDAR, Diffu*, etc.).
**What this is *not*:** a re-survey of the 226-paper PLAN.md map (model-level
hybrids, latent CoT, JEPA, etc.). It is a *kernel / runtime* index focused on
ops that show up in the LLaDA semi-AR loop in `e4/diff_llada.py`.

**Sources read first:** `README.md`, `PLAN.md`, `e4/diff_llada.py` (entire file,
lines 1–641), `e4/runner.py`, `phase2/proposals/kernel-survey/00_OUR_IDEA.md`.

**Search queries actually run** (raw, for reproducibility):
1. `dKV-Cache diffusion language model 2505.15781 KV cache prefix block reuse`
2. `Fast-dLLM diffusion LLM KV cache parallel decoding 2025 speedup`
3. `Sparse-dLLM 2509.24095 sparse attention diffusion LLM compute saving`
4. `LLaDA inference optimization triton fused attention kernel`
5. `Block Diffusion BD3-LMs 2503.09573 kuleshov-group GitHub kernel attention`
6. `SDAR semi-autoregressive diffusion 2510.06303 inference speedup kernel`
7. `Focus-dLLM long-context confidence-guided diffusion LLM inference 2025`
8. `SEDD score entropy discrete diffusion 2310.16834 inference acceleration`
9. `D2F diffusion-forcing speculative decoding diffusion LLM 2025`
10. `DiffuLLaMA 2410.17891 attention diffusion adaptation inference`
11. `DFlash block diffusion flash speculative decoding 2602.06036 kernel`
12. `MDLM masked diffusion language model 2406.07524 inference repository`
13. `CDLM consistency diffusion language model 2511.19269 14x faster inference together`
14. `LLaDA-V vision language masked diffusion 2505.16933 inference`
15. `LoSA locality aware sparse attention block diffusion 2604.12056 kernel`
16. `MaskGIT 2202.04200 parallel masked generation iterative decoding kernel`
17. `Fast-dLLM v2 2509.26328 block diffusion inference efficient GitHub`
18. `Slot-dLLM SlowFast dLLM diffusion language model inference 2025 kv cache`
19. `Prophet diffusion LLM accelerator commit confidence early exit 2025`
20. `dLLM-Cache DualCache prefix suffix diffusion language model 2025`
21. `Adaptive Parallel Decoding diffusion LLM Israel UCLA NeurIPS 2025`
22. `WeDLM causal attention diffusion language model 2512.22737 KV cache`
23. `dParallel 2509.26488 diffusion LLM learnable parallel decoding`
24. `learning to parallel 2509.25188 diffusion LLM learnable parallel decoding`
25. `FlashAttention bidirectional mask diffusion language model attention kernel`
26. `vLLM SGLang LLaDA diffusion language model integration support`
27. `SlowFast Sampling 2506.10848 diffusion LLM three golden principles`
28. `Self-Speculative Decoding diffusion language model 2025 OpenReview`
29. `variable generation length EOS diffusion LLM 2510.24605 inference`
30. `FlashInfer top-k top-p sampling sorting-free GPU kernel categorical`
31. `PEFT LoRA merge unmerge fast switch adapter inference kernel`
32. `Liger-Kernel triton fused cross entropy LLM training`

---

## 0. The hot path in `e4/diff_llada.py` (one-paragraph recap)

The denoising loop lives in `_Real._generate`, lines 369–548. Per call we run
`num_blocks = gen_length / sub_block_length = 128 / 32 = 4` semi-AR sub-blocks.
Per sub-block we run `steps_per_block = steps / num_blocks = 64/4 = 16` (or
8 at `K_STEPS=32`) denoising rounds. Each round runs **a full bidirectional
forward pass over the entire prompt+gen sequence** (`logits = self._model(x)`,
line 444) — there is no KV cache. After the forward pass we add Gumbel
noise (line 446 / `_add_gumbel_noise`), argmax over the vocab (line 447),
softmax + gather to get per-position confidence (449–450), block-end masking
(452), `torch.where` to keep already-committed tokens (454–455), `torch.topk`
to pick the `k` lowest-confidence positions to commit (459–460), and a
boolean scatter back into `x` (480). For `cmaj`/`cmajc` (`runner.py`
lines 203–261) we run this entire loop `BRANCHES=5` times sequentially.

Total forward-pass count per `cmajc-v3` problem at the headline recipe
(`K_STEPS=64`, `BRANCHES=5`, `gen_length=128`, `sub_block_length=32`):
**`5 × 64 = 320` full-context forward passes**, each over a sequence of
`prompt_len + 128` tokens. With LLaDA-8B in bf16 on a 4090 this is the
single largest cost in the project (~$0.20/hr × multiple hours per N=200
sweep).

Every kernel below targets one or more of these ops.

---

## 1. Catalog (≥15 kernels)

| # | Name | Source | What it fuses / replaces | Reported speedup | Touches `diff_llada.py`? | OSS / drop-in? |
|---|---|---|---|---|---|---|
| 1 | **dKV-Cache (Decode + Greedy)** | `arXiv:2505.15781` (NeurIPS'25) · [horseee/dKV-Cache](https://github.com/horseee/dkv-cache) | Delayed KV cache for bidirectional dLLMs: caches K/V for tokens whose representations have stabilised (one-step delay), reuses across remaining denoising rounds. Two variants: `Decode` (long-term reuse) and `Greedy` (per-step time-cx reduction). | 2–10× on LLaDA / Dream, near-zero perplexity hit | **Yes** — replaces `self._model(x)` call at line 444. K/V from previous step is held; only newly committed positions are re-projected. | Python wrapper around HF transformers attention; **no Triton/CUDA**. License unstated in repo top page (paper uses Apache-style). |
| 2 | **Fast-dLLM v1 (KV cache + confidence-aware parallel decoding)** | `arXiv:2505.22618` (NVlabs, ICLR-26) · [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) · [project page](https://nvlabs.github.io/Fast-dLLM/) | (a) Block-wise approximate KV cache for bidirectional attention, (b) confidence-aware parallel decoding that commits *all* tokens above a confidence threshold in one step (vs. fixed `n_transfer`). | Up to **27.6× throughput** on LLaDA/Dream at GSM8K/MATH/HumanEval/MBPP with negligible accuracy loss | **Yes, both layers.** (a) replaces line 444 forward pass; (b) replaces lines 457–460 `torch.topk(conf, k=k)` fixed-budget scheduler with `conf > τ` mask. | Apache-2.0. PyTorch + custom CUDA / C++ for KV-cache extension; no Triton mentioned. Targets LLaDA-8B and Dream-7B specifically — drop-in with the same HF model id sfumato uses. |
| 3 | **Fast-dLLM v2 (block-diffusion hierarchical cache)** | `arXiv:2509.26328` (NVlabs) · same repo | Hierarchical caching: block-level cache stores cross-block context, sub-block cache enables intra-block parallel generation. Designed for AR→dLLM-adapted models (~1B token fine-tune). | 2.5× over AR; 2.54× over Qwen2.5-7B-Instruct throughput | **Conditional.** Sfumato runs LLaDA-8B *without* fine-tuning so v2's training recipe doesn't apply, but the hierarchical cache structure still applies to the semi-AR sub-block schedule (lines 419–542). | Apache-2.0. Same NVlabs repo, separate `v2/` directory. |
| 4 | **Sparse-dLLM (dynamic bidirectional cache eviction)** | `arXiv:2508.02558` · [OpenMOSS/Sparse-dLLM](https://github.com/OpenMOSS/Sparse-dLLM) | Training-free framework integrating sparse attention with dynamic eviction of low-saliency KV entries from *both* prefix and suffix tokens (vs. AR which only evicts prefix). Delayed updates by one step. | Up to **10× higher throughput** vs. vanilla dLLM, similar peak memory, comparable accuracy on LLaDA & Dream | **Yes** — line 444 forward pass becomes a sparse-attention call; eviction logic sits between denoising rounds (line 442 loop). | OpenMOSS repo, Python on HF transformers attention path; the sparse pattern itself is a custom mask but the actual matmul is plain `torch.matmul` / SDPA. |
| 5 | **dLLM-Cache (adaptive prompt + response caching)** | `arXiv:2506.06295` · [maomaocun/dLLM-cache](https://github.com/maomaocun/dLLM-cache) | Long-interval prompt caching + partial response updates guided by feature similarity. Supports LLaDA, Dream, LLaDA-V, MMaDA. | Up to **9.1×** over standard dLLM, no perf loss most tasks | **Yes** — directly applies to the semi-AR loop (444). Prompt prefix is re-used across all 64 denoising rounds in sfumato by construction; this kernel just makes that explicit. | Apache-2.0. PyTorch only; **no custom CUDA/Triton**. |
| 6 | **SlowFast Sampling (three golden principles)** | `arXiv:2506.10848` · [LiangrunFlora/Slow-Fast-Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling) | Adaptive two-phase sampler: *exploratory* phase finds high-certainty/converging positions, *accelerated* phase rapidly commits them in parallel. Three "principles" govern when/where: certainty, convergence, positional. | **15.63×** alone on LLaDA, **34.22×** combined with dLLM-Cache | **Yes** — replaces the fixed `_num_transfer_tokens` schedule (line 248–263) and the `torch.topk` commit step (459–460). | Open PyTorch impl; algorithmic, not a fused kernel. |
| 7 | **D2F (Discrete Diffusion Forcing, faster-than-AR)** | `arXiv:2508.09192` · [zhijie-group/Discrete-Diffusion-Forcing](https://github.com/zhijie-group/Discrete-Diffusion-Forcing) | Train a block-wise causal-attention student to mimic a bidirectional dLLM teacher → enables prefix KV caching and inter-block parallel decoding. | **2.5×** vs LLaMA3-8B / Qwen2.5-7B AR on GSM8K; **>50×** vs vanilla LLaDA/Dream | **No (training-required).** Sfumato uses the published LLaDA-8B-Instruct base; D2F would require re-training a student model. Useful as a comparison baseline only. | OSS repo. PyTorch + HF transformers. |
| 8 | **CDLM (Consistency DLM)** | `arXiv:2511.19269` · [SqueezeAILab/CDLM](https://github.com/SqueezeAILab/CDLM) · [Together blog](https://www.together.ai/blog/consistency-diffusion-language-models) | Consistency-style fine-tune that enforces block-wise causal attention mask → multi-token finalisation per step + standard KV cache compatibility. | **3.6–14.5× lower latency** at competitive accuracy on math/coding | **No (training-required).** Same constraint as D2F; baseline reference. | OSS, Apache-style. |
| 9 | **WeDLM (Topological Reordering for causal-attention DLM)** | `arXiv:2512.22737` (Tencent) · [Tencent/WeDLM](https://github.com/Tencent/WeDLM) · [tencent/WeDLM-8B-Instruct](https://huggingface.co/tencent/WeDLM-8B-Instruct) | Reorders observed tokens to physical prefix while preserving logical positions → strict causal mask, full FlashAttention compatibility, immediate KV cache reuse. | **3–6×** vs vLLM-optimised AR on GSM8K, up to **10×** in low-entropy regimes | **No (training-required).** WeDLM-8B is a separate model, not a LoRA on LLaDA-8B. Reference baseline + lesson: causal-mask reorder is the cleanest "drop into FlashAttention" path. | Apache-style. |
| 10 | **dParallel (certainty-forcing distillation)** | `arXiv:2509.26488` · [HF paper page](https://huggingface.co/papers/2509.26488) | LoRA distillation that forces parallel certainty convergence so dLLM commits more tokens per step. | LLaDA-8B-Instruct: 256→30 steps GSM8K (**8.5×**), 256→24 MBPP (**10.5×**); 10 hr on 8×A5000 | **Conditional.** As a LoRA on LLaDA-8B-Instruct it's directly stackable with sfumato's existing PEFT path (`diff_llada.py:331-365`). The LoRA changes the *commit schedule* not the kernel. | OSS, LoRA-trainable. |
| 11 | **Learn2PD (learnable parallel decoder filter)** | `arXiv:2509.25188` (ICLR-26) · [project page](https://ims-kdks.github.io/learning-to-parallel/) · [GitHub](https://github.com/ims-kdks/Learning-to-Parallel-Decoding) | Tiny pre-trained filter MLP predicts which positions match the final output → adaptive per-position parallel commit. | **22.58×** alone, **57.51×** + KV cache | **Yes** — replaces sfumato's fixed `n_transfer` topk (lines 457–460) with a learned per-position gate. | OSS. Filter is small enough to run in the same kernel as Gumbel sampling. |
| 12 | **APD (Adaptive Parallel Decoding)** | `arXiv:2506.00413` (NeurIPS'25 oral) · [danielmisrael/apd](https://github.com/danielmisrael/apd) | Multiplicative mixture between dLLM marginal probs and a small AR auxiliary's joint probability → dynamic parallel-block size. | "Substantially higher throughput" with minimal accuracy drop on benchmark suite | **Yes** — the AR auxiliary maps cleanly onto sfumato's existing Qwen2.5-0.5B in `c3` / `cmerge`. Could re-use that model rather than load a new one. | OSS. |
| 13 | **Prophet (early-commit decoding)** | `arXiv:2508.19982` ("Diffusion LMs Know the Answer Before Decoding") | Training-free early-commit rule based on top-2 logit gap; halts decoding once the answer is stable. | **3.4×** reduction in decoding steps on LLaDA-8B / Dream-7B, no quality loss | **Yes** — sits between line 442 (step loop) and the boundary callback at line 484; emits an early `StepDirective.stop` when the top-2 gap exceeds threshold. | Paper-only at search time; algorithm is ~20 lines. Mark **drop-in** but **needs reimplementation**. |
| 14 | **Self-Speculative Decoding for dLLMs (SSD)** | `arXiv:2510.04147` (ICLR-26 submission) · [OpenReview](https://openreview.net/forum?id=rKJ7A30lQQ) | Use the same dLLM as draft + verifier in one forward pass: self-drafts all masked positions, verifies in a single step. No auxiliary model. | **Up to 3.46×** vs stepwise decoding, **identical output** on LLaDA / Dream | **Yes** — replaces the per-step commit at line 480 with a verify-then-commit batch. Identical-output guarantee makes this one of the safest drop-ins for sfumato's accuracy regression criterion. | OSS-pending; preprint code not yet released as of search time. **Speculative drop-in.** |
| 15 | **Focus-dLLM (long-context confidence-guided sparsification)** | `arXiv:2602.02159` | Past-confidence indicator predicts future unmasked regions; sink-aware pruning removes redundant attention computation. | **>29× lossless speedup** at 32K context | **Conditional.** Sfumato uses prompt+128 ≈ <1K tokens, so the long-context regime doesn't bite; gain would be much smaller (~1.5–2×). | Paper-only; targets H100-class GPUs. |
| 16 | **LoSA (Locality-aware Sparse Attention for block dLLMs)** | `arXiv:2604.12056` | Reuses cached prefix-attention for *stable* tokens and applies sparse attention only on *active* (changing) tokens. Solves the "KV inflation" problem where naive sparse fails on dLLMs. | Reported speedup ratios in paper; targets block-wise dLLMs | **Yes** — the active/stable token split lines up exactly with sfumato's `mask_index` (line 443). | Paper. **Speculative drop-in** (no public repo at search time). |
| 17 | **dLLM-Var (native variable length via [EOS])** | `arXiv:2510.24605` | Block-by-block append until EOS detected. Removes need to over-allocate `gen_length=128` when the answer is short. | **30.1×** vs traditional dLLM inference; **2.4×** vs Qwen / LLaMA AR | **Conditional.** Sfumato hard-codes `gen_length=128` (line 279) — a variable-length scheduler is structurally an inference change rather than a kernel. | Paper. |
| 18 | **DFlash (block-diffusion flash speculative decoding)** | `arXiv:2602.06036` · [z-lab/dflash](https://github.com/z-lab/dflash) · [project](https://z-lab.ai/projects/dflash/) | Lightweight block-diffusion *draft* model for speculative decoding of an AR target. Single-forward draft, conditioned on target features. | **>6× lossless** vs AR, **2.5× over EAGLE-3** | **Inverse use case** — DFlash uses dLLM to accelerate AR. Not directly applicable to sfumato (LLaDA is the *target*, not a drafter). Mentioned for completeness; useful if we ever flip Qwen-as-target / LLaDA-as-drafter. | OSS, Apache-style. MLX port also exists for Apple Silicon. |
| 19 | **FlashMask (column-wise sparse mask FlashAttention)** | `arXiv:2410.01359` (PaddlePaddle) | Column-wise sparse mask representation extends FlashAttention to bidirectional + complex masks at near-dense throughput. | **1.65–3.22×** end-to-end vs dense FA on LLM fine-tune/alignment | **Yes** — every step in sfumato's loop is bidirectional; FlashMask is the cleanest path to keep FA's fused softmax-matmul without losing mask-aware sparsity. | OSS in PaddlePaddle ecosystem; PyTorch port via FlexAttention is the practical path. |
| 20 | **FlexAttention (PyTorch 2.5+ programmable attention)** | [PyTorch blog](https://pytorch.org/blog/flexattention/) · in-tree | Python-level mask/score-mod functions JIT-compiled to fused FA kernels via `torch.compile` + Triton. Enables LLaDA's bidirectional + commit-position mask without writing kernels by hand. | Within ~5–10% of hand-written FA on standard masks; slower on exotic ones | **Yes** — line 444 wrapped with FlexAttention gives sfumato a one-line bidi-FA path on H100/4090. | PyTorch core, PyTorch BSD license. **Most ergonomic path.** |
| 21 | **FlashInfer sorting-free top-k/top-p sampling kernel** | [flashinfer.ai/2025/03/10/sampling.html](https://flashinfer.ai/2025/03/10/sampling.html) · [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Fused CUDA kernel for top-k / top-p / min-p sampling **without** sorting the vocab; rejection-based, single-launch. Used by vLLM, SGLang, MLC-LLM. | **>50% reduction** in sampling latency in vLLM 1×H100 across three models | **Yes** — sfumato's `_add_gumbel_noise` + `argmax` (lines 237–245, 446–447) is logically a temperature-Gumbel-then-sample. The per-step categorical sample over LLaDA's 126K-vocab × 128 positions × 64 steps × 5 branches is currently fp64 PyTorch — FlashInfer's kernel is the single-largest sampling speedup available. | Apache-2.0. |
| 22 | **Liger-Kernel (Triton fused LLM kernels)** | `arXiv:2410.10989` · [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Fused Triton kernels for RMSNorm (7×), CrossEntropy (3× faster, 5× less mem), RoPE (8×, 3× less mem), SwiGLU. | Avg **20% throughput** + **60% memory** reduction vs HF baselines | **Yes** for LoRA-train (sfumato `scripts/train_track*.py`); minor for inference (LLaDA already runs fp16 RoPE/RMS in HF). | MIT. |
| 23 | **SGLang Block-Diffusion RFC + LLaDA 2.0 day-0 support** | [sgl-project/sglang#12766](https://github.com/sgl-project/sglang/issues/12766) · [LMSYS blog 2025-12-19](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/) | Adapts SGLang's chunked-prefill pipeline to block-diffusion. CAP (Confidence-Aware Parallel) decoder ships day-0 for LLaDA 2.0 — **500 TPS** on `LLaDA2.0-flash-CAP` vs 383 TPS standard. | **1.9×** over AR baseline at 0.95 conf threshold | **Conditional** — LLaDA 2.0 ≠ LLaDA-8B-Instruct (sfumato's model). Migration cost is a sub-issue. | Apache-2.0. The most production-grade open kernel path. |
| 24 | **vLLM dLLM support (experimental)** | [vllm-project/vllm#18532](https://github.com/vllm-project/vllm/issues/18532) | Tracking issue for LLaDA / dLLM family. As of April 2026 still experimental, not production. | n/a | **Speculative.** Worth tracking but not drop-in. | Apache-2.0. |
| 25 | **LoRA-Switch SGMM kernel (fast adapter switching)** | `arXiv:2405.17741` | Custom SGMM CUDA kernel for token-wise LoRA merge/unmerge → makes per-token adapter switching practical. | Order-of-magnitude reduction in CUDA-kernel-call overhead vs PEFT default | **Yes** — addresses the comment at `diff_llada.py:319-321, 552-553` that PEFT adapter switching "walks every Linear layer". Sfumato switches at most twice per call but each switch is non-trivial; SGMM eliminates the cost entirely. | Paper; reference impl referenced. |
| 26 | **MaskGIT iterative parallel decoding (foundational)** | `arXiv:2202.04200` (Google) | Original confidence-based parallel commit schedule that LLaDA inherits: keep top-k confident predictions per step, remask the rest. | **Up to 64×** vs AR on ImageNet-256 (8–12 steps total) | **Yes by construction** — sfumato's `_num_transfer_tokens` (lines 248–263) and the topk-commit at lines 457–460 are direct ports of MaskGIT's scheduler. *Any* alternative schedule (Fast-dLLM's threshold, SlowFast, dParallel) is a strict generalisation. | TF / PyTorch ports; Apache-2.0. |
| 27 | **MDLM Rao-Blackwell sampler (3–4× over D3PM/SEDD)** | `arXiv:2406.07524` (NeurIPS'24) · [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | Simplified absorbing-state objective → sampler that's 3–4× faster than D3PM / SEDD samplers. | 3–4× over predecessors at matched perplexity | **Conditional** — LLaDA already uses an MDLM-class objective; the *sampler* equation matters. Could swap LLaDA's discrete-diffusion sampler at line 444 for the MDLM Rao-Blackwell variant. | MIT. |
| 28 | **SEDD score-entropy sampler** | `arXiv:2310.16834` (ICML'24 Best Paper) · [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | Score-entropy parameterisation enables **32×** fewer NFE at matched quality vs absorbing-state baselines. | 6–8× over GPT-2; 32× fewer NFE vs SEDD predecessors | **Conditional** — same constraint as MDLM: LLaDA's parameterisation is fixed at training time. SEDD is a comparison baseline, not a drop-in. | MIT. |
| 29 | **D3PM categorical / absorbing kernel** | `arXiv:2107.03006` (NeurIPS'21) · [google-research/d3pm](https://github.com/google-research/google-research/tree/master/d3pm) | Foundational categorical diffusion implementation with absorbing-state transition matrices. | Reference, not a speedup. | **No** — too foundational; LLaDA's training-time loss is downstream of this. Listed for completeness. | Apache-2.0. |
| 30 | **BD3-LMs reference impl (FlexAttention + SDPA backends)** | `arXiv:2503.09573` (ICLR'25 oral) · [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | Reference Block-Diffusion impl with `attn_backend ∈ {sdpa, flex}`. Block-causal mask is the cleanest example for sfumato. | n/a (reference); MDLM baseline requires `flash-attn==2.5.6` | **Yes** — the `flex` backend in BD3-LMs is the template sfumato should copy for the `sub_block_length=32` block-causal pattern. | Apache-2.0. |

**Distinct kernels surveyed: 30** (well above the ≥15 floor; 7 of them are
non-drop-in baselines / training-required, 23 are at-least-conditional drop-ins).

---

## 2. Hot-path map of `e4/diff_llada.py`

Per-line annotation of the kernel-amenable spots, ranked by current cost.

### 2.1 Line 444 — `logits = self._model(x).logits`  *(forward pass)*
- **What it is:** full bidirectional forward pass over `prompt_len + 128`
  tokens. Runs `64` times per `c2c` problem, `5 × 64 = 320` times per
  `cmajc` problem. **This is >95% of wall-clock time.**
- **Current cost:** 8B params × bf16 × ≤256 tokens × 320 calls
  = ~2.5 TFLOP × 320 = ~800 GFLOP/problem (forward only). On a 4090 at
  ~80 TFLOP/s bf16 effective ≈ 10 s/problem of pure matmul; observed
  wall-clock is ~12–20 s/problem ⇒ ~30% non-matmul overhead.
- **Kernel speedup ceiling:** with KV cache (dKV-Cache / Fast-dLLM v1):
  re-projection cost drops from `O(L)` to `O(L_new)` per step where
  `L_new` is the count of newly committed positions ≈ 2–8 of 128 ⇒
  theoretical **8–40× per step** if cache hits 100%. Empirically capped
  by re-projection of *moving* (boundary) tokens at **~3–10×** (matches
  dKV-Cache and Fast-dLLM v1 reported numbers). With Sparse-dLLM eviction:
  another **1.5–3×** stacked on that.
- **Best drop-in:** Fast-dLLM v1 (kernel #2) — same model id as sfumato.

### 2.2 Lines 442–480 — denoising loop body
- **Sub-ops (per step):**
  - 446: `_add_gumbel_noise` — `torch.rand_like` (fp64) + double `log` + divide. Currently fp64 across the entire `[1, L, 126K]` logits tensor.
  - 447: `argmax(dim=-1)` over 126K vocab × ~256 positions.
  - 449: `softmax(fp64)` over the same tensor.
  - 450: `gather` per-position confidence.
  - 452: scalar mask write.
  - 454–455: two `torch.where` calls.
  - 459–460: `torch.topk(conf[0], k=k)` over 256 positions.
  - 480: boolean scatter into `x`.
- **Current cost:** the fp64 softmax + gumbel noise on the full vocab is the second-largest hot spot; on a 4090 it is ~5–10% of step time. Topk-of-256 is negligible.
- **Kernel speedup ceiling:** **fuse 446–460 into a single Triton kernel**: input = bf16 logits, output = `(transfer_mask, x_new)`. Save the fp64↔bf16 round-trip + the gumbel materialisation. Plausible **2–5×** on this section, **0.1–0.5× end-to-end** because forward pass dominates.
- **Best drop-in:** FlashInfer top-k/top-p sampling kernel (#21) for the categorical step; Liger-Kernel (#22) primitives for cross-entropy / softmax fusions.

### 2.3 Lines 248–263 — `_num_transfer_tokens` (commit schedule)
- **What it is:** Python-level mask-count split. Runs once per sub-block,
  then a Python loop with `.item()` to make the per-step schedule.
- **Current cost:** trivially cheap (~µs); listed for completeness because
  Fast-dLLM v1 / SlowFast / Learn2PD / dParallel all *replace* this
  schedule.
- **Kernel speedup ceiling:** algorithmic, not kernel: a confidence-threshold
  schedule (Fast-dLLM) commits **2–4× more tokens per step** at iso-quality
  ⇒ **2–4× fewer steps overall** (multiplies the line-444 savings).

### 2.4 Lines 419–542 — sub-block while-loop / boundary callback
- **What it is:** semi-AR scaffolding plus the workstream-C
  `step_callback` hook. The boundary fires after each sub-block, accumulates
  trace data, and lets caller decide continue/stop/branch.
- **Kernel speedup ceiling:** `MaskGIT`-class (#26) confidence-based
  early-commit + Prophet (#13) early-stop on top-2 gap could collapse the
  number of sub-blocks adaptively. Stacks with everything else.

### 2.5 Lines 319–365 — PEFT adapter load / `_enable_commit` / `_disable_commit`
- **What it is:** PEFT walks every `Linear` to flip adapters. Sfumato
  switches at most twice per `denoise_block` call (never per step) — this
  is already the cheap regime.
- **Current cost:** small (one-shot per call; comment at line 319 already
  notes this is "not free").
- **Kernel speedup ceiling:** LoRA-Switch SGMM (#25) eliminates the cost
  entirely. Marginal because we already amortise.

### 2.6 Lines 203–261 (`runner.py`) — `cmaj` / `cmajc` branch loop
- **What it is:** sequential Python `for b in range(n_branches)` running
  full `denoise_block` per branch with different seed. **No batching.**
- **Current cost:** **5×** wall-clock for `BRANCHES=5`. This is the single
  largest *missed* batching opportunity in sfumato.
- **Kernel speedup ceiling:** batched LLaDA forward pass on `(B=5, L+128)`
  inputs with per-branch independent Gumbel noise → near-**5×** speedup at
  same memory cost (LLaDA-8B in bf16 ≈ 16 GB, 5× ≈ 24 GB ≈ 4090 ceiling
  — feasible but tight). Or run on H100/A100. **Highest-leverage change**
  in the whole project, ahead of any of the 30 kernels above.

### 2.7 Per-call overheads not in the loop
- Lines 440 / 463–478: trace accumulators (entropy, top-5 logits) that
  fire only when `step_callback is not None`. Negligible at default
  callback; can become 5–15% under the visualiser. Outside of kernel scope
  but a clean Triton fused entropy-of-row + top-5 would zero this.

---

## 3. Drop-in candidates (ranked)

Direct, open-source, applicable to LLaDA-8B-Instruct without re-training.

### 3.1 **Fast-dLLM v1** (NVlabs) — strongest single drop-in
- **Kernel:** Block-wise approximate KV cache + confidence-aware parallel decoding.
- **Repo:** [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) (Apache-2.0).
- **Engineering days to integrate:** **3–5 days.**
  - Day 1: clone repo, run their LLaDA-8B GSM8K reproduction at their config to confirm the kernel is alive on our hardware.
  - Day 2: wrap their `LLaDAModelWithKVCache` behind sfumato's `_Real._ensure_loaded` (line 296). No change to `denoise_block`'s public API.
  - Day 3: replace lines 442–460 with their `parallel_decode_with_kv_cache(x, threshold=τ)` call. Keep the confidence-based topk path as a fallback.
  - Day 4: re-run our `c2c` 5-problem mock test, then real on N=5 to verify text equality at temperature=0 (tolerance: identical extracted answer on ≥4/5).
  - Day 5: cmaj/cmajc batched: run `BRANCHES=5` against the cached forward pass.
- **Expected speedup on sfumato:** **5–12×** end-to-end on `c2`/`c2c` at GSM8K; **3–5×** on `cmaj`/`cmajc` (smaller because branches are not yet batched).
- **Kill criterion:** if exact-match accuracy on `c2c` N=20 drops by **>1.0pp** vs current 79%, abort. (The paper claims "negligible accuracy loss" — sfumato should verify on its own dev set, not theirs.) Secondary kill: if extracted-answer disagreement rate at temp=0 exceeds 10% on N=20, abort — that means the cache is degrading the *content* of the CoT, not just speeding it up.

### 3.2 **dKV-Cache** (NeurIPS'25) — the safer / simpler alternative
- **Kernel:** Delayed bidirectional KV cache, two variants.
- **Repo:** [horseee/dKV-Cache](https://github.com/horseee/dkv-cache) (Python only, no custom CUDA).
- **Engineering days:** **2–4 days.** Simpler than Fast-dLLM because no
  parallel-decoding semantics change — still uses sfumato's existing
  topk-commit schedule.
- **Expected speedup:** **2–4×** on `c2`/`c2c`; less than Fast-dLLM but
  zero risk to commit schedule.
- **Kill criterion:** same as Fast-dLLM (>1pp drop on `c2c` N=20 → abort).

### 3.3 **dLLM-Cache** (adaptive caching)
- **Kernel:** Long-interval prompt + partial response caching.
- **Repo:** [maomaocun/dLLM-cache](https://github.com/maomaocun/dLLM-cache) (Apache-2.0).
- **Engineering days:** **1–2 days** (their LLaDA wrapper is closest to the
  HF transformers shape sfumato uses).
- **Expected speedup:** **3–6×** on `c2`/`c2c`. Pairs cleanly with #3.4 below.
- **Kill criterion:** as above.

### 3.4 **SlowFast Sampling** — algorithmic, no kernel work
- **Kernel:** Three-principle adaptive scheduler, replaces fixed `n_transfer`.
- **Repo:** [LiangrunFlora/Slow-Fast-Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling).
- **Engineering days:** **1–2 days** to port the schedule into
  `_num_transfer_tokens`.
- **Expected speedup:** **3–8×** alone; **stacks** with caching (paper
  shows 34× combined with dLLM-Cache).
- **Kill criterion:** as above.

### 3.5 **Branch batching** (no external kernel; pure runner.py rewrite)
- **What:** rewrite `cmaj`/`cmajc` (`runner.py:203–261`) to batch
  `BRANCHES=5` into a single `(B=5, L+128)` LLaDA call.
- **Engineering days:** **2–3 days.** Mostly: making `_add_gumbel_noise`
  per-branch-seedable, making `_num_transfer_tokens` work on `B>1`, and
  ensuring the `confidence` and `transfer` tensors keep their `B` dim.
- **Expected speedup:** **~4×** on `cmaj`/`cmajc` (sub-linear in 5 because
  bf16 LLaDA-8B at B=5 is near 4090 memory ceiling — A100/H100 is ideal).
- **Kill criterion:** memory OOM ⇒ fall back to B=2 (still 1.8×) or use
  H100 spot. Accuracy: voting result on N=20 must match current `cmajc-v3`
  to within ±1pp.

### 3.6 **FlashInfer top-k sampling kernel**
- **Kernel:** Sorting-free top-k/top-p categorical sampling.
- **Repo:** [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
- **Engineering days:** **1 day.**
- **Expected speedup:** **0.05–0.15×** (5–15%) end-to-end (categorical
  sampling is small share of total time in sfumato; this is high-leverage
  only if the forward pass is already cached down).
- **Kill criterion:** numerical drift on temp=0 deterministic path > 0.

---

## 4. Applicability to sfumato by condition

`runner.py` ships eight conditions (`c1`, `c2`, `c2c`, `c2hint`, `c2empty`,
`c3`, `c3p`, `c4`, `crev`, `cmaj`, `cmajc`, `cmerge`). The four flagged in
the prompt:

### 4.1 `c2c` — single-shot diffusion + commit-LoRA
- **Forward passes per problem:** 64 (at `K_STEPS=64`).
- **Applicable kernels:** Fast-dLLM v1, dKV-Cache, dLLM-Cache, SlowFast,
  FlashInfer sampler, Prophet early-commit, FlexAttention (#20),
  MaskGIT-class scheduler tweaks. LoRA-Switch SGMM is marginal here since
  we switch adapters once per call.
- **Expected combined speedup:** Fast-dLLM v1 + SlowFast scheduler:
  **8–15×** end-to-end (multiplicative because they touch different ops:
  cache reduces matmul cost per step, SlowFast reduces step count).
- **Engineering days:** ~8 days for both (#3.1 + #3.4) plus 2 days
  validation (`c2c` N=20 + N=200 confidence-interval recheck).
- **Risk:** none kernel-side; risk is that combined optimisations push
  accuracy below the published 79% [72.7, 84.4] CI. **Kill if N=200
  exact-match accuracy drops below 76%.**

### 4.2 `cmaj` — 5 stochastic branches, no LoRA
- **Forward passes per problem:** 5 × 64 = 320.
- **Applicable kernels:** all of `c2c`'s list, **plus branch batching
  (#3.5)** which is unique to `cmaj`/`cmajc`/`cmerge`.
- **Expected combined speedup:** branch batching (4×) × Fast-dLLM v1 (5×)
  × SlowFast (3×) = nominal 60×. Memory and overlap losses cap this at
  **~20–30×** realistic end-to-end on a 4090; **~40–60×** on H100.
- **Engineering days:** ~12 days total (3.1 + 3.4 + 3.5 sequenced).
- **Risk:** branch batching changes the seed→token mapping subtly because
  Gumbel noise ordering across batch dim differs from sequential
  `torch.manual_seed(seed * 100 + b)`. **Kill if `cmaj` N=200 majority-vote
  accuracy diverges by >1.5pp** from the current 79.0% [72.7, 84.4].

### 4.3 `cmajc` — `cmaj` + commit-LoRA per branch
- **Forward passes per problem:** 5 × 64 = 320 (same as `cmaj`).
- **Applicable kernels:** identical to `cmaj`, *plus* LoRA-Switch SGMM
  (#25) becomes meaningfully relevant because the per-branch commit-LoRA
  switch fires `BRANCHES × 2 = 10` times per problem.
- **Expected combined speedup:** **~25–35× on 4090, ~50–70× on H100**.
- **Engineering days:** ~14 days (cmaj + LoRA-Switch port).
- **Risk:** higher than `cmaj` because LoRA-Switch is a custom kernel and
  `cmajc-v3 = 82.5%` is the headline number — any regression here is
  visible in the paper. **Kill if `cmajc-v3` N=200 drops below 80%.**

### 4.4 `cmerge` — diffusion germinate + AR finalize
- **Forward passes per problem:** 3 × 64 (LLaDA) + ~50 AR (Qwen).
- **Applicable kernels:** all `cmaj` kernels for the LLaDA leg; the AR
  leg is already well-optimised in HF (FlashAttention + KV cache by default
  via `transformers`).
- **Expected combined speedup:** **~15–25×** (less than `cmaj` because
  branches=3, AR leg is small but constant).
- **Engineering days:** ~10 days (LLaDA leg same as `cmaj`; AR leg unchanged).
- **Risk:** lowest of the four — `cmerge` has no headline number on the
  line. **Kill criterion:** N=200 accuracy within ±2pp of current.

### 4.5 `c2c` vs `cmaj`/`cmajc`/`cmerge` — sequencing recommendation

Implement in this order:
1. **`c2c` first** with Fast-dLLM v1 + SlowFast — proves the kernel path
   on the simplest condition with a 5–15× win and the lowest accuracy risk.
2. **Branch batching** for `cmaj` — unlocks the 4× sequential-loop saving
   that *every* branched condition shares, with no kernel work.
3. **`cmajc`** — once 1+2 are stable, the LoRA-Switch SGMM port lands the
   final speedup.
4. **`cmerge`** — by-product, near-free after `cmaj`.

---

## 5. Citations (≥15)

Papers (arXiv):
1. Nie et al., **LLaDA**: `arXiv:2502.09992` — base model used by sfumato.
2. You et al., **LLaDA-V**: `arXiv:2505.16933` — vision-language LLaDA (referenced for completeness).
3. Sahoo et al., **MDLM**: `arXiv:2406.07524` (NeurIPS'24) — `kuleshov-group/mdlm`.
4. Lou et al., **SEDD**: `arXiv:2310.16834` (ICML'24 best paper) — `louaaron/Score-Entropy-Discrete-Diffusion`.
5. Austin et al., **D3PM**: `arXiv:2107.03006` (NeurIPS'21) — `google-research/d3pm`.
6. Arriola et al., **BD3-LMs**: `arXiv:2503.09573` (ICLR'25 oral) — `kuleshov-group/bd3lms`.
7. Gong et al., **DiffuLLaMA / DiffuGPT**: `arXiv:2410.17891` (ICLR'25) — `HKUNLP/DiffuLLaMA`.
8. Chang et al., **MaskGIT**: `arXiv:2202.04200` (CVPR'22).
9. **JetAstra/SDAR**: `arXiv:2510.06303` — `JetAstra/SDAR`.
10. Ma et al., **dKV-Cache**: `arXiv:2505.15781` (NeurIPS'25) — `horseee/dKV-Cache`.
11. Wu et al., **Fast-dLLM v1**: `arXiv:2505.22618` (ICLR-26) — `NVlabs/Fast-dLLM`.
12. Wu et al., **Fast-dLLM v2**: `arXiv:2509.26328` — same repo, `v2/`.
13. **Sparse-dLLM**: `arXiv:2508.02558` — `OpenMOSS/Sparse-dLLM`.
14. **dLLM-Cache**: `arXiv:2506.06295` — `maomaocun/dLLM-cache`.
15. Liu et al., **SlowFast Sampling**: `arXiv:2506.10848` — `LiangrunFlora/Slow-Fast-Sampling`.
16. **D2F (Discrete Diffusion Forcing)**: `arXiv:2508.09192` — `zhijie-group/Discrete-Diffusion-Forcing`.
17. Kim et al., **CDLM**: `arXiv:2511.19269` — `SqueezeAILab/CDLM`. Together AI [blog](https://www.together.ai/blog/consistency-diffusion-language-models).
18. Tencent, **WeDLM**: `arXiv:2512.22737` — `Tencent/WeDLM`, `tencent/WeDLM-8B-Instruct`.
19. Bao et al., **dParallel**: `arXiv:2509.26488`.
20. **Learn2PD**: `arXiv:2509.25188` (ICLR-26) — `ims-kdks/Learning-to-Parallel-Decoding`.
21. Israel et al., **APD**: `arXiv:2506.00413` (NeurIPS'25 oral) — `danielmisrael/apd`.
22. **Prophet**: `arXiv:2508.19982` ("Diffusion LMs Know the Answer Before Decoding").
23. **Self-Speculative Decoding for dLLMs**: `arXiv:2510.04147` (ICLR-26 OR).
24. **Focus-dLLM**: `arXiv:2602.02159`.
25. **LoSA**: `arXiv:2604.12056`.
26. **dLLM-Var ([EOS] lead)**: `arXiv:2510.24605`.
27. Chen et al., **DFlash**: `arXiv:2602.06036` — `z-lab/dflash`.
28. **FlashMask**: `arXiv:2410.01359`.
29. **Liger-Kernel**: `arXiv:2410.10989` — `linkedin/Liger-Kernel`.
30. **LoRA-Switch (SGMM kernel)**: `arXiv:2405.17741`.

Repos / blogs (non-paper):
31. `NVlabs/Fast-dLLM` project page: <https://nvlabs.github.io/Fast-dLLM/>; v2: <https://nvlabs.github.io/Fast-dLLM/v2/>.
32. `flashinfer-ai/flashinfer` — sampling: <https://flashinfer.ai/2025/03/10/sampling.html>.
33. PyTorch FlexAttention blog: <https://pytorch.org/blog/flexattention/>.
34. SGLang block-diffusion RFC: [sgl-project/sglang#12766](https://github.com/sgl-project/sglang/issues/12766); LMSYS LLaDA 2.0 day-0 post: <https://www.lmsys.org/blog/2025-12-19-diffusion-llm/>.
35. vLLM dLLM tracking: [vllm-project/vllm#18532](https://github.com/vllm-project/vllm/issues/18532).
36. ML-GSAI/LLaDA reference sampler (the code sfumato ported in `diff_llada.py:1-13`): <https://github.com/ML-GSAI/LLaDA>.
37. VILA-Lab Awesome-DLMs survey index: <https://github.com/VILA-Lab/Awesome-DLMs>.
38. NVIDIA dev blog on FlashInfer in vLLM: <https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/>.
39. The Anatomy of a Triton Attention Kernel: `arXiv:2511.11581`.
40. MarkTechPost coverage of Fast-dLLM: <https://www.marktechpost.com/2025/06/01/nvidia-ai-introduces-fast-dllm-a-training-free-framework-that-brings-kv-caching-and-parallel-decoding-to-diffusion-llms/>.

---

## 6. Bottom line

Sfumato's `cmajc-v3 = 82.5%` is paid for by **320 LLaDA forward passes per
problem**, none of them KV-cached and none of them branch-batched. The
single highest-leverage change is **branch batching the `cmaj`/`cmajc`/
`cmerge` loop in `runner.py:203-261`** (~4× wall-clock; pure Python, no
kernel work). The single highest-leverage *kernel* change is **dropping in
Fast-dLLM v1's KV cache + confidence-aware parallel decoder** (~5–12× on
`c2c`, ~3–5× on `cmaj` after branch batching). The two stack
multiplicatively.

Conservative aggregate estimate, sequenced as recommended in §4.5:
- `c2c`: today 16-step diffusion, ~12 s/problem → **~1 s/problem**.
- `cmajc-v3`: today ~60 s/problem → **~3–5 s/problem on 4090**, **~1–2 s/problem on H100**.

That collapses an N=200 `cmajc-v3` run from ~3.5 hr ($0.70 spot) to
**~10–15 min ($0.05 spot)**, which is the difference between
"single-headline-number budget" and "ablation-grid budget".

The kill criterion that ties the entire kernel work to the paper's
science: **on N=200 GSM8K-test, `cmajc-v3` accuracy must remain within
[80.0, 85.0] of the current 82.5%.** Any kernel change that breaches this
is reverted before merge.
