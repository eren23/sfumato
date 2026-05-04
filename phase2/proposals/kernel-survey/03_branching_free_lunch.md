# 03 — Branching free lunch: prefix sharing, paged batching, and pruning for `cmaj` / `cmajc`

**Scope.** Engineering-only optimizations specific to sfumato's branching codepath
(`cmaj` / `cmajc` / `cmerge`). No new science, no new model. The premise: the
current implementation runs `b=5` LLaDA continuations in a **Python `for` loop**
(`e4/runner.py:216-223` for `cmaj`, `:243-251` for `cmajc`) where every branch
re-prefills the **same** prompt + AR-plan tokens through a frozen LLaDA-8B and
re-runs the **same** mask schedule independently. There is no prefix sharing,
no paged KV, no inter-branch batching, no pruning, no continuous re-batching.
Eliminating these redundancies is a free lunch in the strict sense — same
accuracy, less wall-clock, no new accuracy/cost tradeoffs.

This document scopes which existing techniques (vLLM/SGLang/Hydragen/PagedAttention
/early-exit-self-consistency etc.) port cleanly to sfumato's mask-diffusion
sampler, what they cost in engineering days, and which one is the right
≤$5 / ≤2-day spike candidate for Phase 3.

---

## 0. The branching codepath, line-by-line

Read first:

- `e4/diff_llada.py:369-548` — `_Real._generate()` semi-AR sampler. Loops
  over `num_blocks=4` sub-blocks, runs `steps_per_block` denoising rounds per
  block (line 442), each step is one full `self._model(x).logits` forward pass
  (line 444) over `prompt_len + gen_length` tokens. **No batch dimension above
  1.** `x` is allocated `(1, prompt_len + gen_length)` at line 409.
- `e4/runner.py:216-223` — `cmaj` branch loop:
  ```python
  for b in range(n_branches):
      cot, used = diff_model.denoise_block(
          prompt=q, k_steps=k_steps, seed=seed * 100 + b, temperature=temperature,
      )
  ```
  Pure Python sequential; each call re-tokenizes the prompt
  (`diff_llada.py:608`), reloads `prompt_ids` to device, allocates a fresh
  `x` tensor of shape `(1, L+128)`, and runs `4 × steps_per_block` forward
  passes through LLaDA-8B from scratch.
- `e4/runner.py:243-251` — `cmajc` branch loop, identical except
  `apply_commit=True` / `commit_n_blocks=_get_commit_n_blocks()`. Adapter
  switch fires once per branch via `_enable_commit()` / `_disable_commit()`
  (`diff_llada.py:554-572`).
- `e4/runner.py:262-288` — `cmerge` branch loop (default `b=3`), then a
  single AR `finalize_answer` call.
- `e4/runner.py:117-131` — `c2c` is single-branch (`b=1`) baseline.

**The redundancy.** With `b=5`, prompt+AR-plan ~150 tokens, `gen_length=128`,
`steps_per_block=16` (k=64), each branch runs 64 forward passes over a
`L+128 ≈ 280`-token sequence. That is `5 × 64 = 320` forward passes per
problem, every one of them re-attending over the **same prompt prefix tokens**
(positions `[0:L]`) which never change across branches or across sub-blocks.

The first sub-block of every branch is **bit-identical for the first
`steps_per_block` rounds of attention over the prefix** — only the gumbel
noise on the masked positions differs. This is the prefix-KV-reuse
opportunity that nobody has exploited.

---

## 1. Per-technique table

| # | Technique | Citation | What it gives | Eng days | Touch points in `e4/diff_llada.py` | Risk | Expected wall-clock speedup on cmaj b=5 |
|---|-----------|----------|----------------|----------|-------------------------------------|------|------------------------------------------|
| A1 | **Prefix-KV reuse across branches** (RadixAttention-style) | Zheng et al., SGLang, `arXiv:2312.07104` [^sglang] | Skip the `O(L_prefix)` prefill on branches 2..b. | 2 | Replace `for b in range(n_branches)` (`runner.py:216-223`) with a single batched call; in `diff_llada._generate` accept `(B, L+gen)` `x`, gather logits per row. Prefix KV computed once via custom KV-cache manager. | M (LLaDA's HF `AutoModel` has no exposed KV-cache API; need to call `forward_with_past_kv` or hack `attention_mask`). | **1.4-1.7×** for prefill-bound regime; the prefill is `~30%` of total under `gen=128`, `L=150`, so ceiling ~1.42× from prefix alone. |
| A2 | **Hydragen — shared-prefix attention kernel** | Juravsky et al., `arXiv:2402.05099` [^hydragen] | Decomposes attention into `softmax(Q·[K_pref;K_branch])`; computes the prefix term ONCE across all `b` rows then per-branch suffix. | 3-5 | Drop-in replacement for `self._model.forward()` if Hydragen's CUDA kernels accept the LLaDA model. Realistically need to fork LLaDA's modeling file (`modeling_llada.py` from trust_remote_code) and patch attention. | M-H (LLaDA isn't on the Hydragen-supported list; needs a port). | Hydragen paper reports up to **32× end-to-end** at b=256, prefix=2048 tokens. At sfumato's b=5, L_pref=150, gen=128, the formula in [^hydragen] §4.2 gives **~1.6× wall-clock**. |
| A3 | **Cascade Inference / FlashInfer paged-cache** | Ye et al., FlashInfer; Pope et al., `arXiv:2211.04905`; FlashInfer `arXiv:2501.01005` [^flashinfer][^cascade] | Two-level KV cache: shared prefix in HBM read once, divergent suffix paged. Targets exactly the b-branched same-prompt case. | 4-6 | Same touch points as A1, plus need FlashInfer 0.2+ kernels. LLaDA's bidirectional attention pattern is the unknown — FlashInfer's cascade is built for causal masks. | H (bidirectional + mask-replacement may require new kernel). | **~1.5-1.8×** in the FlashInfer cascade benchmark at the relevant batch size. |
| B1 | **PagedAttention / vLLM** | Kwon et al., `arXiv:2309.06180` [^vllm] | Block-paged KV cache → batch `b=5` branches as a request batch with shared prefix block. | 3 (LLaDA in vLLM); 7+ (LLaDA isn't supported, need adapter) | Replace direct HF call with vLLM `LLM.generate(..., n=5, prompt=q)`. But: vLLM has **no LLaDA / mask-diffusion support** as of the cutoff — would need to register a custom model class with bidirectional attention + iterative-refinement scheduler. | H | **~1.3×** when LLaDA-shim works; vLLM's `n>1` shared-prompt path benchmarks at 1.3-1.5× over naive HF loops. |
| B2 | **SGLang RadixAttention** | Zheng et al., `arXiv:2312.07104` [^sglang]; SGLang blog Jan 2024 | Radix-tree-indexed prefix cache reuses across requests automatically. Same-prompt b=5 hits 100% prefix cache. | 5 (SGLang doesn't ship a LLaDA backend either) | New SGLang model registration; rewrite `_Real._generate` as an SGLang program. | H | RadixAttention paper [^sglang] §6.2 reports **2.2× on parallel-sampling benchmarks** with shared prefixes. At our scale closer to **1.5×**. |
| B3 | **FlashInfer paged-cache kernels (direct)** | FlashInfer `arXiv:2501.01005` [^flashinfer] | Bypass full vLLM, just use the kernel. | 3-4 | Hand-write a paged-cache attention call inside LLaDA's `modeling_llada.py`. | M-H | **~1.4×**, similar to A1. |
| C1 | **Speculative decoding (Leviathan / Chen)** | Leviathan et al., `arXiv:2211.17192` [^speculative]; Chen et al., `arXiv:2302.01318` [^chen-speculative] | Cheap drafter proposes, big model verifies. | N/A — **does not directly port to mask diffusion**; speculative decoding is causal-AR specific. | — | — | **0×** (architectural mismatch — see §C below). |
| C2 | **Self-speculative decoding for mask-diffusion LMs** | Christopher et al., `arXiv:2502.06768` (SpecDiff-LM); Israel et al., `arXiv:2508.02193` (Acc-Spec on dLLMs) [^specdiff-lm][^accelerated-spec-dllm] | Treat `b` branches as `b` drafts; commit positions where `≥3/5` agree, re-mask the rest. | 4-6 | New denoising loop in `_generate`: run b branches **in parallel** through one forward pass (batch dim = b), at each step take majority across rows for high-confidence positions, keep the rest masked, continue. | M | SpecDiff-LM paper [^specdiff-lm] reports **1.7-2.4× wall-clock**. |
| C3 | **Lookahead decoding analogue** | Fu et al., `arXiv:2402.02057` [^lookahead] | Lookahead is a Jacobi-iteration decoder; mask diffusion is already Jacobi-style. Adaptable: extra "lookahead" step at each block boundary checks future-block consensus. | 3 | New step in `_generate`'s outer loop. | M | **~1.2×** estimated; Jacobi gains plateau when the base sampler is already iterative. |
| D1 | **Early-exit self-consistency (ESC)** | Li et al. "Escape Sky-high Cost: Early-Stopping Self-Consistency for Multi-step Reasoning", `arXiv:2401.10480` [^esc] | When a quorum (e.g. 3 of 5) of branches is reached early, abort the rest. | 1 | `runner.py:216-223` — wrap loop with running `Counter`, exit when `most_common(1)[0][1] >= ceil(b/2)+ε`. Branches must run **in parallel** for this to save wall-clock; serially it only saves FLOPs. | L | **1.3-1.5×** if branches batched (see §3); **~1.0× wall-clock** in the current pure-serial implementation (still saves FLOPs). |
| D2 | **Confidence-margin early kill (Adaptive-Consistency)** | Aggarwal et al. "Let's Sample Step by Step", `arXiv:2305.11860` [^adaptive-consistency] | Stop drawing samples when posterior confidence on majority answer crosses threshold; uses Beta-prior over agreement rate. | 1-2 | Same hook as D1 but with the Beta-test instead of hard quorum. | L | Paper reports **40% sample reduction** at matched accuracy, ≈ **1.4× wall-clock under parallel batching**. |
| D3 | **Mid-denoising branch pruning** (sfumato-novel) | This proposal; informed by D1+ESC and "Early-Exit Diffusion" `arXiv:2305.10816` [^early-exit-diffusion] | Inspect the **partial** answer span (block 3, the `#### N` block in LLaDA's mock vocab; see `diff_llada.py:172-175`) at sub-block boundaries; if `≥3/5` branches already agree on the digits, kill the other 2 branches before they finish. | 2-3 | New `step_callback` (`diff_llada.py:96-101`) that votes across branches mid-sample. Requires A1 (parallel branches) for wall-clock savings. | M (false-pruning risk if early agreement reverses by final block — **falsifiable** on the 1750-branch substrate, see §3). | **~1.25×** on top of A1. |
| D4 | **Cost-efficient self-consistency (Kim et al.)** | Wang et al., "Self-Consistency Improves Chain of Thought Reasoning", `arXiv:2203.11171` [^selfconsistency]; Aggarwal `arXiv:2305.11860` [^adaptive-consistency]; Wan et al., "Dynamic SC", `arXiv:2408.17017` [^dynamic-sc] | Dynamic `b` per problem: easy problems need b=1, hard ones b=10. Calibrated on per-problem difficulty proxy (e.g. branch entropy at block 0). | 2-3 | New `runner.py` dispatch: one warm-up branch, then decide `b_remaining` based on its block-0 entropy. | L-M | **~1.5×** assuming a fat tail of "obvious" problems where one branch suffices. |
| E1 | **Continuous batching across problems** | vLLM `arXiv:2309.06180` [^vllm]; TGI [^tgi] | When branch j of problem i finishes early, slot in branch 0 of problem i+1 instead of idling. | 5-7 | Major restructure of `runner.py:368` outer loop into a request-batched scheduler. | H | **~1.2-1.4×** end-to-end on the full 200-problem run, but only meaningful if D1/D2/D3 are also active (otherwise all branches finish in lockstep). |
| F1 | **FlashInfer batched sampling kernel** | FlashInfer [^flashinfer] | Replaces the `torch.argmax(logits_n, dim=-1)` (`diff_llada.py:447`) + `_add_gumbel_noise` (`:237-245`) with a single fused Triton kernel. | 1-2 | Replace `_add_gumbel_noise` and the argmax in `_generate`. | L | **~1.05-1.10×**; sampling is not the bottleneck — forward pass dominates. |
| F2 | **vLLM custom sampler kernels** | vLLM source `csrc/quantization/sampler.cu` | Same as F1, vLLM flavor. | 1 | Same. | L | **~1.05×**. |
| F3 | **Categorical-sampling Triton kernel (custom)** | Tillet et al., Triton `arXiv:2006.07683` [^triton] | Hand-roll a fused gumbel + argmax + topk-confidence kernel. | 2 | `diff_llada.py:237-245`, `:447-461`. | L | **~1.05×**. |

---

## 2. Prefix-sharing concrete plan (technique A1)

### Quantifying the redundancy

For `cmaj` at the canonical Phase-1 settings:

- Prompt + AR-plan tokens **L_pref** ≈ **100-300**. (System prompt
  `_DENOISE_SYS` from `diff_llada.py:106-109` is ~30 tokens; GSM8K problem
  text 30-200 tokens; `Plan: ...` from `c3` adds 32 tokens by AR
  budget — but `cmaj` skips the plan. So `cmaj` baseline is L_pref ≈ 60-200,
  and `c3p`-style plan-prefixed cmaj is L_pref ≈ 100-260.)
- Generation length **L_gen** = 128 (from `diff_llada.py:279`).
- Sub-blocks `num_blocks` = 4; `steps_per_block` = `k_steps / 4` = **16 at k=64**.
- Branches **b** = 5.
- Per branch: `4 × 16 = 64` forward passes over a `(L_pref + 128)`-token sequence.
- Per problem: `5 × 64 = 320` forward passes.

**Redundant FLOPs** are exactly the share of compute spent on the prefix
columns of attention. LLaDA-8B is bidirectional, so for sequence length
`N = L_pref + L_gen`, attention is `O(N²)` per layer per step. The fraction
of attention compute spent on prefix-token columns is
`L_pref · N / N² = L_pref / N`. The fraction of MLP compute is `L_pref / N`
linearly (MLP is per-token).

For **L_pref = 150, L_gen = 128, N = 278**: prefix = **54%** of every
forward pass. Across `b=5` branches that 54% is computed **5 times** when
it could be computed once.

**Speedup ceiling for prefix-sharing alone.** If we eliminate `(b-1) × 0.54`
of total work, total work drops to `1 - (5-1)/5 × 0.54 = 0.568`. Theoretical
ceiling **~1.76× wall-clock**.

This matches Hydragen [^hydragen] §4 closely: their analytic model gives
`speedup = b / (1 + (b-1) × L_gen/N)` for the attention term; plug
`b=5, L_gen=128, N=278` and you get `5/(1 + 4 × 128/278) = 5/2.84 = 1.76×`.
Reality will land at **1.4-1.6×** after kernel-launch and gather overhead
([^hydragen] consistently reports 80-90% of the analytic ceiling for short
prefixes; 95%+ for long prefixes).

RadixAttention's reported **2.2× on parallel-sampling benchmarks** [^sglang]
is for longer prefixes (1k+ tokens) — sfumato is in the short-prefix regime,
so the realistic target is **1.4-1.6×**.

### Engineering sketch

1. Refactor `_Real._generate` to accept a batched `prompt_ids` of shape
   `(b, L_pref)`, allocate `x` as `(b, L_pref + L_gen)`, and broadcast all
   five seeds' gumbel noise across the batch dim. Already most of the work
   given LLaDA's HF `AutoModel(x)` likely accepts a batch dim — verify with
   one-line probe `model(torch.stack([x, x])).logits.shape`.
2. Reuse the **single** prefill pass for tokens `[0:L_pref]` across all
   branches. The cheapest hack: just batch normally and accept the prefix
   redundancy gets *batched* (same ops, batch dim = 5, wall-clock dominated
   by memory bandwidth not compute). This alone yields the **1.4-1.6×**
   from the b-batching effect, no kernel work needed.
3. Higher tier: write a custom KV-cache manager that holds the `[0:L_pref]`
   K/V tensors once and broadcasts them in the attention call. This is the
   Hydragen / RadixAttention payoff and squeezes out the last 10-20%.

### Why this is real free lunch

Branches are **identically distributed** at sub-block 0 (only differ in
gumbel noise on masked positions). Their prefix attention is bit-identical
up to the first masked-position commit. Sharing it doesn't change any
sample's distribution — same accuracy, same `cmaj=79.5%` and
`cmajc-v3=82.5%` headlines, less wall-clock.

**Estimated cost.** A single A100 / RTX 4090 day @ $0.50-1/hr to validate
the batched path on the existing 200-problem GSM8K-dev set. **≤$3**.

---

## 3. Branch-pool pruning concrete plan (technique D3)

### The hypothesis

Majority-of-5 forms when **3 branches agree**. If 3 branches agree at the
end of sub-block **3 of 4** (the answer-span block, where `#### N` lands
in LLaDA's schedule per `diff_llada.py:172-175`), they will almost certainly
still agree at sub-block 4 (commit-LoRA only sharpens, doesn't redirect
[^abl-b-result]). So we can **kill the other 2 branches** at the sub-block
3→4 boundary and save 25% of their wall-clock.

Stronger version: even at the end of sub-block **2 of 4**, the digits in
the partial CoT may already be majority-set. Pruning at block 2→3 saves 50%.

### Triggering frequency, estimated from substrate

Existing labeled substrate (per `phase2/STATUS.md` line 52):

- 1750 labeled branches: 1000 from `raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`
  (N=200 × b=5) plus 750 from earlier τ ∈ {0.7, 1.0, 1.3} sweeps.
- Per-branch full text (final sub-block 4 output) is logged in
  `trace[f"branch_{i}"]` (`runner.py:228-229`).

We do **not** currently log mid-denoising state per branch. But we can
emulate: re-run the existing 200 problems with a `step_callback`
(`diff_llada.py:96-101`) that logs the **partial extracted answer** at
each sub-block boundary (just call `grade.extract_answer(decode(x))` on
the in-flight `x` tensor at each callback). This gives per-branch (sub-block,
partial_answer) tuples and is the same scaffolding Workstream-C already
shipped — see `phase2/inference_viz/traces/trace_*.jsonl`.

**Pre-experiment estimate** (from Phase-1 anecdote): cmaj's `oracle=88%`
vs `cmaj=79.5%` voting-rule gap (`STATUS.md:52`) implies that on
**~80% of problems, ≥3 of 5 branches converge on the right answer**. The
question is *when* — block 2, block 3, or only at block 4. Two
testable predictions:

- **At block 3 boundary**: ~70% of problems already show 3-of-5 agreement
  on extracted digits. Pruning-trigger rate **~70%**, savings per
  triggered problem **25%**, expected speedup **1 / (1 - 0.7 × 0.25 × 0.4) ≈ 1.07×**
  on top of A1 (the 0.4 factor is "fraction of remaining wall-clock the
  pruned branches were going to consume").
- **At block 2 boundary**: ~30-40% of problems show 3-of-5 agreement
  (numbers in partial CoT but answer slot still masked). Triggers less
  often but saves more. Expected speedup **~1.10× on top of A1**.

Combined with A1: total **~1.7-1.8× wall-clock** for cmaj b=5.

### Experiment design (pre-reg sketch)

- **Substrate**: 1750 branches already on disk. Augment by re-running
  the 200-problem `cmaj` set with a partial-answer-logging `step_callback`
  (~30min on one 4090, **~$0.20**).
- **Metric 1** (triggers): for each problem, at each sub-block boundary,
  count how often `Counter(partial_answers[branches]).most_common(1)[0][1] >= 3`.
  Report distribution over sub-block index.
- **Metric 2** (false-pruning rate): for each triggered (problem, sub-block)
  pair, check whether the final majority winner equals the early-trigger
  winner. Failure mode = pruning commits to A early, but the unpruned
  branches would have flipped to B.
- **Pass criterion**: trigger rate ≥ 50% AND false-prune rate ≤ 2pp
  accuracy degradation (so cmaj drops at most 79.5 → 77.5%).
- **Kill criterion**: false-prune rate ≥ 4pp OR trigger rate ≤ 20%.
- **Budget**: $0.20 for substrate augmentation, no GPU for the analysis
  (pure pandas on the JSONL).

---

## 4. Spike candidate: A1 + D1 (batched branches + early-quorum exit)

**Pick.** Technique **A1 (batched prefix-sharing via simple b-batching)**
combined with **D1 (early-exit self-consistency at quorum=3)** — the
simplest free lunch that's bounded above by ~1.4× alone but combines
with D1 to ~1.5-1.7×.

**Why this combo and not the bigger ones?**

- A1 alone is a `BRANCHES` for-loop refactor + a batch-dim probe. ~1
  engineering day if LLaDA's HF model accepts batched inputs natively
  (very likely: every transformer does).
- D1 alone needs only `runner.py:216-223` to maintain a running `Counter`
  and `break` when quorum hit. ~0.5 days. But D1 only saves wall-clock if
  branches run in parallel (else other branches still execute). So D1
  needs A1.
- Hydragen / FlashInfer / SGLang give 10-20% more speedup but cost 3-7
  more engineering days and carry a real risk that LLaDA's bidirectional
  attention + custom modeling code doesn't slot into their kernels. Not
  ≤2 days, not ≤$5, not the right spike.

### Pre-registration

- **Hypothesis**: A1+D1 yields **≥1.3× wall-clock speedup** on `cmaj b=5`
  with **≤1pp accuracy delta** vs the baseline `cmaj=79.5% [73.3, 84.9]`
  at N=200.
- **Success criterion**: median `cmaj b=5` wall-clock at N=20 ≥ 1.3×
  faster than baseline `cmaj` on the same 20-problem subset, AND
  `cmaj-batched` accuracy on full N=200 lands inside `[78.0%, 81.0%]`
  (the binom-CI ±1pp band of 79.5%). **WIN.**
- **Kill criterion**: speedup < 1.15× OR accuracy outside `[77.5%, 81.5%]`. **LOSS.**
- **Inconclusive**: speedup ∈ [1.15×, 1.3×) — proceed to A2/Hydragen as
  Phase 3 spike.
- **Measurement plan**:
  1. Same RTX 4090 spot pod ($0.50/hr) used for Phase 2 substrate runs.
  2. Baseline: `cmaj b=5 k=64 SEED=0 N=20 v3-LoRA` — log per-problem
     wall-clock from `runner.py` row dict (already has `flops`, add
     `wallclock_ms`).
  3. Treatment: same run with the A1+D1 patch.
  4. Compare distribution of per-problem wall-clock with paired Wilcoxon
     (each problem run under both conditions, same seed).
  5. If WIN at N=20 → re-run at N=200 for the accuracy CI confirmation.
- **Budget**: 1 day pod ($12 raw, but spot < $1/hr → ~**$3-5** end-to-end).
  ~1.5 engineering days. Total **~$5 GPU + 2 eng days**.
- **Falsification artifacts**: per-problem wall-clock JSONL diff,
  binom-CI accuracy comparison. Land in `phase2/spikes/branching-free-lunch/`.

---

## 5. Applicability to sfumato's four conditions

| Condition | b | Speedup ceiling (A1) | Speedup ceiling (A1+D1+D3) | Eng days | Risk | Notes |
|-----------|---|----------------------|----------------------------|----------|------|-------|
| `c2c` (`runner.py:117-131`) | 1 | **1.00×** (no branches) | 1.00× | — | — | Single-branch baseline. No branching free lunch applies. F1/F3 (sampler kernels) still give ~5%. |
| `cmaj` (`runner.py:203-232`) | 5 | **1.4-1.6×** | **1.7-1.9×** | A1: 1.5; +D1: 0.5; +D3: 2 | M | **Primary target**. Highest ROI. |
| `cmajc` (`runner.py:233-261`) | 5 | **1.3-1.5×** | **1.6-1.8×** | A1: 1.5; +D1: 0.5; +D3: 2.5 | M | Slightly lower than `cmaj` ceiling because the per-branch PEFT adapter switch (`_enable_commit` / `_disable_commit`, `diff_llada.py:554-572`) is a serial fixed cost that doesn't batch. Acceptable: switch is once-per-call, not once-per-step. |
| `cmerge` (`runner.py:262-288`) | 3 | **1.2-1.35×** | **1.4-1.55×** | A1: 1.5 | M-L | Lower b means lower prefix-sharing ceiling. Plus there's a final AR `finalize_answer` call (`:282-284`) that's not batched — Amdahl-bottlenecks the gains if the AR pass dominates. Diminished but still positive. |

---

## 6. Bibliography

[^sglang]: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", `arXiv:2312.07104`, 2024. RadixAttention details in §4. SGLang blog: <https://lmsys.org/blog/2024-01-17-sglang/>.

[^hydragen]: Juravsky et al., "Hydragen: High-Throughput LLM Inference with Shared Prefixes", `arXiv:2402.05099`, 2024. Decomposed-attention analytic speedup model in §4.2; empirical ~32× at b=256 prefix=2048 in §5.

[^vllm]: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, `arXiv:2309.06180`. vLLM project: <https://github.com/vllm-project/vllm>.

[^flashinfer]: Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", `arXiv:2501.01005`, 2025. Cascade inference / paged-cache kernels: <https://github.com/flashinfer-ai/flashinfer>.

[^cascade]: Pope et al., "Efficiently Scaling Transformer Inference", `arXiv:2211.04905`, 2022. The original "shared prefix at scale" framing; informs FlashInfer cascade.

[^speculative]: Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding", `arXiv:2211.17192`, 2023.

[^chen-speculative]: Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", `arXiv:2302.01318`, 2023.

[^specdiff-lm]: Christopher et al., "Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion", `arXiv:2502.06768`, 2025. Mask-diffusion-specific drafter→verifier scheme; reports 1.7-2.4× on dLLM benchmarks.

[^accelerated-spec-dllm]: Israel et al., "Accelerating Diffusion Large Language Models with SlowFast Sampling", `arXiv:2508.02193`, 2025. Speculative-style acceleration for masked-diffusion LMs (LLaDA family).

[^lookahead]: Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", `arXiv:2402.02057`, 2024. Jacobi-style decoder.

[^esc]: Li et al., "Escape Sky-high Cost: Early-Stopping Self-Consistency for Multi-step Reasoning", ICLR 2024, `arXiv:2401.10480`. Quorum-based early termination of self-consistency sampling.

[^adaptive-consistency]: Aggarwal et al., "Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs", EMNLP 2023, `arXiv:2305.11860`. Beta-prior adaptive stopping rule.

[^selfconsistency]: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", `arXiv:2203.11171`, 2022. The original self-consistency CoT.

[^dynamic-sc]: Wan et al., "Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling", `arXiv:2408.17017`, 2024.

[^early-exit-diffusion]: Tang et al., "DeeDiff: Dynamic Uncertainty-Aware Early Exiting for Accelerating Diffusion Model Generation", `arXiv:2305.10816`, 2023. Per-step early-exit in diffusion samplers.

[^triton]: Tillet, Kung, Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL 2019; release notes / kernels documented in PyTorch `torch.compile` integration.

[^tgi]: Hugging Face Text Generation Inference (TGI). Continuous batching reference implementation: <https://github.com/huggingface/text-generation-inference>.

[^abl-b-result]: sfumato Phase-2 ABL_B sanity probe (`phase2/spikes/abl_b_RESULT.md`, commit `27da1c9`): commit-LoRA-v3 systematically shifts answer-format markers but does not flip extracted digits on the 5 problems tested — i.e. final-block commit reinforces rather than redirects. Supports the D3 false-prune-rate prior estimate.

---

## 7. Open exposures

- **LLaDA-on-vLLM/SGLang/FlashInfer**: none of the three production engines
  ship a LLaDA / mask-diffusion model class. A2/B1/B2/B3 all carry a
  `port LLaDA` line item that is itself a 3-7 day project. The proposal
  filters these to lower-priority Phase 3 candidates.
- **Bidirectional vs causal attention kernels**: most cited kernels
  (PagedAttention, FlashInfer cascade, Hydragen) target causal attention.
  LLaDA is bidirectional. Hydragen's decomposition does not depend on
  causality (works for any prefix-suffix split), so A2 is more portable
  than B1/B2/B3. Confirm before any kernel work.
- **PEFT adapter switching cost**: `cmajc` adds the `_enable_commit` /
  `_disable_commit` PEFT round-trip (`diff_llada.py:554-572`). Per-call,
  not per-step, but if branches batch then this overhead amortizes 5×.
  Verify it doesn't dominate at b=5 with one timing print before claiming
  the 1.6× ceiling.
- **Voting-rule gap is a different problem**: the 8.5pp `oracle - cmaj`
  gap (`STATUS.md:52`) is an *aggregation-quality* problem, separately
  studied in `phase2/proposals/verifier-based-aggregation.md` and
  D3.5-LOSS in `STATUS.md:65-67`. The free lunch in this proposal is
  orthogonal: it makes the existing 79.5% cheaper, not better. Combining
  a future verifier win with these speedups is multiplicative.
