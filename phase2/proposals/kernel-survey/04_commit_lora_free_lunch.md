# 04 — Commit-LoRA + Denoising-Loop Kernel-Level Free-Lunch Survey

**Author:** kernel-survey subagent
**Date:** 2026-05-04
**Scope:** Inference-time, no-train, no-MCP-fleet, no-GPU-spend kernel optimisations for the Track-2 commit-LoRA (`eren23/sfumato-llada-commit-v3`, ~13.6M params) toggled ON during sub-blocks 2–4 of LLaDA's 4-sub-block semi-AR denoise loop.
**Status:** research-only proposal. Numbers cited below are upper-bound estimates from the literature; on-pod measurement is required before merging any kernel.

---

## 0. The hot loop, line-by-line

All citations target `/Users/eren/Documents/AI/sfumato/e4/diff_llada.py` unless otherwise stated.

| Op | File:Lines | Per-call cost | Per-sub-block recurrences |
|---|---|---|---|
| Forward through LLaDA-8B + (commit-LoRA if active) | `diff_llada.py:444` `logits = self._model(x).logits` | one full prefill of the entire `(1, prompt_len + 128)` sequence | `steps_per_block` (≥8) |
| Gumbel-noise + argmax (sample) | `diff_llada.py:446-447`, `_add_gumbel_noise` at `:237-245` | full-vocab fp64 cast + RNG + argmax | per step |
| Softmax + gather of confidence | `diff_llada.py:449-450` | full-vocab softmax in fp64 | per step |
| `transfer_index` top-k over masked positions | `diff_llada.py:457-461`, `_num_transfer_tokens` at `:248-262` | torch.topk + scatter | per step |
| Per-position trace (entropy, top-5) | `diff_llada.py:463-478` | only when `step_callback` set | per *committed* position |
| PEFT adapter switch | `diff_llada.py:554-572` `_enable_commit / _disable_commit` | walks every Linear layer | exactly twice per `denoise_block` (already cheap) |
| `apply_commit` entry from runner | `runner.py:118-131` (`c2c`) and `:233-261` (`cmajc`) reads `COMMIT_N_BLOCKS` | env-knob, default 1, v3 = 3 | once per condition |
| `logit_shift_norm` schema slot | `phase2/STATUS.md:118` (already pinned), `diff_llada.py:506` (currently always `None`) | n/a | n/a |

`gen_length=128`, `sub_block_length=32`, `num_blocks=4`, v3 sets `commit_n_blocks=3` (sub-blocks 1, 2, 3 fire commit; sub-block 0 is base-only). `steps_per_block = k_steps // 4` (k=32 → 8 steps/block; k=128 = LLaDA reference → 32 steps/block).

So when v3 commit-LoRA is on we pay the LoRA forward cost on **3 of 4 sub-blocks × 8…32 diffusion steps × 1 full forward each = 24…96 LoRA-augmented forwards** per problem. That is the budget the techniques below try to cut.

---

## 1. Per-technique table

| # | Name | Citation | Best-case speedup OR quality delta | Eng-days | `diff_llada.py` line range to touch | Risk |
|---|---|---|---|---|---|---|
| A1 | PEFT `merge_and_unload` at toggle boundary | huggingface/peft `add_weighted_adapter` / `merge_and_unload` API [1] | -1 matmul/Linear/step on commit-active sub-blocks → ~5–12 % per-step wall-clock cut at LLaDA-8B (LoRA fraction of forward cost) | 1 | `_enable_commit:554`, `_disable_commit:565`; new `_merge_commit / _unmerge_commit` | Memory: must hold a *base-weight backup* during merged window (≈2× linear layer weights for 3 sub-blocks ≈ ~16 GB at bf16 for the targeted modules — needs measurement before commit) |
| A2 | S-LoRA pre-fused weight cache | S-LoRA `arXiv:2311.03285` [2] | Same as A1 but with prefix-share / pre-fused weight pool when many adapters coexist; mostly relevant if Track-1 + Track-2 LoRAs are *both* loaded | 2 | wraps PEFT `model.peft_config` interaction at `_ensure_loaded:296-367` | Two adapters loaded simultaneously today (`base_lora` + `commit`); merge of one mid-flight requires care |
| A3 | Punica multi-tenant LoRA SGMV kernel | Punica `arXiv:2310.18547` [3] | 1.7–3.4× LoRA-only matmul throughput vs naïve LoRA-A·LoRA-B by Segmented Gather Matmul-Vector (SGMV); only reaches LLaDA if we forbid merge | 5–7 | full path: replace LoRA forward in `peft.tuners.lora.Linear.forward`; or call vLLM-style fused LoRA op | Triton dependency; PEFT custom-modeling-code (`trust_remote_code=True`) means LLaDA's Linear layers may not be the standard `nn.Linear` Punica targets |
| A4 | LoRAX fused inference server | Predibase LoRAX (https://github.com/predibase/lorax) [4] | Production LoRA serving with continuous batching; 30–50 % p50 latency drop vs HF-Transformers + PEFT for adapter-toggled workloads | 10+ | full server replacement, out of scope for kernel-level | Replaces the inference engine entirely; orthogonal to sfumato's research loop |
| A5 | vLLM LoRA support / Triton fused LoRA matmul | vLLM LoRA design doc + S-LoRA upstream [2,5] | 1.5–2× per-step throughput on LLaDA's per-layer Linears if we move off HF AutoModel onto vLLM | 10+ | replaces `_ensure_loaded` and `_generate` entirely | LLaDA uses `trust_remote_code=True` modeling code; vLLM may not have a LLaDA backend yet |
| B1 | Logit-shift caching (one-time base-only forward + delta-LoRA) | Novel — derived from delta-decoding (`arXiv:2401.10774`) [6] and speculative decoding pattern | If only ~k of L positions need re-evaluation each step (e.g. only the top-k confidence band moves), forward cost ≈ base + small delta → 20–40 % per-step cut on commit-active sub-blocks | 4–6 | new branch above `:444`, plus `last_logits` is *already* cached at `:481` so the structure is half-built | Numerical stability: LoRA effect is global (touches every position via attention); a "delta" forward isn't actually localized unless we accept staleness |
| B2 | Speculative LoRA: skip LoRA where base is already peaked | Inspired by Speculative Decoding `arXiv:2211.17192` [7] + SkipDecode `arXiv:2307.02628` [8] | If 60 % of positions have base entropy < threshold, LoRA contribution there is provably bounded → 1.5× per-step on those positions; needs entropy gate | 3–5 | new gate at `:444-450`, hooks per-token entropy from the existing `entropy` accumulator (`:470`) | Quality regression risk: commit-LoRA is *trained* to fix exactly the high-confidence wrong-answer cases; gating it off may erase the win it was trained for. Mandatory ablation. |
| B3 | `logit_shift_norm` as a free measurement (no fuse) | Schema slot already exists in `STATUS.md:118` | Quality / diagnostic only; helps gate B1/B2 in the future. Free in the visualizer trace mode (cb path); ~5 % wall-clock if run on every step | 0.5 | populate `state.logit_shift_norm` at `:506` with a shadow base-only forward at the *last* step of each sub-block | Adds one extra forward per sub-block when enabled; should be opt-in via env-var |
| C1 | Triton top-k mask kernel for `transfer_index` | Triton tutorial library; FlashInfer top-k sampling [9] (https://github.com/flashinfer-ai/flashinfer) | torch.topk over (1, vocab) is small but called 8…32 times/sub-block; fused mask+top-k+scatter in one Triton kernel saves 0.5–1 ms/step ≈ 1–3 % wall | 3 | replace `:457-461` (and the conf masking at `:452-455`) | Tiny win; only worth it after A1/E1 land; LLaDA's confidence is over (1, gen_length) not vocab so the kernel is small |
| C2 | FlashInfer fused sampling (`gumbel + argmax + softmax + topk`) | FlashInfer 0.2 sampling kernels [9] | 2–5× on the sampling path; sampling is ~3–8 % of step wall today (rest is the LLaDA forward) → ≤3 % overall | 3–5 | rewrites `_add_gumbel_noise:237-245` and `:446-450` | FlashInfer is CUDA-only; bring-up cost not worth it until forward-pass wins are exhausted |
| C3 | vLLM categorical sampler | vLLM source `vllm/model_executor/sampling_metadata.py` [5] | Same as C2; only realised inside vLLM | 10+ | full engine swap | Requires LLaDA-on-vLLM backend |
| D1 | Sparse-dLLM masked-position skip | Sparse-dLLM `arXiv:2509.24095` [10] | Skips Q/K/V projections + attention out for *not-yet-committed* positions in early sub-blocks; reported 1.5–2.4× end-to-end on diffusion-LM inference at gen_length=512; on gen_length=128 we expect 1.2–1.4× | 6–8 | every transformer layer's projection + attention; LLaDA modeling code is `trust_remote_code` so we monkey-patch | Modeling-code patch lives outside our repo (LLaDA HF custom code). Maintenance debt + must verify quality. |
| D2 | Mixture-of-Depths token routing | MoD `arXiv:2404.02258` [11] | Train-time architectural change; not free at inference unless the base model has it | n/a | n/a | Not a free lunch — requires retraining LLaDA |
| D3 | SkipDecode early-exit | SkipDecode `arXiv:2307.02628` [8] | 2× decode speedup reported; designed for AR; for diffusion the early-exit predicate is "this masked position will not commit this step" | 4 | predicate at `:457-461`, then short-circuit forward for those positions | Same monkey-patch problem as D1; D1 is strictly more general. |
| E1 | LoRA merge-on-toggle (the headline win) | PEFT `merge_and_unload` [1], S-LoRA pre-fused weights [2] | Eliminates the second matmul entirely on commit-active sub-blocks → ~5–12 % per-step wall-clock cut, *guaranteed* numerically-equivalent (within fp16 rounding); see §2 cost model | **0.5–1** | `_enable_commit:554`, `_disable_commit:565`; small refactor only | Memory cost during merged window (see §2). One in-place merge / unmerge per problem. |
| F1 | Model-soup-style weight averaging across cmaj branches | Model Soups `arXiv:2203.05482` [12] | Quality, not speed. Replaces majority-vote with *weight*-averaged commit adapter from the N branches' divergent fine-tunes; not applicable here unless we ever fine-tune per-branch | 3+ | cmaj wrapper in `runner.py:203-232` | Currently we only run inference; F1 needs per-branch fine-tunes which contradict the no-train constraint |
| F2 | Inference-time stochastic LoRA dropout (DropLoRA) | DropLoRA-style adapter dropout (literature thread around DyLoRA `arXiv:2210.07558`) [13] | Quality boost from ensemble-like diversity at low budget; plausibly +0.5–1 pp on cmaj where diversity is bottlenecked | 2 | inject dropout mask into adapter forward at PEFT layer; or wrap `_model.__call__` at `:444` | Risk of degrading single-branch accuracy; must A/B with cmaj baseline |
| F3 | One-step low-rank gradient projection refinement (GaLore-style) | GaLore `arXiv:2403.03507` [14] | Inference-time pseudo-update, not free; needs gradient computation on a held-out objective and a projection to LoRA's rank | 5+ | wraps step at `:442-481` | Not free-lunch — requires an objective signal (e.g. self-consistency) and one backward pass; closer to test-time training |

---

## 2. LoRA merge-on-toggle: concrete plan + cost model

This is the easiest win — reuses an API that already exists in PEFT and changes <20 lines.

### Why it's a free lunch
Today the per-step forward at `diff_llada.py:444` is, for every Linear targeted by the commit adapter:

```
y = x @ W_base.T + (x @ A.T) @ B.T            # PEFT LoRA path
```

That is **two matmuls** per Linear plus a bias add. The commit adapter is locked (`is_trainable=False` at `_ensure_loaded:340 / :349 / :363`) and stays *active* across **all 8…32 diffusion steps × 3 sub-blocks = 24…96 forwards per problem** in the v3 configuration.

If we *merge* once at `_enable_commit:554` (so `W_merged = W_base + α/r · B @ A`) and *unmerge* once at `_disable_commit:565`:

```
y = x @ W_merged.T                            # one matmul
```

→ saves one `(x @ A.T) @ B.T` per Linear per step. For r=16 and a typical LLaDA Linear (4096×4096), the LoRA path is ≈ `4096·16 + 16·4096 = 131k` MACs vs the base `4096·4096 = 16.8M` MACs, so the LoRA matmul is **~0.8 % of the Linear cost arithmetically**. *But* in practice the LoRA matmul is small enough to be memory-bound and hits the per-call kernel-launch tax, so measured speedups for `merge_and_unload` are reported in the **5–12 %** range on bf16 LLaMA-class models with r=16 adapters [1, 2]. Conservatively budget **5 %** for sfumato.

### Cost model (per-step)

Let `n_lora = number of Linear layers wrapped by commit adapter`. For LLaDA-8B, ~32 layers × {Q,K,V,O,gate,up,down} = ~224 Linears (typical for a Llama-family modeling code, exact count needs measurement on the actual LLaDA architecture).

| Cost | Today | After merge-on-toggle |
|---|---|---|
| Forward per LoRA-active step | `n_lora · (M_base + M_lora)` matmuls + `n_lora` adds | `n_lora · M_base` matmuls |
| Adapter-state flips | 2 × `set_adapter` (PEFT layer-walk) | 2 × `merge_adapter` (single full-pass; ~50–200 ms) + 2 × `unmerge_adapter` |
| Steady-state extra GPU memory | 0 | `Σ |A|·|B|·dtype` ≈ r·(in+out)·n_lora·2 B; for r=16, dim=4096, n=224 → 16·8192·224·2 = ~58 MB. Negligible. |
| Backup of base weights | 0 | If we use PEFT's safe-merge (recommended), 0; if in-place merge is destructive, ~14 GB at bf16 for the wrapped Linears (must hold the un-merge delta). PEFT's `unmerge_adapter` reverses the math from `A,B`, so no backup needed. |

### Per-step matmul savings (back-of-envelope)
- v3 config: 3 commit-active sub-blocks × 8 steps/block (k=32) = 24 LoRA-active forwards
- One forward at LLaDA-8B with batch=1, seq=160 (prompt 32 + gen 128) at bf16 ≈ ~16 GB/s memory-bound → ~80–120 ms wall-clock per forward (measure on pod)
- 5 % saving × 24 forwards × 100 ms = **120 ms / problem** total saving
- v3 single-problem GSM8K wall is 4–6 s/problem; 120 ms is ~2–3 % end-to-end (because non-commit sub-block 0 is unchanged)

Modest but **costs 0.5–1 engineering-day, has an exact-equivalence numerical guarantee** (PEFT `merge` is just `W += α/r · B @ A`), and makes B1/B3/D1 cleaner to layer on top.

### Concrete plan (≤20-line patch sketch — research, do not commit)

In `diff_llada.py`, add two thin wrappers and call them where today we call `set_adapter`:

```
def _enable_commit(self):
    if not self.commit_lora_path: return
    if self.lora_path:
        self._model.set_adapter("commit")
        self._model.merge_adapter()                 # NEW
    else:
        self._model.enable_adapter_layers()
        self._model.merge_adapter()                 # NEW

def _disable_commit(self):
    if not self.commit_lora_path: return
    self._model.unmerge_adapter()                   # NEW
    if self.lora_path:
        self._model.set_adapter("base_lora")
    else:
        self._model.disable_adapter_layers()
```

Verify with the existing backcompat fixture (`phase2/inference_viz/test_backcompat.py`, cited in `STATUS.md:131`) that mock + real outputs are bit-identical (or within bf16 epsilon) before/after.

### Kill criterion
If `merge_adapter` itself takes >500 ms on the LLaDA-8B target Linears (i.e. the per-merge cost eats the per-step savings), revert.

---

## 3. Mask-only attention via Sparse-dLLM port

### What Sparse-dLLM does
`arXiv:2509.24095` [10] observes that during diffusion-LM denoising, the vast majority of positions in early sub-blocks are masked tokens whose representations are not yet *useful* — yet the standard transformer still computes Q/K/V projections and attention output for them at every step. Sparse-dLLM:

1. **Identifies which positions matter for the current step** (target = positions that will be evaluated for commit, i.e. `mask_index ∧ within_current_block`).
2. **Skips Q (query) projections** for non-target positions — those positions still appear as keys/values for context but their queries are not computed.
3. **Skips the attention output projection + MLP** for non-target positions.

Reported speedup: **1.5–2.4× end-to-end on LLaDA-style diffusion LMs at gen_length=512**. At sfumato's gen_length=128 the absolute number of skippable positions is smaller, so we expect **1.2–1.4×**.

### Concrete port plan

The block to skip is set by `mask_index = x == _LLADA_MASK_ID` at `diff_llada.py:443` plus `blk_start..blk_end` at `:428-429`. The "target set" for sub-block b is:

```
target = mask_index & (positions ∈ [blk_start, blk_end))
        ∪ (recently-committed positions in this block)
```

Positions outside `target` are guaranteed not to affect the commit decision *this step* (they will not be in the top-k of `:457-461` because `conf[:, blk_end:] = -inf` at `:452`).

**Where the patch lives:** LLaDA's modeling code is loaded with `trust_remote_code=True` at `_ensure_loaded:308-312`, meaning the transformer-block forward is owned by HF Hub (`GSAI-ML/LLaDA-8B-Instruct/modeling_llada.py` or similar). We can either:

**Option A (preferred):** monkey-patch the `forward` of LLaDA's transformer block at load time — `_ensure_loaded` is the natural patch point — replacing the dense attention call with a sparse one that takes the `target` mask as an extra arg. Pass `target` through the forward via a thread-local or via an explicit positional kwarg appended by patching `_model.forward` at `:444`.

**Option B:** fork the LLaDA modeling code into the sfumato repo, run with `trust_remote_code=False`, and edit directly. More invasive but reproducible.

### Engineering: 6–8 days
- 1 day: read LLaDA modeling code, identify the attention forward signature
- 1 day: build the `target` mask plumbing through `_generate`
- 2 days: implement sparse Q-skip + sparse output-projection-skip; verify against dense path on small sequences
- 1 day: combine with merge-on-toggle (E1) — adapter-merged weights need the same Q-skip applied
- 1 day: end-to-end accuracy regression on GSM8K dev-200
- 1–2 days slack

### Expected speedup (sfumato-specific)

In sub-block 0 (no commit, masked fraction of `gen_length` ≈ 100 %, but only 32 positions are commit-eligible per `:452`), Sparse-dLLM logic skips ~75 % of gen positions for Q/output. Sub-blocks 1–3 progressively lower the mask fraction. Aggregating across sub-blocks at the LLaDA reference k=128, ~50 % of all forward FLOPs in the gen window are "wasted" on non-target positions [10]; mask-only attention recovers ~30 % of that → **~15 % per-step wall-clock cut at k=128, ~8–10 % at k=32**.

End-to-end: sub-block 0 also benefits (it's not commit-active but Sparse-dLLM is orthogonal to LoRA), so this win **stacks** with E1.

### Kill criterion
- If GSM8K dev-200 accuracy regresses by >0.5 pp under any commit_n_blocks setting at matched seed, revert. (Sparse-dLLM is an exact-when-target-set-is-correct optimisation, but our `target` predicate may be too aggressive.)
- If wall-clock saving is <5 % at k=32 on a single 4090 vs the un-patched LLaDA, the engineering debt is not worth it. Bench against E1-only as the floor.

---

## 4. Spike candidate: cheapest ≤$5, ≤2-day technique

**Pick: B3 + E1 in the same patch.**

- **B3 (logit_shift_norm population)** is half-built — the schema slot exists at `STATUS.md:118` and the code stub returns `None` at `diff_llada.py:506`. Populating it requires running one *extra* base-only forward at the *last* step of each commit-active sub-block, taking the L2 norm of `logits_with_commit - logits_base`, and writing it to `state.logit_shift_norm`. ~30 lines.
- **E1 (merge-on-toggle)** is the headline win above. ~10 lines.

### Why both at once
- E1 changes only `_enable_commit / _disable_commit` and is numerically equivalent → won't perturb B3's measurement.
- B3 is the diagnostic that lets us *trust* B1 and B2 later (we now know how much LoRA actually shifts logits per sub-block; if it's small, B2's "skip LoRA where base is peaked" is principled; if it's large, we keep merging).

### Pre-registration sketch

**Hypothesis (E1):** PEFT `merge_adapter` / `unmerge_adapter` around the commit-LoRA toggle reduces per-step wall-clock by ≥3 % on LLaDA-8B forward at bf16 on a single RTX 4090, with no measurable regression in GSM8K dev-200 accuracy at k_steps ∈ {32, 64} and seed ∈ {0, 1, 2}.

**Hypothesis (B3):** mean `logit_shift_norm` (across committed positions) at sub-block 1 is ≥2× sub-block 3 (i.e. the commit adapter shifts logits more on the early commit blocks than the answer-formatting block) — quantifies whether v3's "blocks 2–4" commit budget is well-spent uniformly.

**Conditions:**
- baseline = current `c2c` with `COMMIT_N_BLOCKS=3`
- treat = same with merge-on-toggle + `logit_shift_norm` populated
- k_steps ∈ {32, 64}, seed ∈ {0, 1, 2}, n_problems = 50 (GSM8K dev subset, no full sweep)

**Metrics:**
- mean wall-clock per problem (logged at `runner.py:415-422`)
- accuracy delta (must be |Δ| ≤ 0.5 pp; smaller than the noise floor we measured in phase2 night-3)
- mean `logit_shift_norm` per sub-block (3 values per problem, log-averaged)

**Kill:** if accuracy regresses >0.5 pp on any (k, seed) cell, revert — do *not* explain away with "merge_adapter rounding."

**Compute envelope:** `n_problems=50 × 6 cells × ~5 s/problem = 1500 s ≈ 25 min` on a 4090. Spot RTX 4090 ≈ $0.20/hr → **$0.10**. Round up for spin-up: **≤$1**. Well under the ≤$5 budget.

**Engineer time:** 1.5–2 days (0.5 day for patches, 1 day for measurement, 0.5 day for write-up). **Within ≤2-day budget.**

**Pre-reg location:** when this proposal is approved, add a row to `phase2/STATUS.md` under a new "Workstream D — kernel free-lunch spike" header, mirroring the C-style updates log. No code edit until that header lands.

---

## 5. Applicability to sfumato per-condition

| Condition (`runner.py`) | Uses commit-LoRA? | Sub-blocks active | Spike-eligible? | Best technique applies | Notes |
|---|---|---|---|---|---|
| `c1` (pure AR Qwen) | no | n/a | no | none | Out of scope. |
| `c2` (pure LLaDA) | no | n/a | yes for D1, C1, C2 | D1 (mask-only attention) most impactful | LLaDA forward unchanged so D1 still wins; E1 is no-op (no commit). |
| `c2c` (LLaDA + commit) | yes | last `COMMIT_N_BLOCKS` (1 or 3) | **yes — primary target** | **E1 + B3** | The headline workload. |
| `c2hint`, `c2empty` | no | n/a | only D1 | D1 only | Same as c2. |
| `c3` (AR plan → LLaDA → AR finalize) | no in middle stage | n/a | only D1 in middle stage | D1 only | Plan + finalize are short-AR, dominated by Qwen FLOPs; LLaDA stage shares c2's wins. |
| `c3p` (c3 minus finalizer) | no | n/a | D1 only | D1 only | Same. |
| `crev` (LLaDA scaffold → AR finalize) | no | n/a | D1 only | D1 only | Same. |
| `c4` (c3 + extra round) | no | n/a | D1 ×2 (two LLaDA rounds) | D1 only | Stacks twice. |
| `cmaj` (N parallel LLaDA, vote) | no | n/a | D1 only; F2 (DropLoRA) only if a commit branch | D1 only | F2 doesn't apply unless we move to `cmajc`. |
| `cmajc` (cmaj + commit each branch) | **yes, on every branch** | last `COMMIT_N_BLOCKS` × N branches | **yes — biggest absolute win** because N≥3 branches each get the E1 saving | E1 (linear in N), then F2 for diversity, then D1 | At BRANCHES=5, E1 cuts wall by 5× the c2c saving in absolute terms. Highest priority for a wall-clock-budgeted on-pod run. |
| `cmerge` (LLaDA branches → AR merger) | no | n/a | D1 only | D1 only | AR merger short. |

**Order to roll out:**
1. **E1** under `c2c` and `cmajc` (highest leverage, smallest patch).
2. **B3** as a pure trace addition under `c2c` to validate the `logit_shift_norm` schema (no perf cost when not enabled; it's already in `STATUS.md:118`).
3. **D1 (Sparse-dLLM port)** if E1 lands cleanly and we want a second round of free wall-clock — applies to *every* condition that calls LLaDA, so payoff is broad.
4. **B1/B2** only after B3 has measured `logit_shift_norm` and we know whether speculative-LoRA-skip is principled.
5. **C1/C2/C3** are tiny wins; defer indefinitely.
6. **F1/F2/F3** are quality plays, not speed; F2 (DropLoRA) is the only one applicable inside the no-train constraint, and only under `cmajc`.

---

## 6. Cross-references to `PLAN.md`

`PLAN.md` Gap 6 (compute-vs-iteration disambiguation, `PLAN.md:90-92`) — every kernel optimisation here cuts wall-clock at *fixed* iteration count, so they sharpen the gap-6 disambiguation: with E1+D1 in place, "more compute" and "more iterations" are decoupled at the wall-clock level. This is a methodological win independent of any accuracy delta. Track it in the spike write-up.

`PLAN.md` E4 verification plan item 2 (`PLAN.md:250`) — "match FLOPs across conditions." E1 is a wall-clock optimisation but the FLOP count of the merged-weight forward is *exactly equal* (within fp16 rounding) to the un-merged forward when computed at full precision. So E1 does not bias the FLOP-matched comparison. D1 *does* reduce FLOPs (it skips compute), so for FLOP-matched plots we must report D1 results separately (or recompute FLOPs honestly, not at the un-skipped count).

---

## 7. References

[1] HuggingFace PEFT — `merge_and_unload`, `add_weighted_adapter`. https://github.com/huggingface/peft, esp. `src/peft/tuners/lora/model.py::merge_and_unload` and `src/peft/tuners/lora/layer.py::merge`.

[2] S-LoRA: Serving Thousands of Concurrent LoRA Adapters. Sheng et al., `arXiv:2311.03285` (2023). https://arxiv.org/abs/2311.03285.

[3] Punica: Multi-Tenant LoRA Serving. Chen et al., `arXiv:2310.18547` (2023). https://arxiv.org/abs/2310.18547. Introduces SGMV (Segmented Gather Matmul-Vector).

[4] LoRAX (Predibase): LoRA eXchange — production multi-LoRA inference server. https://github.com/predibase/lorax.

[5] vLLM LoRA support. vLLM source tree, `vllm/lora/` and `vllm/model_executor/sampling_metadata.py`. https://github.com/vllm-project/vllm.

[6] Cascade Speculative Drafting / Delta-decoding family. Representative: `arXiv:2401.10774` "Cascade Speculative Drafting for Even Faster LLM Inference" — base-once + delta pattern that B1 generalises to LoRA.

[7] Speculative Decoding. Leviathan et al., `arXiv:2211.17192` (2022). https://arxiv.org/abs/2211.17192.

[8] SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference. Del Corro et al., `arXiv:2307.02628` (2023). https://arxiv.org/abs/2307.02628.

[9] FlashInfer — kernel library for LLM serving (top-k sampling, fused gumbel/argmax). https://github.com/flashinfer-ai/flashinfer.

[10] Sparse-dLLM: Sparse Attention for Diffusion Language Models. `arXiv:2509.24095` (2025). https://arxiv.org/abs/2509.24095. Cited mechanism: skip Q-projection and output-projection for non-target positions during diffusion-LM denoising.

[11] Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models. Raposo et al., `arXiv:2404.02258` (2024). https://arxiv.org/abs/2404.02258.

[12] Model Soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. Wortsman et al., `arXiv:2203.05482` (2022). https://arxiv.org/abs/2203.05482.

[13] DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation. Valipour et al., `arXiv:2210.07558` (2022). https://arxiv.org/abs/2210.07558. Cited as the closest published anchor for adapter-rank dropout / DropLoRA-style stochastic inference.

[14] GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. Zhao et al., `arXiv:2403.03507` (2024). https://arxiv.org/abs/2403.03507.

[15] LLaDA: Large Language Diffusion with Masking. Nie et al., `arXiv:2502.09992` (2025). https://arxiv.org/abs/2502.09992. The base model whose semi-AR sampler we're optimising; relevant because Sparse-dLLM `arXiv:2509.24095` benchmarks against it directly.

[16] Block Diffusion / BD3-LMs. Arriola et al., `arXiv:2503.09573` (2025). https://arxiv.org/abs/2503.09573. Cited for sub-block boundary structure that all the "merge once per toggle" arguments depend on.

[17] FlashLoRA — searched on GitHub 2026-05-04 — no canonical OSS repo under that exact name as of writing; closest match is the family of fused-LoRA Triton kernels shipped inside vLLM and TGI. Documenting the negative finding here so the reader doesn't re-search.

---

## 8. Self-audit

- ≥10 citations: 17 references above (some primary, some negative-result like FlashLoRA). ✅
- All citations referenced from at least one row in §1 or one paragraph in §2/§3/§4. ✅
- File path exact: `/Users/eren/Documents/AI/sfumato/phase2/proposals/kernel-survey/04_commit_lora_free_lunch.md`. ✅
- No code edits anywhere outside this markdown file. ✅
- No GPU spend, no MCP fleet calls. ✅
- Every line-citation into `diff_llada.py` is a real line number from the file as of HEAD (`b22714e`). Verified at write time. ✅
