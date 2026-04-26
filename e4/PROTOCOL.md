# E4 — Protocol

## Hypothesis

Interleaving autoregressive (AR) generation with diffusion-style iterative refinement during chain-of-thought reasoning produces higher accuracy than either paradigm alone, *at matched compute*. If the effect is real it should appear at small scale (≤2B parameters) on GSM8K.

## Models (param-golf picks)

- **AR backbone:** `Qwen/Qwen2.5-0.5B-Instruct` (~0.5 B params, ~1 GB bf16). Fallback: `HuggingFaceTB/SmolLM2-1.7B-Instruct` if Qwen-0.5B's GSM8K is <10 % zero-shot.
- **Diffusion LM:** `GSAI-ML/LLaDA-8B-Instruct` (~8 B params, ~16 GB bf16). Smallest serious public discrete-diffusion LM; "LLaDA-1.5" is a v1.5 of the same 8 B model, not a smaller variant. Fallback: `diffusionfamily/diffullama` (~7 B, AR→diffusion adapted from LLaMA-2).
- **Co-residency on 1×4090 (24 GB):** Qwen-0.5B + LLaDA-8B in bf16 ≈ 17 GB weights; KV cache + activations comfortably fit in the remaining 7 GB. The hybrid pipeline is sequential per problem so no concurrent forward passes.
- **Param-golf framing:** total ~8.5 B is small by today's standards; a positive E4 result here is a much stronger plausibility argument than at frontier scale.

## Conditions

| Code | Pipeline |
|---|---|
| **C1** | Pure AR. Qwen emits CoT + answer in one pass. |
| **C2** | Pure diffusion. LLaDA denoises a 256-token CoT-and-answer block in `k` steps. |
| **C3** | AR plan → diffusion CoT → AR answer. Qwen emits ≤32 plan tokens; LLaDA denoises the CoT for `k` steps; Qwen reads the CoT and emits the final answer. |
| **C4** | Multi-round hybrid. C3 + one extra round: Qwen extends the CoT a few tokens, LLaDA re-denoises for `k` more steps, then Qwen emits the answer. |
| **C5 (sanity)** | Frozen-noise control. C2/C3 with LLaDA replaced by random tokens at `k=16`. If accuracy ≈ C1, the diffusion model is doing real work. If accuracy ≈ C2/C3, the experiment is broken. |

## Sweep

- `k ∈ {4, 8, 16, 32, 64}` for C2, C3, C4. C1 has no `k`.
- 200 frozen GSM8K dev problems (indices in `data/gsm8k_dev_200.json`).
- Seeds: `{0, 1, 2}` for `k=16` only (variance estimate); seed 0 for the rest.

Total runs: 1 (C1) + 5×3 conditions × 200 problems = 3,001 base runs + 3 seeds × 3 conditions × 200 problems at k=16 = ~4,800 total inferences.

## FLOPs accounting

Per-token forward FLOPs ≈ `2 × N_params + 2 × n_layers × n_ctx × d_model` (Kaplan-style approximation; see `flops.py` for exact formula).

- C1: `flops = output_tokens × forward_flops(qwen_0.5b)`
- C2: `flops = k × block_len × forward_flops(llada_8b)`
- C3: `flops = plan_tokens × forward_flops(qwen) + k × block_len × forward_flops(llada) + answer_tokens × forward_flops(qwen)`
- C4: `flops = C3 + extension_tokens × forward_flops(qwen) + k × block_len × forward_flops(llada)`

Always plot accuracy vs `log10(flops)`, not vs wall-clock.

## Kill criterion

Continue to E2 only if **both**:
1. `max(C3, C4) ≥ max(C1, C2) + 1.5 pp` at ≥1 FLOP budget on GSM8K-dev200, AND
2. C5 sanity: frozen-noise control's accuracy is at most `C2 - 5 pp` (i.e., LLaDA is doing real refinement, not noise-averaging).

If either fails: write up the negative result and stop the programme.

## Sanity checks

- **Reproducible seed.** Fix seeds for diffusion sampling. Report std-dev across 3 seeds at `k=16`.
- **Manual spot-check.** Read 10 outputs per condition by eye. C3's "AR plan" should be coherent; C2/C3's diffusion CoT should be readable text, not gibberish.
- **FLOPs honesty.** Unit-test `flops.py` against NanoGPT's published formula on a known config.
- **Forced-mode reduction.** C3 with `k=0` should match C1 (no diffusion = pure AR with a redundant plan-prefix).

## Outputs

- `results/raw_runs.parquet` — one row per (problem, condition, k, seed) with: prediction, gold, correct, flops, wall_clock, output_text.
- `results/leaderboard.csv` — aggregated by (condition, k): accuracy, flops_avg, flops_p50, flops_p95.
- `results/plot_accuracy_vs_flops.pdf` — primary deliverable.
- `writeup/e4_note.md` — 4-page artefact.

## Decision

| After E4 | If... | Then |
|---|---|---|
| Both kill criteria pass | Hybrid > pure paradigms | → E2 (span-level refinement) |
| Either fails | No real signal | Write up negative result; reconsider programme |
| Ambiguous | C3 ≈ max(C1, C2) ± 0.5 pp | Pivot to a harder benchmark (MATH-Easy, ARC-Challenge); cap pivot spend at $20 |
