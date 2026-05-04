# Pre-registration — Fast-dLLM v1 drop-in (S4)

**Phase:** Phase 2/3 spike chain
**Spike ID:** S4 (NVlabs Fast-dLLM v1 drop-in for LLaDA-8B)
**Author:** main session
**Date:** 2026-05-04
**Plan source:** `phase2/proposals/kernel-survey/01_diffusion_lm_kernels.md` §3.1.

## Hypothesis

Wrapping LLaDA-8B-Instruct with NVlabs Fast-dLLM v1's `LLaDAModelWithKVCache`
and replacing the per-step body in `e4/diff_llada.py:_generate` with
`parallel_decode_with_kv_cache(x, threshold=τ)` yields:

- **≥3× wall-clock speedup** on `c2c` N=200 (k=64, COMMIT_N_BLOCKS=3,
  seed=0, prefix-robust + commit LoRAs).
- `cmajc-v3` N=200 accuracy in **[80.0%, 85.0%]** (the ±2.5pp band
  around the 82.5% headline; matches the kernel-survey acceptance band).

## Conditions tested

- **Baseline:** `FAST_DLLM=0` (default; legacy denoise loop).
- **S4 treatment:** `FAST_DLLM=1`. Same env knobs otherwise. Confidence
  threshold τ swept ∈ {0.85, 0.90, 0.95} on N=20 first; pick the τ that
  preserves accuracy best, then run N=200 at that τ.
- **Order:** run on `c2c` first (single-branch, lowest accuracy risk).
  Promote to `cmajc` only if c2c WINS.

## Success / kill criteria

| Metric | WIN | LOSS | INCONCLUSIVE |
|---|---|---|---|
| c2c wall-clock speedup at N=200 | ≥ 3× | < 2× | [2×, 3×) |
| c2c N=20 same-extracted-answer rate vs legacy at temp=0 | ≥ 18/20 | < 16/20 | — |
| `cmajc-v3` N=200 accuracy | ∈ [80.0%, 85.0%] | outside [80.0%, 85.0%] | — |

INCONCLUSIVE → keep `c2c`-only path; do not promote to `cmajc` until S0+S1+S2 stable.
LOSS on accuracy → revert; investigate τ + KV-cache approximation correctness.

## Measurement plan

1. **N=20 same-answer rate at temp=0:** deterministic baseline (no Gumbel
   noise). For each of 20 problems, compare legacy `pred` vs Fast-dLLM
   `pred` after `grade.extract_answer`. Pre-reg targets ≥18/20 match.
   Tolerable disagreement is on problems where confidence is genuinely
   borderline; >2/20 disagreement signals the cache is degrading the
   *content* of the CoT, not just the wall-clock.
2. **Wall-clock:** the per-problem `wallclock_ms` column landed in S0 —
   used as-is.
3. **N=200 accuracy:** binom-CI on `cmajc-v3` after promotion.
4. **τ sweep on N=20** (informal): try τ ∈ {0.85, 0.90, 0.95}; pick max
   τ such that same-answer rate ≥18/20.

## Files modified

**Source:**
- New: `e4/fast_dllm_adapter.py` — thin shim that imports the upstream
  repo from `FAST_DLLM_PATH` (env), wraps the loaded LLaDA model in
  Fast-dLLM v1's KV-cache class, and exposes
  `wrap_for_fast_dllm(model, tokenizer)` + `parallel_decode_step(model, x, threshold, ...)`.
- `e4/diff_llada.py:_ensure_loaded` — when `FAST_DLLM=1`, after PEFT
  adapters are loaded, call `wrap_for_fast_dllm`.
- `e4/diff_llada.py:_generate` — when `FAST_DLLM=1`, replace the
  per-step body (lines 550–589) with the Fast-dLLM `parallel_decode_step`
  call. Keep legacy path as fallback when env var is off.

**Tests:**
- `phase2/inference_viz/test_backcompat.py` — must still pass when
  `FAST_DLLM=0` (default). Add an explicit assertion that the env var
  isn't accidentally set during the fixture run.

**Setup:**
- `phase2/spikes/fast-dllm-c2c/SETUP.md` — recipe: clone
  `NVlabs/Fast-dLLM`, set `FAST_DLLM_PATH` to its checkout, install
  any extra Python deps the upstream lists.

## Compute envelope

- N=20 c2c × 3 τ values × 2 (FAST_DLLM=0 vs 1) = 12 short cells ≈ 30 min ≈ **$0.10**.
- N=200 c2c WIN-confirm at chosen τ ≈ 30 min ≈ **$0.10**.
- N=200 cmajc-v3 (post-promotion) with FAST_DLLM=1 + S0+S1 batched
  ≈ 15 min (multiplicative speedup kicks in here) ≈ **$0.05**.
- Total: **≤$5** including spin-up overhead and τ tuning.

## Rollback / safety

- The whole feature is gated behind `FAST_DLLM=1`. Reverting is
  setting the env to `0` (or leaving unset). The legacy code path is
  unchanged.
- If `e4/fast_dllm_adapter.py`'s import fails (upstream not on the
  path), `_ensure_loaded` raises a clear `ImportError` pointing at
  `FAST_DLLM_PATH` setup.

## Falsification artifacts

- `phase2/spikes/fast-dllm-c2c/wallclock_n20_baseline.jsonl`
- `phase2/spikes/fast-dllm-c2c/wallclock_n20_fastdllm_tau{0.85,0.90,0.95}.jsonl`
- `phase2/spikes/fast-dllm-c2c/n20_same_answer_rate.md` — pairwise comparison.
- `phase2/spikes/fast-dllm-c2c/RESULT.md`

## Notes

- Fast-dLLM v1's claimed speedup on LLaDA-8B is up to 27.6× at GSM8K-512.
  Our shapes are smaller (gen_length=128) so realistic ceiling is closer
  to 5–12× per the kernel survey §3.1.
- This spike multiplies *with* S0 (branch batching) — running cmajc-v3
  with FAST_DLLM=1 + S0 batched should compound to ≥10× wall-clock vs
  pre-spike sequential / un-cached. That's the headline target.
