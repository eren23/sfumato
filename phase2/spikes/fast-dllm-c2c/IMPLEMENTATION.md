# S4 — Implementation log

**Date:** 2026-05-04
**Status:** scaffold landed; upstream symbol pinning + pod validation pending.

## What landed

### Adapter shim (offline-pinned, speculative on upstream API)

- New: `e4/fast_dllm_adapter.py` — module that gates on
  `FAST_DLLM=1` env var and, when set, injects `FAST_DLLM_PATH` into
  `sys.path` and imports the upstream NVlabs/Fast-dLLM module. Two
  thin functions:
  - `wrap_for_fast_dllm(model, tokenizer)` — tries
    `LLaDAModelWithKVCache(model, tokenizer)` first, falls back to
    `wrap_llada(model, tokenizer)`, else raises a precise `ImportError`.
  - `parallel_decode_step(wrapped_model, x, threshold, blk_start, blk_end)`
    — tries wrapped_model.parallel_decode_with_kv_cache first, falls
    back to module-level function.
- `is_enabled()` gate — used by `_ensure_loaded` to avoid loading the
  upstream when default `FAST_DLLM=0`.

### Env-gated wrap in `_ensure_loaded`

- `e4/diff_llada.py:_ensure_loaded` (after PEFT adapters loaded):
  ```
  from e4 import fast_dllm_adapter as _fdll
  if _fdll.is_enabled():
      try:
          model = _fdll.wrap_for_fast_dllm(model, self._tokenizer)
      except ImportError as _e:
          raise RuntimeError(...) from _e
  ```
  Hard-fails (rather than silently falling back) when `FAST_DLLM=1` is
  set but the upstream wrap fails — this surfaces config errors instead
  of running un-accelerated and producing misleading speedup numbers.

### What is NOT yet wired

- The **per-step body in `_generate`** is unchanged. The wrapped
  model is loaded but the denoise loop still uses the legacy path. The
  `parallel_decode_step` call must be wired into lines 550–589 of
  `_generate` once the upstream API signature is verified on a pod.
- The legacy `denoise_block_batched` / `_generate_batched` path is
  also unchanged. Wiring Fast-dLLM into the batched path is the
  *headline* multiplicative win (S0 × S4) and should follow the
  single-row wiring once that's validated.

### Why scaffold-only

Fast-dLLM v1's exact upstream API symbol names are version-dependent.
This shim pins two best-guess names with explicit fallbacks so that the
first pod-side run will either work immediately or raise a precise
`ImportError` naming the upstream path. Pinning the wrong symbol now
would create silent bit-rot the first time the upstream releases a
breaking change.

The proper full integration is **3–5 eng-days** per the kernel-survey
budget, of which this offline session contributes ~0.5 eng-days
(scaffold + env gate + pre-reg + setup recipe). The remaining work
(per-step body wiring + accuracy/wallclock validation) requires a pod.

### Backcompat

- `phase2/inference_viz/test_backcompat.py` PASS 3/3 with `FAST_DLLM`
  unset. The legacy denoise path is unchanged when the env gate is off.
- Mock smoke (cmaj, cmajc) PASS with `FAST_DLLM` unset.
- Adapter import test confirms the gate fails cleanly with
  `ImportError` when `FAST_DLLM=1` but `FAST_DLLM_PATH` is unset.

## What's pending

1. **Pin upstream symbols.** First pod-side run inspecting
   `dir(fast_dllm)` to confirm whether the entry-point class is
   `LLaDAModelWithKVCache`, `wrap_llada`, or something else.
2. **Wire `_generate` per-step body.** Replace lines 550–589 with the
   `parallel_decode_step` call when `FAST_DLLM=1`.
3. **Wire `_generate_batched`** for the cmajc multiplicative win.
4. **τ sweep** on N=20 — pick the threshold that preserves
   `c2c` extracted-answer agreement ≥ 18/20.
5. **N=200 WIN-confirm** on c2c (and cmajc post-promotion).
6. **RESULT.md** — paired wall-clock + accuracy CIs.

See `SETUP.md` for the pod recipe.
