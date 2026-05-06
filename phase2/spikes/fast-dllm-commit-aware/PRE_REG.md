# Pre-registration — Fast-dLLM Commit-LoRA-Aware Port

**Date:** 2026-05-06 | **Spike:** Phase-3 T1.C of
`/Users/eren/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`.

## Hypothesis

Wrapping NVlabs/Fast-dLLM v1's `generate()` call once per LLaDA
sub-block (with commit-LoRA toggling between blocks via PEFT
merge/unmerge) reproduces the Phase-2 cmajc accuracy headline
(0.822 mean, σ ≈ 0.85pp) within ±0.05 on N=20 problems while
preserving Fast-dLLM's KV-cache + confidence-aware parallel-decode
speedup. Productizes the K2 inverted-U finding into the Fast-dLLM
fastpath.

## Substrate

- LLaDA-8B-Instruct + prefix-robust-v3 LoRA + commit-v3 LoRA.
- Condition `cmajc`, BRANCHES=5, K_STEPS=64, TEMP=0.7, SEED=0.
- Frozen GSM8K-test idx 0..19 (first 20 problems of
  `e4/data/gsm8k_dev_200.json`).
- Pod: 24GB 4090 spot (Fast-dLLM was already validated at this size
  for c2 / c2c paths in Phase 2).
- Env: `FAST_DLLM=1 FAST_DLLM_PATH=/workspace/Fast-dLLM
  FAST_DLLM_TAU=0.9 COMMIT_N_BLOCKS=3 BATCHED=0`.

## Prediction

| Metric | Phase-2 baseline (no Fast-dLLM) | Predicted with Fast-dLLM commit-aware |
|---|---|---|
| cmajc N=20 acc | 0.80 ± 0.05 | 0.80 ± 0.05 (±1× σ band) |
| wallclock per problem | ~22 s/problem (sequential cmajc, 24GB) | ≤ 6 s/problem (≥4× speedup) |
| commit-LoRA toggles per problem | 2 (enable @ block 1, disable @ block 3) | 2 (same) |

The commit-LoRA toggle adds modest PEFT overhead vs the original
non-toggled c2-Fast-dLLM 6.5× speedup (Phase-2 wandb runs vff3ehdy /
97xmgz1d / gxevocq5). Target speedup is ≥4× (allowing one block's
worth of overhead from the merge/unmerge cycle).

## Decision rules

| Outcome | Verdict |
|---|---|
| acc ∈ [0.75, 0.85] AND speedup ≥4× | **WIN** — productizes commit-LoRA into the Fast-dLLM speedup path |
| acc ∈ [0.75, 0.85] AND speedup ∈ [2×, 4×) | **PARTIAL** — accuracy preserved, speedup capped by toggle overhead |
| acc < 0.75 OR speedup < 2× | **LOSS** — wrapper introduces a regression |

## Method

1. **Static smoke** (DONE): import `fast_dllm_adapter` after refactor;
   confirm `fast_dllm_generate_blockwise` exists and `_generate` in
   `diff_llada.py` calls it when `FAST_DLLM=1` AND `commit_last_block`.
2. **Unit test** (DONE): `test_blockwise_callbacks.py` mocks upstream
   `generate` and verifies per-block callback order, nfe accumulator
   semantics, shape growth, None-callback handling. All 5 assertions
   pass.
3. **Real GPU verify** (this spike): provision 24GB 4090 spot, sync
   code, dispatch cmajc N=20 with FAST_DLLM=1, COMMIT_N_BLOCKS=3.
   Confirm acc + wallclock per pre-reg.

## Anti-goals

- No upstream Fast-dLLM fork — wrap in our shim only.
- No FAST_DLLM=1 + BATCHED=1 path (Fast-dLLM's `generate` is
  per-row; cmajc batched calls multiple branches in one tensor and
  is incompatible without re-implementing the upstream loop).
  Sequential branches are the supported path.
- No N>20 substrate in this spike — N=20 is sufficient to verify
  no regression. Multi-seed N=200 scale-up only if WIN at N=20.
- No measuring against the c2/c2c FAST_DLLM=1 path's 6.5× speedup
  baseline directly — different conditions. Compare to non-FAST_DLLM
  cmajc at the same N=20.

## Cost

~$0.10 (24GB 4090, ~30 min total: pod start + bootstrap + 20 problems).

## Files

- `PRE_REG.md` — this file
- `test_blockwise_callbacks.py` — unit test (passes locally)
- `RESULT.md` — to be filled after real-GPU verify

## Implementation refs

- `e4/fast_dllm_adapter.py:fast_dllm_generate_blockwise` — new function
- `e4/diff_llada.py:_generate` — wires blockwise variant when
  FAST_DLLM=1 AND `commit_last_block=True`
