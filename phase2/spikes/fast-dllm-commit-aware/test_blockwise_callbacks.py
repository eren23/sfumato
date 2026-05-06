"""Unit test for fast_dllm_adapter.fast_dllm_generate_blockwise.

Mocks the upstream Fast-dLLM `generate` symbol so we can verify
per-block callback ordering, growth of x, and accumulator semantics
without needing a real LLaDA-8B load. Verifies T1.C (commit-LoRA-aware
Fast-dLLM port) plumbing.

Usage:
    python phase2/spikes/fast-dllm-commit-aware/test_blockwise_callbacks.py

Pass criteria:
- on_block_start fires before each upstream generate call
- on_block_end fires after each upstream generate call
- callbacks fire exactly num_blocks times each, in 0..num_blocks-1 order
- nfe accumulator sums upstream's per-call counter
- final x has shape (1, prompt_len + num_blocks*block_length)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # type: ignore

from e4 import fast_dllm_adapter as _fdll


class _FakeTensor:
    """Minimal numpy-backed tensor that supports `.shape[1]` indexing."""

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, idx):
        return _FakeTensor(self.arr[idx])

    def __repr__(self):
        return f"_FakeTensor(shape={self.shape})"


def main() -> None:
    # Mock upstream generate: receives x of shape (1, L), returns
    # (x_full of shape (1, L + gen_length), nfe).
    n_calls = 0
    nfe_per_call = 7

    def fake_generate(model, prompt_ids, *, steps, gen_length, block_length,
                      temperature, remasking, mask_id, threshold, factor):
        nonlocal n_calls
        n_calls += 1
        # Append `gen_length` zero tokens to the prompt.
        new_arr = np.concatenate(
            [prompt_ids.arr, np.zeros((1, gen_length), dtype=int)], axis=1
        )
        return (_FakeTensor(new_arr), nfe_per_call)

    # Patch upstream symbols.
    _fdll._UPSTREAM_LOADED = True
    _fdll._GENERATE_FN = fake_generate
    _fdll._LLADA_MODEL_CLS = type("FakeModelCls", (), {})

    # Track callback order.
    events = []

    def on_start(blk):
        events.append(("start", blk))

    def on_end(blk):
        events.append(("end", blk))

    # Run.
    prompt_len = 5
    num_blocks = 4
    block_length = 32
    fake_prompt = _FakeTensor(np.arange(prompt_len, dtype=int).reshape(1, -1))

    x_full, total_nfe = _fdll.fast_dllm_generate_blockwise(
        model="dummy",
        prompt_ids=fake_prompt,
        steps_per_block=16,
        block_length=block_length,
        num_blocks=num_blocks,
        temperature=0.7,
        on_block_start=on_start,
        on_block_end=on_end,
        threshold=0.9,
    )

    # Assertions.
    assert n_calls == num_blocks, f"expected {num_blocks} upstream calls, got {n_calls}"
    expected_events = []
    for blk in range(num_blocks):
        expected_events.append(("start", blk))
        expected_events.append(("end", blk))
    assert events == expected_events, f"events mismatch:\n  got: {events}\n  exp: {expected_events}"
    assert total_nfe == num_blocks * nfe_per_call, f"nfe sum mismatch: got {total_nfe}, exp {num_blocks * nfe_per_call}"
    expected_shape = (1, prompt_len + num_blocks * block_length)
    assert x_full.shape == expected_shape, f"shape mismatch: got {x_full.shape}, exp {expected_shape}"

    # Test None callbacks don't crash.
    n_calls = 0
    events = []
    x2, _ = _fdll.fast_dllm_generate_blockwise(
        model="dummy",
        prompt_ids=fake_prompt,
        steps_per_block=16,
        block_length=block_length,
        num_blocks=num_blocks,
        temperature=0.7,
        on_block_start=None,
        on_block_end=None,
    )
    assert n_calls == num_blocks
    assert events == []  # no callbacks fired
    assert x2.shape == expected_shape

    print("all tests passed:")
    print(f"  - {num_blocks} upstream generate calls")
    print(f"  - callback order: start/end alternation across {num_blocks} blocks")
    print(f"  - nfe accumulator sums correctly ({total_nfe} = {num_blocks} × {nfe_per_call})")
    print(f"  - final x shape: {x_full.shape}")
    print(f"  - None callbacks gracefully skipped")


if __name__ == "__main__":
    main()
