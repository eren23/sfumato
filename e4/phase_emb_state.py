"""Phase-4 Direction A: shared thread-local for sub-block-aware LoRA-A.

`scripts/train_track2_commit_rl.py` monkey-patches each `lora_A.default`
linear in commit-LoRA's targeted layers to ADD `phase_emb[t]` to the
LoRA-A output, where `t` is the current diffusion sub-block index. The
monkey-patch reads `t` from a thread-local set by the diffusion runner.

This module is the neutral library home for that thread-local so:
  - `e4/diff_llada.py:_Real.denoise_block` can WRITE the index from
    inside the inner forward loop without importing anything from
    `scripts/`.
  - `scripts/train_track2_commit_rl.py` can READ the index from the
    monkey-patched LoRA-A forward without owning the writer side.

Production paths bypass entirely: `set_sub_block` is a no-op unless
`SCHEDULE_RL=1` is set, so paper §3 numbers / T2.A re-runs / etc. pay
exactly one `os.environ.get` per sub-block (negligible).

Thread-local (not a module attr) so concurrent eval workers / data-
loader threads don't trip on each other's `t`.
"""

from __future__ import annotations

import os
import threading

_state = threading.local()


def is_enabled() -> bool:
    """SCHEDULE_RL=1 enables the writer side. Reader is unconditional."""
    return os.environ.get("SCHEDULE_RL", "0") == "1"


def set_sub_block(t: int) -> None:
    """Write the current sub-block index. No-op unless SCHEDULE_RL=1."""
    if is_enabled():
        _state.t = int(t)


def get_sub_block() -> int | None:
    """Read the current sub-block index, or None if unset."""
    return getattr(_state, "t", None)


def clear() -> None:
    """Drop any current sub-block index. Always safe to call."""
    if hasattr(_state, "t"):
        del _state.t


def force_set_sub_block(t: int | None) -> None:
    """Test-only: bypass the SCHEDULE_RL gate.

    Used by unit smokes that want to drive the monkey-patch without
    setting an env var. Production code should use `set_sub_block`.
    """
    if t is None:
        clear()
    else:
        _state.t = int(t)
