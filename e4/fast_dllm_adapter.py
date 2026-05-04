"""Fast-dLLM v1 adapter shim — gated behind FAST_DLLM=1 env var.

Imports the upstream NVlabs/Fast-dLLM repo from `FAST_DLLM_PATH` and
exposes a thin wrapper that the sfumato `e4/diff_llada.py:_ensure_loaded`
and `_generate` paths can call without taking a hard dependency.

Upstream repo: https://github.com/NVlabs/Fast-dLLM (Apache-2.0)
Paper: arXiv:2505.22618 (ICLR 2026)

# Setup

```
git clone https://github.com/NVlabs/Fast-dLLM /path/to/Fast-dLLM
export FAST_DLLM_PATH=/path/to/Fast-dLLM
export FAST_DLLM=1
```

The upstream provides (as of paper version) a wrapper that exposes
KV-cache + confidence-aware parallel decoding for LLaDA-8B-Instruct.
The exact symbol names are version-dependent — that's why this shim
keeps the upstream import inside try/except and raises a precise
ImportError when the path isn't set.

The shim currently exposes two thin functions:
  - wrap_for_fast_dllm(model, tokenizer) -> wrapped_model
  - parallel_decode_step(wrapped_model, x, threshold, blk_start, blk_end)
    -> (committed_mask, x_after)

The `_generate` callsite uses these only when FAST_DLLM=1 — the legacy
path stays bit-identical when the env var is off.
"""

from __future__ import annotations

import os
import sys
from typing import Any


_UPSTREAM_LOADED = False
_UPSTREAM_MOD = None


def _ensure_upstream_on_path() -> None:
    """Inject FAST_DLLM_PATH at the head of sys.path if set."""
    global _UPSTREAM_LOADED, _UPSTREAM_MOD
    if _UPSTREAM_LOADED:
        return
    path = os.environ.get("FAST_DLLM_PATH")
    if not path:
        raise ImportError(
            "FAST_DLLM=1 set but FAST_DLLM_PATH is not. Clone "
            "https://github.com/NVlabs/Fast-dLLM and export "
            "FAST_DLLM_PATH=/path/to/Fast-dLLM"
        )
    if not os.path.isdir(path):
        raise ImportError(
            f"FAST_DLLM_PATH={path} is not a directory; clone NVlabs/Fast-dLLM there."
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    # Try the most likely entry-point names. Adjust at integration time
    # once the upstream layout is known on the pod.
    try:
        # Speculative — verify against the upstream README + module list.
        import fast_dllm as _mod  # type: ignore
        _UPSTREAM_MOD = _mod
    except ImportError as e:
        raise ImportError(
            f"FAST_DLLM_PATH={path} loaded but upstream module 'fast_dllm' "
            f"not importable. Verify the package layout. Underlying: {e}"
        )
    _UPSTREAM_LOADED = True


def is_enabled() -> bool:
    return os.environ.get("FAST_DLLM", "0") == "1"


def wrap_for_fast_dllm(model: Any, tokenizer: Any) -> Any:
    """Wrap an HF-loaded LLaDA model with Fast-dLLM's KV-cache + parallel decoder.

    Returns a model-like object that supports the upstream's parallel
    decoding API. On error, raises ImportError so `_ensure_loaded` can
    surface a useful message.
    """
    _ensure_upstream_on_path()
    assert _UPSTREAM_MOD is not None
    # Speculative API call. The upstream (per the paper) exposes a
    # function or class named something like `LLaDAModelWithKVCache` or
    # `wrap_llada_with_kv_cache`. The exact symbol must be chosen on
    # first integration on the pod.
    if hasattr(_UPSTREAM_MOD, "LLaDAModelWithKVCache"):
        return _UPSTREAM_MOD.LLaDAModelWithKVCache(model, tokenizer)
    if hasattr(_UPSTREAM_MOD, "wrap_llada"):
        return _UPSTREAM_MOD.wrap_llada(model, tokenizer)
    raise ImportError(
        "Upstream Fast-dLLM module loaded but neither LLaDAModelWithKVCache "
        "nor wrap_llada are exported. Inspect the upstream repo's __init__.py "
        "and pin the correct symbol here."
    )


def parallel_decode_step(
    wrapped_model: Any,
    x: Any,
    threshold: float,
    blk_start: int,
    blk_end: int,
) -> tuple[Any, Any]:
    """One Fast-dLLM v1 confidence-aware parallel decode step.

    Returns (committed_mask, x_after) where committed_mask is a bool
    tensor over the gen window indicating which positions were
    committed this step, and x_after is the updated sequence tensor.

    The exact upstream signature is version-dependent; see the
    paper-pinned blog post and repo README.
    """
    _ensure_upstream_on_path()
    assert _UPSTREAM_MOD is not None
    if hasattr(wrapped_model, "parallel_decode_with_kv_cache"):
        return wrapped_model.parallel_decode_with_kv_cache(
            x, threshold=threshold, blk_start=blk_start, blk_end=blk_end
        )
    if hasattr(_UPSTREAM_MOD, "parallel_decode_step"):
        return _UPSTREAM_MOD.parallel_decode_step(
            wrapped_model, x, threshold, blk_start, blk_end
        )
    raise ImportError(
        "Upstream Fast-dLLM does not expose a parallel_decode hook on the "
        "wrapped model. Check the integration point in the upstream README."
    )
