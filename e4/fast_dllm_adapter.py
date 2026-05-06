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


_LLADA_MODEL_CLS = None
_GENERATE_FN = None


def _ensure_upstream_on_path() -> None:
    """Inject Fast-dLLM v1/llada and v1/llada/model into sys.path.

    Upstream layout (verified via gh API on commit @ 2026-05-04):
      Fast-dLLM/
      ├── v1/llada/generate.py   → exports `generate(model, prompt, steps=..., gen_length=..., block_length=..., temperature=..., threshold=..., factor=...)`
      ├── v1/llada/model/modeling_llada.py → `LLaDAModelLM` (custom HF subclass with KV-cache support)
      └── v2/                    → block-diffusion variant (not used here)

    Both paths must be on sys.path because generate.py does
    `from model.modeling_llada import LLaDAModelLM` (relative-style with
    no leading dot, requires `v1/llada` on path).
    """
    global _UPSTREAM_LOADED, _LLADA_MODEL_CLS, _GENERATE_FN
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
    llada_root = os.path.join(path, "v1", "llada")
    if not os.path.isdir(llada_root):
        raise ImportError(
            f"FAST_DLLM_PATH={path} does not contain v1/llada/. "
            f"Pull the latest upstream main."
        )
    # Prepend the LLaDA-specific paths.
    for p in (llada_root, os.path.join(llada_root, "model")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from model.modeling_llada import LLaDAModelLM  # type: ignore
        _LLADA_MODEL_CLS = LLaDAModelLM
    except ImportError as e:
        raise ImportError(
            f"FAST_DLLM_PATH={path}/v1/llada loaded but `model.modeling_llada.LLaDAModelLM` "
            f"is not importable. Underlying: {e}"
        )
    try:
        from generate import generate as _g  # type: ignore
        _GENERATE_FN = _g
    except ImportError as e:
        raise ImportError(
            f"FAST_DLLM_PATH={path}/v1/llada loaded but `generate.generate` "
            f"is not importable. Underlying: {e}"
        )
    _UPSTREAM_LOADED = True


def is_enabled() -> bool:
    return os.environ.get("FAST_DLLM", "0") == "1"


def load_fast_dllm_model(name: str, dtype, device, hf_token=None):
    """Load LLaDA via Fast-dLLM's `LLaDAModelLM.from_pretrained` instead of HF AutoModel.

    Drop-in replacement for `AutoModel.from_pretrained(name, trust_remote_code=True)`
    that wires the LLaDA modeling code with KV-cache + parallel-decoding hooks.
    """
    _ensure_upstream_on_path()
    assert _LLADA_MODEL_CLS is not None
    kwargs = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token
    model = _LLADA_MODEL_CLS.from_pretrained(name, **kwargs)
    if "cuda" in str(device):
        model = model.to("cuda")
    model.requires_grad_(False)
    return model


def fast_dllm_generate(
    model,
    prompt_ids,                 # (1, L)
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    threshold: float | None = None,
    factor: float | None = None,
    mask_id: int = 126336,
    remasking: str = "low_confidence",
):
    """Call Fast-dLLM v1's `generate()` end-to-end.

    Upstream returns `(x_full, nfe)` where x_full is (1, L+gen_length) and
    nfe is the number-of-forward-evaluations counter. Returns the tensor.
    """
    _ensure_upstream_on_path()
    assert _GENERATE_FN is not None
    out = _GENERATE_FN(
        model,
        prompt_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        threshold=threshold,
        factor=factor,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def fast_dllm_generate_blockwise(
    model,
    prompt_ids,                 # (1, L)
    steps_per_block: int,
    block_length: int,
    num_blocks: int,
    temperature: float,
    on_block_start=None,        # Callable[[int blk_idx], None] | None
    on_block_end=None,          # Callable[[int blk_idx], None] | None
    threshold: float | None = None,
    factor: float | None = None,
    mask_id: int = 126336,
    remasking: str = "low_confidence",
):
    """Sub-block-aware Fast-dLLM wrapper for commit-LoRA-toggled inference.

    Iterates upstream `generate()` once per sub-block. Between blocks,
    invokes the on_block_start/on_block_end callbacks so the caller can
    toggle a PEFT adapter (e.g. commit-LoRA) via merge/unmerge at the
    schedule boundary identified by the K2 ablation result
    (sub-block-1 boundary, see phase2/spikes/k2-commit-blocks-ablation).

    The cumulative gen_length matches the one-shot caller:
        total_gen = num_blocks * block_length

    Returns the full sequence (1, L + total_gen). nfe-style counters from
    upstream are summed in `_BLOCKWISE_NFE_TOTAL` (reset on each call).
    """
    _ensure_upstream_on_path()
    assert _GENERATE_FN is not None
    x = prompt_ids
    total_nfe = 0
    for blk in range(num_blocks):
        if on_block_start is not None:
            on_block_start(blk)
        out = _GENERATE_FN(
            model,
            x,
            steps=steps_per_block,
            gen_length=block_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            threshold=threshold,
            factor=factor,
        )
        if isinstance(out, tuple):
            x = out[0]
            if len(out) > 1 and isinstance(out[1], (int, float)):
                total_nfe += int(out[1])
        else:
            x = out
        if on_block_end is not None:
            on_block_end(blk)
    return x, total_nfe
