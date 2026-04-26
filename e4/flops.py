"""FLOPs accountant. Kaplan-style approximation; conservative.

For dense transformers: forward ≈ 2 * N_params per token (Hoffmann et al. 2022).
For diffusion LMs operating on a fixed block: forward ≈ 2 * N_params per token,
applied k_steps times to the whole block.

Numbers are approximate; the goal is consistent FLOPs accounting *across*
conditions, not absolute calibration. Calibration is doable later by replacing
the per-model constant with a measured forward-pass FLOPs from
`torch.profiler` or `fvcore`.
"""

from __future__ import annotations

# Approximate parameter counts per published model. Used in the per-token
# forward-FLOPs approximation forward_flops ≈ 2 * N_params.
_KNOWN_PARAMS: dict[str, int] = {
    "Qwen/Qwen2.5-0.5B-Instruct": 494_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_540_000_000,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1_710_000_000,
    "GSAI-ML/LLaDA-1.5B-Instruct": 1_500_000_000,
    "diffusionfamily/diffullama": 1_000_000_000,
}


def _params(name: str) -> int:
    if name in _KNOWN_PARAMS:
        return _KNOWN_PARAMS[name]
    # Conservative default: 1B params.
    return 1_000_000_000


def qwen_forward_flops(name: str, n_tokens: int) -> int:
    """AR forward: 2 * N_params per token decoded."""
    return 2 * _params(name) * max(n_tokens, 0)


def llada_forward_flops(name: str, n_tokens: int, n_steps: int) -> int:
    """Diffusion forward over a block of n_tokens, repeated n_steps times.

    Each denoising step is a full forward over the block, so total =
    n_steps * 2 * N_params * n_tokens.
    """
    return n_steps * 2 * _params(name) * max(n_tokens, 0)
