"""Phase P.0 — nnsight wrap around CompositeLM.

Returns an nnsight.NNsight handle that exposes the backbone blocks,
the two heads, and the residual stream at every layer for hooking +
activation cache.

Usage:
    from e5.interp.load_model import load_composite_for_interp
    nn_model, raw_model, cfg = load_composite_for_interp(
        "e5/results/f10_mixed/composite/model_slim_final.pt",
        device="mps")
    with nn_model.trace("Question: ..."):
        h = nn_model.blocks[10].output.save()
"""
from __future__ import annotations

from pathlib import Path

import torch

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID


def load_composite_for_interp(ckpt_path: Path | str, device: str = "cpu"):
    """Load a CompositeLM checkpoint and wrap it with nnsight.

    Returns:
      nn_model:   nnsight.NNsight handle (use for tracing)
      raw_model:  the underlying CompositeLM (for direct calls)
      cfg:        CompositeConfig
    """
    import nnsight

    ck = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    cfg = CompositeConfig(**{
        k: ck["config"][k]
        for k in CompositeConfig.__dataclass_fields__
        if k in ck["config"]
    })
    raw_model = CompositeLM(cfg)
    raw_model.load_state_dict(ck["state_dict"])
    raw_model.to(device)
    raw_model.train(False)
    nn_model = nnsight.NNsight(raw_model)
    return nn_model, raw_model, cfg


def architecture_summary(cfg: CompositeConfig) -> str:
    """One-line description of the model architecture for sanity prints."""
    return (
        f"CompositeLM(d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads}, "
        f"V={cfg.vocab_size}, ctx={cfg.block_size}, "
        f"head_diff_proj={cfg.head_diff_proj}, "
        f"use_xsa={getattr(cfg, 'use_xsa', False)})"
    )


__all__ = ["load_composite_for_interp", "architecture_summary", "MASK_TOKEN_ID"]
