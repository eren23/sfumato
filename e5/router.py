"""Phase I.3 — RouterMLP: learned mode-router on top of frozen composite.

A tiny MLP head that reads the final-block hidden state at the last
position of a chunk and chooses among {ar_chunk, diff_short, diff_long,
end}. Trained via REINFORCE with the rollout reward from gold-CoT NLL +
loop-rate penalty + AR coherence shaping (see e5/train_router.py).

The composite backbone (model_composite.CompositeLM) is frozen at F11
weights; only the router's parameters update.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Action ids — keep in sync with e5/train_router.py and any inference driver
ACTION_AR_CHUNK = 0     # extend AR by k_ar tokens
ACTION_DIFF_SHORT = 1   # diff-fill the next k_diff_short tokens
ACTION_DIFF_LONG = 2    # diff-fill the next k_diff_long tokens
ACTION_END = 3          # stop generating (only if rollout length allows)
N_ACTIONS_DEFAULT = 4


@dataclass
class RouterConfig:
    d_model: int = 1024
    hidden: int = 512    # 2-layer MLP hidden width; d_model // 2 by default
    n_actions: int = N_ACTIONS_DEFAULT
    dropout: float = 0.0  # tiny model, no dropout by default


class RouterMLP(nn.Module):
    """2-layer MLP that turns a hidden state into action logits.

    Input: hidden state (B, d_model) at the LAST position of the current
    chunk. Output: action logits (B, n_actions). Inference uses
    `torch.distributions.Categorical` over a temperature-scaled softmax.
    """

    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.d_model, cfg.hidden, bias=True)
        self.fc2 = nn.Linear(cfg.hidden, cfg.n_actions, bias=True)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        # Small init so initial policy is near-uniform
        for m in (self.fc1, self.fc2):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(h_last))
        x = self.dropout(x)
        return self.fc2(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def extract_last_hidden(model, idx: torch.Tensor, mode: str = "ar") -> torch.Tensor:
    """Run model.forward up to ln_f, return the (B, d_model) hidden state
    at the last position. Used both for routing decisions and for AR
    coherence shaping in REINFORCE.

    NB: this duplicates the forward pass through blocks. If we run router
    training at scale, refactor CompositeLM to expose `hidden_only=True`
    rather than recomputing.
    """
    B, T = idx.shape
    assert T <= model.cfg.block_size
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    x = model.drop(x)
    for block in model.blocks:
        x = block(x, mode=mode)
    x = model.ln_f(x)
    return x[:, -1, :]  # (B, d_model)
