"""T0 — composite AR + diffusion model.

nanoGPT-style transformer with two output heads sharing a single backbone:
  - head_ar: causal-masked, predicts next token (AR mode)
  - head_diff: bidirectional-masked, predicts masked tokens (mask-diffusion
    mode, MDLM-style)

Mode is selected per forward call by the `mode` kwarg ("ar" or "diff"),
which sets the attention mask and the active head.

Architectural choices (justified in /Users/eren/.claude/plans/...rosy-jellyfish.md):
  - vocab = GPT-2 BPE 50257 (standard, cheap, well-tokenized for English math)
  - d_model = 512, n_layers = 8, n_heads = 8 → ~50M params (close to the
    60M T0 target; tied input/output embeddings keep it small)
  - block_size = 256 (fits GSM8K problems + CoT)
  - tied token embedding ↔ AR head (standard)
  - head_diff = 1 linear projection (d → d) + tied embedding output, so the
    backbone learns *one* representation that two heads consume slightly
    differently

For the B3 paired-separate baseline (two 30M models), instantiate
`CompositeLM(d_model=384, n_layers=6)` twice — once trained AR-only, once
diff-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- mask id (MDLM convention: the last vocab slot) ---------------------
# We extend the GPT-2 vocab by 1 to hold a dedicated [MASK] token.
GPT2_VOCAB = 50257
MASK_TOKEN_ID = GPT2_VOCAB  # 50257 ; vocab_size becomes 50258


@dataclass
class CompositeConfig:
    vocab_size: int = GPT2_VOCAB + 1  # +1 for [MASK]
    block_size: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    head_diff_proj: bool = True  # learn a d_model→d_model projection before the diff head's tied-output
    use_xsa: bool = False  # Exclusive Self Attention (arxiv:2603.09078): forbid attending to own position


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a runtime-selectable mask.

    mode="ar" → causal lower-triangular mask.
    mode="diff" → no mask (bidirectional).

    Implemented via torch.nn.functional.scaled_dot_product_attention with
    is_causal flag toggled.
    """

    def __init__(self, cfg: CompositeConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.use_xsa = cfg.use_xsa

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, nh, T, hd)
        is_causal = mode == "ar"
        attn_mask = None
        if self.use_xsa:
            # XSA: build a no-self-attn mask (diagonal = -inf, off-diagonal = 0)
            eye = torch.eye(T, device=x.device, dtype=torch.bool)
            attn_mask = torch.zeros(T, T, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(eye, float("-inf"))
            if is_causal:
                # SDPA cannot combine attn_mask + is_causal=True, so we bake the
                # causal triangle into attn_mask explicitly here.
                causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                                    diagonal=1)
                attn_mask.masked_fill_(causal, float("-inf"))
                is_causal = False
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: CompositeConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: CompositeConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mode)
        x = x + self.mlp(self.ln2(x))
        return x


class CompositeLM(nn.Module):
    """A single transformer with two output heads.

    Forward call selects mode:
      - mode="ar":   causal attention, returns next-token logits over vocab
      - mode="diff": bidirectional attention, returns logits over vocab at
                     every position (caller picks masked positions)

    The `head_diff_proj` flag controls whether the diffusion head has its
    own d→d projection before the tied output (recommended ON: lets the
    backbone learn one shared representation that each mode reshapes).
    """

    def __init__(self, cfg: CompositeConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Heads. The AR head is tied to tok_emb (standard). The diff head
        # has an optional pre-projection, then ties to tok_emb.
        if cfg.head_diff_proj:
            self.head_diff_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        else:
            self.head_diff_proj = nn.Identity()

        # Weight init (GPT-2 style).
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc.weight") or pn.endswith("qkv.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, mode: str) -> torch.Tensor:
        """idx: (B, T) long tensor of token ids. Returns logits (B, T, vocab)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"context {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, mode=mode)
        x = self.ln_f(x)
        if mode == "ar":
            logits = x @ self.tok_emb.weight.T
        elif mode == "diff":
            h = self.head_diff_proj(x)
            logits = h @ self.tok_emb.weight.T
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ar_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Standard shifted next-token cross-entropy.
    logits: (B, T, V), targets: (B, T) — targets at position t already
    correspond to "predict the (t+1)-th token from positions [0..t]".
    Caller is responsible for the shift.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def diff_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_positions: torch.Tensor,
) -> torch.Tensor:
    """MDLM-style mask-prediction loss.
    Only positions in `masked_positions` (bool, (B, T)) contribute.
    targets: (B, T) original (unmasked) ids.
    """
    B, T, V = logits.shape
    mask_flat = masked_positions.reshape(-1)
    if not mask_flat.any():
        return logits.new_zeros((), requires_grad=True)
    logits_flat = logits.reshape(B * T, V)[mask_flat]
    targets_flat = targets.reshape(-1)[mask_flat]
    return F.cross_entropy(logits_flat, targets_flat)


def apply_mask(
    idx: torch.Tensor,
    mask_ratio_per_seq: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row, mask ~mask_ratio[i] fraction of positions.
    Returns (idx_masked, masked_positions_bool).
    """
    B, T = idx.shape
    u = torch.rand(B, T, device=idx.device)
    threshold = mask_ratio_per_seq.unsqueeze(1).expand(-1, T)
    masked = u < threshold
    idx_masked = torch.where(masked, torch.full_like(idx, mask_token_id), idx)
    return idx_masked, masked


def sample_mask_ratios(B: int, device, mode: str = "uniform") -> torch.Tensor:
    """Sample per-sequence mask ratios.
    mode="uniform": ~Uniform[0.01, 0.99] (MDLM-style).
    mode="midrange": ~Uniform[0.30, 0.70] (less variance, faster signal).
    mode="span_uniform" / "span_midrange": same ratio sampling; the
        caller is expected to dispatch to `apply_span_mask` instead of
        `apply_mask`. The mode tag is informational only.
    """
    if mode in ("uniform", "span_uniform"):
        return 0.01 + 0.98 * torch.rand(B, device=device)
    if mode in ("midrange", "span_midrange"):
        return 0.30 + 0.40 * torch.rand(B, device=device)
    raise ValueError(mode)


def apply_span_mask(
    idx: torch.Tensor,
    mask_ratio_per_seq: torch.Tensor,
    span_mean_length: int = 4,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """T5/SpanBERT-style span masking.

    For each sequence, walk left-to-right; at each position with
    probability p_start = mask_ratio / span_mean_length, start a masked
    span of geometric length (mean = span_mean_length). Skip ahead by
    the span length to avoid overlaps.

    Composite-specific motivation: forces the diff head to learn
    in-context completion rather than independent-token mask-fill,
    which matches downstream FIM-shaped evaluation (HumanEval-Infill,
    SantaCoder-FIM) directly. Span masking is the data lever D2 from
    the situation report at /Users/eren/.claude/plans/...crane.md.

    Returns (idx_masked, masked_positions_bool) with the same shape
    contract as `apply_mask`.
    """
    B, T = idx.shape
    p_start = (mask_ratio_per_seq / float(span_mean_length)).clamp(0.0, 1.0)  # (B,)
    masked = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
    # Sample span starts independently per position
    u = torch.rand(B, T, device=idx.device)
    starts = u < p_start.unsqueeze(1)  # (B, T)
    # Sample geometric span lengths (mean span_mean_length, min 1)
    # Use torch.empty(...).geometric_(p): p = 1/mean; returns k>=1 with mean=1/p
    p = 1.0 / float(span_mean_length)
    # Pre-sample lengths for all positions; only the ones at span starts are used
    lengths = torch.empty(B, T, device=idx.device).geometric_(p).clamp(max=T).long()
    # Per-batch left-to-right walk; B and T are small at training time so the
    # python loop is not a bottleneck.
    for b in range(B):
        i = 0
        while i < T:
            if starts[b, i]:
                L = min(int(lengths[b, i].item()), T - i)
                if L < 1:
                    L = 1
                masked[b, i:i + L] = True
                i += L
            else:
                i += 1
    idx_masked = torch.where(masked, torch.full_like(idx, mask_token_id), idx)
    return idx_masked, masked


def apply_position_biased_mask(
    idx: torch.Tensor,
    mask_ratio_per_seq: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anti-AR-biased mask placement: P(mask|pos) proportional to pos/T.

    The first position is almost never masked; the last position is
    masked at ~2x the target rate. Average mask ratio per sequence still
    equals `mask_ratio_per_seq[i]` by construction.

    Composite-specific motivation: AR's causal attention sees no future
    context, so the suffix is structurally where AR is weakest. Biasing
    the diff-head's training signal toward suffix positions forces it
    to specialise on the positions AR can't help with --- amplifying
    the disjoint-circuit specialisation we measure in Phase P.2. Data
    lever D1 from the situation report.

    Returns (idx_masked, masked_positions_bool).
    """
    B, T = idx.shape
    # pos_weight[i] = 2*(i+1)/(T+1) so its mean over i is 1.0
    arange = torch.arange(1, T + 1, device=idx.device, dtype=torch.float32)
    pos_weight = 2.0 * arange / float(T + 1)  # (T,), mean=1.0
    threshold = mask_ratio_per_seq.unsqueeze(1) * pos_weight.unsqueeze(0)  # (B, T)
    threshold = threshold.clamp(0.0, 1.0)
    u = torch.rand(B, T, device=idx.device)
    masked = u < threshold
    idx_masked = torch.where(masked, torch.full_like(idx, mask_token_id), idx)
    return idx_masked, masked
