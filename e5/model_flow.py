"""T0b — toy continuous-flow language model (ELF-flavoured).

A transformer that predicts a velocity field in continuous embedding space.
Rectified-flow / OT-CFM parameterisation: training samples t ~ Uniform[0, 1],
interpolates z_t = (1-t)·z_clean + t·z_noise where z_noise ~ N(0, I), and
trains the model to predict the velocity v = z_clean − z_noise. At inference
we integrate the ODE dz/dt = v_θ(z_t, t) from t=1 (pure noise) to t=0 (clean
embedding) over N Euler steps, then "decode" by looking up the nearest token
via the embedding table.

Differences from the e5 discrete-mask `CompositeLM`:
  - Inputs and outputs are CONTINUOUS in d_model space (not token ids).
  - There is no MASK token. Noising is Gaussian on embedding vectors.
  - The model receives the timestep t as an AdaLN-style modulation signal.

Architectural choices:
  - 60M target via d_model=512, n_layers=8, n_heads=8 (matches T0's
    discrete CompositeLM for a fair comparison).
  - Bidirectional attention (no causal mask) — flow needs to see the
    whole sequence to denoise consistently. This matches the diff_only
    baseline in T0 and lets us compare apples-to-apples.
  - Time conditioning: sinusoidal time embedding → MLP → per-layer
    (scale, shift, gate) triples that modulate the LN outputs of each
    Block. Standard DiT-style AdaLN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GPT2_VOCAB = 50257  # same as model_composite.py for tokenizer compatibility


@dataclass
class FlowConfig:
    vocab_size: int = GPT2_VOCAB
    block_size: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    sigma_min: float = 1e-4  # avoid singular noise at t=0


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal time embedding. t shape: (B,) or (B, 1). Returns (B, dim)."""
    if t.dim() == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(1, half)
    )
    args = t.float().reshape(-1, 1) * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class BiAttention(nn.Module):
    """Bidirectional self-attention. No causal mask."""

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FlowMLP(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class AdaLNBlock(nn.Module):
    """DiT-style block with adaptive LN modulated by time embedding.
    The time embedding produces (scale, shift, gate) for both the attn
    and MLP sublayers.
    """

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = BiAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = FlowMLP(cfg)
        # 6 modulation params: (scale, shift, gate) × 2 sublayers
        self.mod = nn.Linear(cfg.d_model, 6 * cfg.d_model, bias=True)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, d_model). Broadcast over sequence positions.
        scale1, shift1, gate1, scale2, shift2, gate2 = self.mod(t_emb).chunk(6, dim=-1)
        scale1, shift1, gate1 = scale1.unsqueeze(1), shift1.unsqueeze(1), gate1.unsqueeze(1)
        scale2, shift2, gate2 = scale2.unsqueeze(1), shift2.unsqueeze(1), gate2.unsqueeze(1)

        h = self.ln1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        h = self.ln2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


class FlowLM(nn.Module):
    """Continuous-flow language model.

    forward(z_t, t, cond_emb=None) → predicted velocity v(z_t, t) of same
    shape as z_t.

    z_t shape: (B, T, d_model) — noisy continuous embeddings.
    t shape: (B,) — timestep in [0, 1].
    cond_emb (optional): (B, T_prompt, d_model) — prompt embeddings to
    concatenate as a prefix that the model attends to but does NOT
    predict velocity for. Used for conditional generation.
    """

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # Project the noisy continuous input into the same space (identity-init).
        self.input_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.eye_(self.input_proj.weight)
        # Time embedding MLP.
        self.t_embed = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.SiLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        # Velocity head: linear projection back to d_model.
        self.velocity_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.velocity_head.weight)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            if module.weight is not None and module.weight.data.abs().max() < 1e-9:
                # Skip already-zeroed linears (velocity_head, mod)
                if module.bias is not None and module.bias.data.abs().max() < 1e-9:
                    return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z_t: (B, T_gen, d_model). t: (B,) ∈ [0,1].
        prompt_ids (optional): (B, T_prompt) long — prefix tokens used as
        clean conditioning. They are embedded, prepended, and the model
        produces velocities for ALL positions; we return only the
        velocities for the z_t positions.
        Returns velocity of shape (B, T_gen, d_model).
        """
        B, T_gen, D = z_t.shape
        device = z_t.device

        # Build the full sequence: [prompt_emb | z_t]
        if prompt_ids is not None:
            T_p = prompt_ids.shape[1]
            prompt_emb = self.tok_emb(prompt_ids)
        else:
            T_p = 0
            prompt_emb = z_t.new_zeros((B, 0, D))

        x_gen = self.input_proj(z_t)
        x = torch.cat([prompt_emb, x_gen], dim=1)
        T_total = x.shape[1]
        assert T_total <= self.cfg.block_size, f"context {T_total} > block_size {self.cfg.block_size}"
        pos = torch.arange(T_total, device=device)
        x = x + self.pos_emb(pos)
        x = self.drop(x)

        # Time embedding.
        sinu = timestep_embedding(t, self.cfg.d_model)
        t_emb = self.t_embed(sinu)  # (B, d_model)

        for block in self.blocks:
            x = block(x, t_emb)
        x = self.ln_f(x)
        v_all = self.velocity_head(x)
        return v_all[:, T_p:, :]  # (B, T_gen, d_model)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def decode_to_tokens(self, z_clean: torch.Tensor) -> torch.Tensor:
        """Map predicted clean embeddings back to discrete tokens via
        nearest neighbour in the embedding table (cosine).

        z_clean: (B, T, D). Returns (B, T) long.
        """
        # Cosine similarity:
        z_norm = F.normalize(z_clean.float(), dim=-1)
        e_norm = F.normalize(self.tok_emb.weight.float(), dim=-1)
        sims = z_norm @ e_norm.T  # (B, T, V)
        return sims.argmax(dim=-1)


def flow_loss(
    model: FlowLM,
    clean_ids: torch.Tensor,
    prompt_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rectified-flow training step.

    For each sequence in the batch, sample t ~ Uniform[0, 1], build
        z_t = (1-t) * z_clean + t * z_noise,
        target_velocity = z_clean - z_noise   # so dz/dt = target moves z_t toward z_clean as t decreases.
    Train MSE between model(z_t, t) and target_velocity.

    Note: in this convention, t=0 is clean and t=1 is pure noise. At
    inference we integrate ODE from t=1 (sampled noise) to t=0 (clean
    embedding) by stepping along the predicted velocity.
    """
    device = clean_ids.device
    B, T_gen = clean_ids.shape
    z_clean = model.tok_emb(clean_ids).detach()  # (B, T_gen, D); detach so emb table is only learned via velocity gradient or prompt path
    z_noise = torch.randn_like(z_clean)
    t = torch.rand(B, device=device)
    t_b = t.view(B, 1, 1)
    z_t = (1 - t_b) * z_clean + t_b * z_noise
    target = z_clean - z_noise

    pred = model(z_t, t, prompt_ids=prompt_ids)
    return F.mse_loss(pred, target)
