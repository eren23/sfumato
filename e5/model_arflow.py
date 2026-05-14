"""T1 — composite AR + continuous-flow (ELF-flavoured) language model.

The discrete-mask T0 composite (model_composite.py) trains two heads
(causal AR + bidirectional mask-prediction) on a single backbone. T1
swaps the discrete-mask diffusion half for a continuous-flow head over
the same token-embedding space:

  - head_ar : standard LM head, causal-masked attention, next-token CE
  - head_flow : velocity-prediction head, bidirectional attention,
                rectified-flow / OT-CFM loss in continuous embedding space

The backbone is shared between modes. Mode is selected per forward call:
  - mode="ar": causal attention, input = token ids, output = next-token logits
  - mode="flow": bidirectional attention, input = noisy embeddings z_t + t,
                 output = velocity field v_θ(z_t, t) of shape (B, T, d_model)

Time conditioning for the flow mode is DiT-style AdaLN (scale, shift, gate
per sublayer per layer, predicted by an MLP from sinusoidal t embedding).
For the AR mode the AdaLN modulation is set to its identity (scale=0,
shift=0, gate=1) so the same backbone applies cleanly to both modes.

The composite-paired inference mode is "AR generates first K tokens, then
flow fills the remaining (T-K) via ODE integration from noise". This
matches T0's `paired` mode but with continuous flow replacing the discrete
mask sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GPT2_VOCAB = 50257


@dataclass
class ARFlowConfig:
    vocab_size: int = GPT2_VOCAB
    block_size: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal time embedding. t shape: (B,). Returns (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(1, half)
    )
    args = t.float().reshape(-1, 1) * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class SelfAttention(nn.Module):
    """Self-attention with runtime-selectable mask: causal for ar, full for flow."""

    def __init__(self, cfg: ARFlowConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, is_causal: bool) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLPBlock(nn.Module):
    def __init__(self, cfg: ARFlowConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class DualModeBlock(nn.Module):
    """Transformer block with mode-aware AdaLN modulation.

    In flow mode: time embedding produces (scale, shift, gate) for both
    LN sublayers via a learned linear projection (zero-init).
    In AR mode: modulation is the identity (scale=0, shift=0, gate=1),
    which makes the block behave like a standard pre-norm transformer block.
    """

    def __init__(self, cfg: ARFlowConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLPBlock(cfg)
        # 6 modulation params: (scale, shift, gate) × 2 sublayers
        self.mod = nn.Linear(cfg.d_model, 6 * cfg.d_model, bias=True)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        # Default LN-affine for AR mode (also elementwise_affine=False above, so
        # we apply explicit weight=1 bias=0 to mimic the "no modulation" path).
        self.register_buffer("_ones", torch.ones(cfg.d_model))
        self.register_buffer("_zeros", torch.zeros(cfg.d_model))

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool,
        t_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        if t_emb is not None:
            scale1, shift1, gate1, scale2, shift2, gate2 = self.mod(t_emb).chunk(6, dim=-1)
            scale1, shift1, gate1 = scale1.unsqueeze(1), shift1.unsqueeze(1), gate1.unsqueeze(1)
            scale2, shift2, gate2 = scale2.unsqueeze(1), shift2.unsqueeze(1), gate2.unsqueeze(1)
            h = self.ln1(x) * (1 + scale1) + shift1
            x = x + gate1 * self.attn(h, is_causal=is_causal)
            h = self.ln2(x) * (1 + scale2) + shift2
            x = x + gate2 * self.mlp(h)
        else:
            # AR mode: identity modulation; gate=1 means residuals are unblocked.
            x = x + self.attn(self.ln1(x), is_causal=is_causal)
            x = x + self.mlp(self.ln2(x))
        return x


class ARFlowLM(nn.Module):
    """Composite AR + continuous-flow LM.

    forward(idx_or_z, mode, t=None, prompt_ids=None) where:
      mode="ar":   idx_or_z is token ids (B, T) long.
                   Returns next-token logits (B, T, V).
      mode="flow": idx_or_z is noisy embeddings (B, T_gen, d_model).
                   Requires t shape (B,) ∈ [0,1] and optionally
                   prompt_ids shape (B, T_prompt) for conditional flow.
                   Returns velocity field (B, T_gen, d_model).
    """

    def __init__(self, cfg: ARFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # Project noisy continuous input into hidden space (identity-init).
        self.flow_input_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.eye_(self.flow_input_proj.weight)
        # Time embedding MLP (only used in flow mode).
        self.t_embed = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.SiLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([DualModeBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        # AR head ties to tok_emb (no extra params).
        # Flow head: zero-init linear, then tied to embedding inner product.
        self.flow_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.flow_proj.weight)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # Skip already-zeroed (mod, flow_proj) and already-identity (flow_input_proj)
            if module.weight is None:
                return
            w_abs_max = module.weight.data.abs().max().item()
            if w_abs_max < 1e-9:
                return  # leave zeroed
            if abs(w_abs_max - 1.0) < 1e-6 and module.weight.shape[0] == module.weight.shape[1]:
                # Likely the identity-init flow_input_proj. Leave it.
                if torch.allclose(module.weight.data, torch.eye(module.weight.shape[0], device=module.weight.device)):
                    return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _backbone(
        self,
        x: torch.Tensor,
        is_causal: bool,
        t_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, is_causal=is_causal, t_emb=t_emb)
        return self.ln_f(x)

    def forward(
        self,
        idx_or_z: torch.Tensor,
        mode: str,
        t: torch.Tensor | None = None,
        prompt_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = idx_or_z.shape[0]
        device = idx_or_z.device

        if mode == "ar":
            assert idx_or_z.dtype == torch.long
            T = idx_or_z.shape[1]
            assert T <= self.cfg.block_size
            pos = torch.arange(T, device=device)
            x = self.tok_emb(idx_or_z) + self.pos_emb(pos)
            x = self.drop(x)
            x = self._backbone(x, is_causal=True, t_emb=None)
            return x @ self.tok_emb.weight.T  # (B, T, V)

        if mode == "flow":
            assert idx_or_z.dim() == 3, "flow mode expects noisy embeddings (B, T_gen, D)"
            assert t is not None
            T_gen = idx_or_z.shape[1]
            if prompt_ids is not None:
                T_p = prompt_ids.shape[1]
                prompt_emb = self.tok_emb(prompt_ids)
            else:
                T_p = 0
                prompt_emb = idx_or_z.new_zeros((B, 0, self.cfg.d_model))
            x_gen = self.flow_input_proj(idx_or_z)
            x = torch.cat([prompt_emb, x_gen], dim=1)
            T_total = x.shape[1]
            assert T_total <= self.cfg.block_size
            pos = torch.arange(T_total, device=device)
            x = x + self.pos_emb(pos)
            x = self.drop(x)
            sinu = timestep_embedding(t, self.cfg.d_model)
            t_emb = self.t_embed(sinu)
            x = self._backbone(x, is_causal=False, t_emb=t_emb)
            v_all = self.flow_proj(x)
            return v_all[:, T_p:, :]  # velocity for the generated positions only

        raise ValueError(f"unknown mode {mode!r}")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def decode_to_tokens(self, z_clean: torch.Tensor) -> torch.Tensor:
        """Nearest-token lookup via cosine similarity to the embedding table."""
        z_norm = F.normalize(z_clean.float(), dim=-1)
        e_norm = F.normalize(self.tok_emb.weight.float(), dim=-1)
        sims = z_norm @ e_norm.T
        return sims.argmax(dim=-1)


def ar_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def flow_loss(
    model: ARFlowLM,
    clean_ids: torch.Tensor,
    prompt_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rectified-flow training step.

    z_t = (1 - t)·z_clean + t·z_noise
    target = z_clean − z_noise
    pred = v_θ(z_t, t)
    loss = MSE(pred, target)
    """
    device = clean_ids.device
    B, T_gen = clean_ids.shape
    z_clean = model.tok_emb(clean_ids).detach()
    z_noise = torch.randn_like(z_clean)
    t = torch.rand(B, device=device)
    t_b = t.view(B, 1, 1)
    z_t = (1 - t_b) * z_clean + t_b * z_noise
    target = z_clean - z_noise
    pred = model(z_t, mode="flow", t=t, prompt_ids=prompt_ids)
    return F.mse_loss(pred, target)
