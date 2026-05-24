"""Phase P.1 — minimal TopK SAE module + trainer.

We roll our own instead of pulling in eai-sparsify because:
- Sparsify's hookpoint resolver assumes HF model structure; Sfumato
  is custom PyTorch.
- For a one-shot ~12-SAE training run, a 200-line trainer is simpler
  than fighting a framework.
- TopK SAE is conceptually tiny: encoder → TopK → decoder.

API:
    sae = TopKSAE(d_in=1024, d_features=16384, k=64).cuda()
    optim = torch.optim.AdamW(sae.parameters(), lr=3e-4)
    # During training, collect activations from a CompositeLM forward
    # pass at a specific hookpoint, then:
    z = sae.encode(acts)            # (B*T, d_features), sparse
    recon = sae.decode(z)           # (B*T, d_in)
    loss = (acts - recon).pow(2).mean()
    loss.backward(); optim.step()
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Top-K sparse autoencoder for residual-stream activations.

    Encoder: Linear(d_in, d_features), then keep only the top-k
    activations per token (set rest to 0).
    Decoder: Linear(d_features, d_in), no nonlinearity.

    Decoder columns are unit-normalised periodically (every
    `renorm_every` steps) to keep features comparable across SAEs.
    """

    def __init__(self, d_in: int, d_features: int, k: int = 64):
        super().__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.k = k
        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_in, d_features))
        self.b_enc = nn.Parameter(torch.zeros(d_features))
        # Decoder
        self.W_dec = nn.Parameter(torch.empty(d_features, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Init: orthogonal-ish; decoder columns unit-norm
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) → z: (..., d_features), TopK-sparse."""
        x_centered = x - self.b_dec
        pre = x_centered @ self.W_enc + self.b_enc
        # TopK per token: keep top-k positive activations
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals.relu())
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renormalize_decoder(self):
        """Force each decoder column to unit norm. Improves feature
        interpretability and cosine-comparison consistency."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.W_dec /= norms
        # Compensate encoder so reconstruction is unchanged (approx)
        self.W_enc *= norms.squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def topk_sae_loss(x: torch.Tensor, recon: torch.Tensor, z: torch.Tensor,
                  reg_lambda: float = 0.0):
    """Reconstruction MSE + optional L1 on the sparse codes."""
    recon_loss = (x - recon).pow(2).mean()
    if reg_lambda > 0:
        sparsity_loss = z.abs().mean()
        total = recon_loss + reg_lambda * sparsity_loss
    else:
        sparsity_loss = z.abs().mean().detach()
        total = recon_loss
    return total, recon_loss.detach(), sparsity_loss
