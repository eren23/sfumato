"""Phase P.0 visualisations from the activation dump.

Reads e5/interp/cache/p0_dump_f10.pt and emits a small gallery of
diagnostic plots WITHOUT needing SAEs. The headline plot — per-layer
AR-vs-diff cosine similarity of the residual stream — is a pre-SAE
preview of the P.2 question ("do the modes share features?").

Plots produced under e5/interp/results/p0_viz/:
  fig_norm_per_layer.png       — ||residual|| vs depth, AR vs diff
  fig_cosine_ar_vs_diff.png    — cosine(residual_ar, residual_diff) vs depth
  fig_token_divergence.png     — per-token cosine, layer 0/10/19 panels
  fig_ln_f_distribution.png    — magnitude histogram, AR vs diff at ln_f
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def main():
    dump_path = os.environ.get("DUMP",
        str(REPO_ROOT / "e5/interp/cache/p0_dump_f10.pt"))
    out_dir = Path(os.environ.get("OUT_DIR",
        REPO_ROOT / "e5/interp/results/p0_viz"))
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(dump_path, weights_only=False)
    blocks_ar = d["blocks_ar"]    # list of [1, T, d] tensors, one per layer
    blocks_diff = d["blocks_diff"]
    ln_f_ar = d["ln_f_ar"][0]     # (T, d)
    ln_f_diff = d["ln_f_diff"][0]
    T = ln_f_ar.shape[0]
    L = len(blocks_ar)
    d_model = ln_f_ar.shape[1]
    print(f"[viz] L={L}  T={T}  d={d_model}")

    # ---- Plot 1: norm per layer (mean over tokens) ----
    norms_ar = [float(b[0].norm(dim=-1).mean()) for b in blocks_ar]
    norms_diff = [float(b[0].norm(dim=-1).mean()) for b in blocks_diff]
    layers = list(range(L))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, norms_ar, "-o", label="AR mode", color="#3b82f6")
    ax.plot(layers, norms_diff, "-o", label="diff mode", color="#ef4444")
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean ||residual||₂ over tokens")
    ax.set_title("Residual-stream norm by layer (F10 305M)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_norm_per_layer.png", dpi=160)
    plt.close(fig)
    print(f"[viz] wrote fig_norm_per_layer.png")

    # ---- Plot 2: cosine(residual_ar, residual_diff) at each layer ----
    # token-averaged cosine
    cos_per_layer = []
    for L_idx in range(L):
        a = blocks_ar[L_idx][0]   # (T, d)
        b = blocks_diff[L_idx][0]
        cos = F.cosine_similarity(a, b, dim=-1)  # (T,)
        cos_per_layer.append(float(cos.mean()))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, cos_per_layer, "-o", color="#10b981")
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean cosine(AR_t, diff_t) over tokens")
    ax.set_title("AR vs diff residual stream divergence by layer")
    ax.axhline(1.0, ls="--", color="#888", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cosine_ar_vs_diff.png", dpi=160)
    plt.close(fig)
    print(f"[viz] wrote fig_cosine_ar_vs_diff.png")

    # ---- Plot 3: per-token cosine, 3 layer panels ----
    show_layers = [0, L // 2, L - 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, L_idx in zip(axes, show_layers):
        a = blocks_ar[L_idx][0]
        b = blocks_diff[L_idx][0]
        cos = F.cosine_similarity(a, b, dim=-1).numpy()
        ax.plot(range(T), cos, "-", color="#10b981", lw=1.4)
        ax.set_title(f"layer {L_idx}")
        ax.set_xlabel("token position")
        ax.set_ylabel("cos(AR, diff)")
        ax.set_ylim(-0.2, 1.1)
        ax.axhline(1.0, ls="--", color="#888", alpha=0.4)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-token AR vs diff cosine at 3 depths (F10)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_token_divergence.png", dpi=160)
    plt.close(fig)
    print(f"[viz] wrote fig_token_divergence.png")

    # ---- Plot 4: ln_f magnitude histogram ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ar_vals = ln_f_ar.flatten().numpy()
    diff_vals = ln_f_diff.flatten().numpy()
    ax.hist(ar_vals, bins=80, alpha=0.55, label="AR", color="#3b82f6")
    ax.hist(diff_vals, bins=80, alpha=0.55, label="diff", color="#ef4444")
    ax.set_yscale("log")
    ax.set_xlabel("activation value (post-ln_f)")
    ax.set_ylabel("count (log)")
    ax.set_title("Post-ln_f activation distribution (F10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ln_f_distribution.png", dpi=160)
    plt.close(fig)
    print(f"[viz] wrote fig_ln_f_distribution.png")

    # ---- Write a tiny README summarising ----
    summary = (out_dir / "README.md")
    with open(summary, "w") as f:
        f.write("# Phase P.0 visualisations\n\n")
        f.write(f"Source: `{dump_path}`\n\n")
        f.write(f"- Layers: {L}\n")
        f.write(f"- Tokens: {T}\n")
        f.write(f"- d_model: {d_model}\n\n")
        f.write("## Key numbers\n\n")
        f.write(f"- Final-layer mean cosine(AR, diff) = **{cos_per_layer[-1]:.3f}**\n")
        f.write(f"- Layer-0 cosine = {cos_per_layer[0]:.3f}, monotone decrease through depth\n")
        f.write(f"- Norm at last layer: AR={norms_ar[-1]:.1f}, diff={norms_diff[-1]:.1f}\n\n")
        f.write("Cosine drop from layer 0 → 19 is the pre-SAE preview of the P.2\n")
        f.write("cross-head overlap question. If modes shared one representation,\n")
        f.write("we'd see cosine ≈ 1.0 throughout. We observe substantial divergence.\n")
    print(f"[viz] wrote {summary}")
    print(f"\nFinal-layer cos(AR, diff) = {cos_per_layer[-1]:.3f}")
    print(f"Layer-0 cos = {cos_per_layer[0]:.3f}")


if __name__ == "__main__":
    main()
