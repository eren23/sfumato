"""Phase P.5 visual — cross-head SAE feature overlap as a 2-D heatmap +
distribution histogram. Plus an animated GIF of layer-by-layer cosine
divergence from the Phase P.0 dump.

Outputs:
  e5/interp/results/p5_dashboard/fig_overlap_heatmap.png
  e5/interp/results/p5_dashboard/fig_overlap_histogram.png
  e5/interp/results/p5_dashboard/fig_layer_divergence.gif
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
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn.functional as F

from e5.interp.topk_sae import TopKSAE
from e5.interp.p3_per_prompt_analysis import load_sae


def main():
    out_dir = Path(REPO_ROOT / "e5/interp/results/p5_dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_ar, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt", "cpu")
    sae_diff, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt", "cpu")

    W_ar = F.normalize(sae_ar.W_dec.detach(), dim=1)
    W_di = F.normalize(sae_diff.W_dec.detach(), dim=1)

    # Heatmap: pick a 256x256 subset for legibility. We sort both axes by
    # their best-cross-cosine so the diagonal-ish pattern (if any) shows up.
    print("[viz] computing per-feature best cosines ...", flush=True)
    best_ar = torch.zeros(W_ar.shape[0])
    chunk = 1024
    for i in range(0, W_ar.shape[0], chunk):
        best_ar[i:i+chunk] = (W_ar[i:i+chunk] @ W_di.T).max(dim=1).values
    sort_ar = best_ar.argsort(descending=True)
    top_ar_idx = sort_ar[:256]
    # Same for diff
    best_di = torch.zeros(W_di.shape[0])
    for i in range(0, W_di.shape[0], chunk):
        best_di[i:i+chunk] = (W_di[i:i+chunk] @ W_ar.T).max(dim=1).values
    sort_di = best_di.argsort(descending=True)
    top_di_idx = sort_di[:256]

    sub_sim = W_ar[top_ar_idx] @ W_di[top_di_idx].T  # (256, 256)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sub_sim.numpy(), cmap="magma", vmin=-0.1, vmax=0.9, aspect="auto")
    ax.set_xlabel("diff features (sorted by best-cosine)")
    ax.set_ylabel("AR features (sorted by best-cosine)")
    ax.set_title("Cross-head SAE cosine matrix (top-256 × top-256)\n"
                 "Mean of full 16k×16k matrix = 0.169; visualized here = best-bucket")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_overlap_heatmap.png", dpi=160)
    plt.close(fig)
    print(f"[viz] wrote fig_overlap_heatmap.png", flush=True)

    # Histogram of best-cosines (the full 16k distribution)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(best_ar.numpy(), bins=80, alpha=0.7, color="#3b82f6", label="AR→diff best")
    ax.hist(best_di.numpy(), bins=80, alpha=0.5, color="#ef4444", label="diff→AR best")
    ax.axvline(best_ar.mean().item(), ls="--", color="#3b82f6",
               label=f"AR→diff mean = {best_ar.mean():.3f}")
    ax.axvline(0.5, ls=":", color="#888", label="0.5 threshold (1.84%)")
    ax.axvline(0.7, ls=":", color="#444", label="0.7 threshold (0.42%)")
    ax.set_xlabel("cosine similarity to best cross-head feature")
    ax.set_ylabel("count")
    ax.set_title("Distribution of per-feature best cross-head cosine (16384 features)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_overlap_histogram.png", dpi=160)
    plt.close(fig)
    print(f"[viz] wrote fig_overlap_histogram.png", flush=True)

    # Animated GIF: layer-by-layer cosine evolution
    p0_path = REPO_ROOT / "e5/interp/cache/p0_dump_f10.pt"
    if not p0_path.exists():
        print(f"[viz] skip GIF — {p0_path} missing", flush=True)
        return
    d = torch.load(p0_path, weights_only=False)
    blocks_ar = d["blocks_ar"]
    blocks_diff = d["blocks_diff"]
    L = len(blocks_ar)
    cos_per_layer = []
    for L_idx in range(L):
        a = blocks_ar[L_idx][0]
        b = blocks_diff[L_idx][0]
        cos = F.cosine_similarity(a, b, dim=-1).mean().item()
        cos_per_layer.append(cos)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("backbone layer index")
    ax.set_ylabel("cos(AR_residual, diff_residual)")
    ax.set_title("AR vs diff residual divergence — building up by layer")
    ax.axhline(1.0, ls="--", color="#888", alpha=0.5)
    ax.grid(alpha=0.3)
    (line,) = ax.plot([], [], "-o", color="#10b981", lw=2)
    annotation = ax.text(0.05, 0.95, "", transform=ax.transAxes,
                         fontsize=11, va="top",
                         bbox=dict(facecolor="white", edgecolor="#ccc"))

    def animate(frame):
        line.set_data(range(frame + 1), cos_per_layer[: frame + 1])
        annotation.set_text(
            f"layer {frame}  cos={cos_per_layer[frame]:.3f}"
        )
        return line, annotation

    ani = animation.FuncAnimation(fig, animate, frames=L, interval=200,
                                  blit=False, repeat=True, repeat_delay=2000)
    gif_path = out_dir / "fig_layer_divergence.gif"
    try:
        ani.save(gif_path, writer="pillow", fps=4, dpi=140)
        print(f"[viz] wrote fig_layer_divergence.gif", flush=True)
    except Exception as e:
        print(f"[viz] GIF save failed: {e}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
