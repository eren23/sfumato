"""Phase P.2 — cross-head SAE feature overlap analysis.

Loads two SAEs (one trained on ln_f.ar activations, one on ln_f.diff)
and computes:
  1. Cosine similarity matrix between every AR decoder column and
     every diff decoder column. (d_features × d_features)
  2. Fraction of AR features with at least one diff sibling above
     cosine threshold (0.5, 0.7, 0.9). The headline number for the
     paper.
  3. Top-k matched pairs (highest cosine) — these are the "shared"
     features that both modes use.

ENV:
  SAE_AR=path/to/ar_sae.pt   (default: e5/interp/saes/ln_f_ar/sae.pt)
  SAE_DIFF=path/to/diff_sae.pt
  OUT=path/to/results.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F


def main():
    sae_ar_path = Path(os.environ.get("SAE_AR",
        str(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt")))
    sae_diff_path = Path(os.environ.get("SAE_DIFF",
        str(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt")))
    out_path = Path(os.environ.get("OUT",
        str(REPO_ROOT / "e5/interp/results/cross_head_overlap.json")))

    print(f"[overlap] loading SAEs:\n  AR  : {sae_ar_path}\n  diff: {sae_diff_path}", flush=True)
    ar = torch.load(sae_ar_path, map_location="cpu", weights_only=False)
    di = torch.load(sae_diff_path, map_location="cpu", weights_only=False)
    W_ar = ar["state_dict"]["W_dec"]   # (d_features, d_in)
    W_di = di["state_dict"]["W_dec"]
    print(f"[overlap] AR decoder: {W_ar.shape}, diff decoder: {W_di.shape}", flush=True)

    # Normalize columns to unit length so dot = cosine
    W_ar_n = F.normalize(W_ar, dim=1)
    W_di_n = F.normalize(W_di, dim=1)

    # Per-AR-feature: highest cosine to any diff feature
    # Compute in chunks to control memory
    chunk = 1024
    best_cosines_ar_to_diff = []
    best_diff_idx_for_ar = []
    for i in range(0, W_ar_n.shape[0], chunk):
        sub = W_ar_n[i:i+chunk]                      # (B, d_in)
        sims = sub @ W_di_n.T                        # (B, F_diff)
        vals, idx = sims.max(dim=1)
        best_cosines_ar_to_diff.append(vals)
        best_diff_idx_for_ar.append(idx)
    best_cosines_ar_to_diff = torch.cat(best_cosines_ar_to_diff)
    best_diff_idx_for_ar = torch.cat(best_diff_idx_for_ar)

    # Symmetric: per diff-feature best AR
    best_cosines_diff_to_ar = []
    best_ar_idx_for_diff = []
    for i in range(0, W_di_n.shape[0], chunk):
        sub = W_di_n[i:i+chunk]
        sims = sub @ W_ar_n.T
        vals, idx = sims.max(dim=1)
        best_cosines_diff_to_ar.append(vals)
        best_ar_idx_for_diff.append(idx)
    best_cosines_diff_to_ar = torch.cat(best_cosines_diff_to_ar)
    best_ar_idx_for_diff = torch.cat(best_ar_idx_for_diff)

    # Headline numbers
    def share_frac(cosines, threshold):
        return float((cosines >= threshold).float().mean())

    headline = {
        "ar_to_diff_share_at_0.5": share_frac(best_cosines_ar_to_diff, 0.5),
        "ar_to_diff_share_at_0.7": share_frac(best_cosines_ar_to_diff, 0.7),
        "ar_to_diff_share_at_0.9": share_frac(best_cosines_ar_to_diff, 0.9),
        "diff_to_ar_share_at_0.5": share_frac(best_cosines_diff_to_ar, 0.5),
        "diff_to_ar_share_at_0.7": share_frac(best_cosines_diff_to_ar, 0.7),
        "diff_to_ar_share_at_0.9": share_frac(best_cosines_diff_to_ar, 0.9),
    }

    # Cosine distribution stats
    distribution_stats = {
        "ar_to_diff_mean": float(best_cosines_ar_to_diff.mean()),
        "ar_to_diff_median": float(best_cosines_ar_to_diff.median()),
        "ar_to_diff_max": float(best_cosines_ar_to_diff.max()),
        "ar_to_diff_min": float(best_cosines_ar_to_diff.min()),
        "diff_to_ar_mean": float(best_cosines_diff_to_ar.mean()),
    }

    # Top 20 highest-cosine matched pairs
    sort_idx = best_cosines_ar_to_diff.argsort(descending=True)
    top20 = []
    for i in sort_idx[:20].tolist():
        top20.append({
            "ar_feature": int(i),
            "diff_feature": int(best_diff_idx_for_ar[i]),
            "cosine": float(best_cosines_ar_to_diff[i]),
        })

    result = {
        "sae_ar_path": str(sae_ar_path),
        "sae_diff_path": str(sae_diff_path),
        "d_features": int(W_ar.shape[0]),
        "d_in": int(W_ar.shape[1]),
        "headline_overlap_share": headline,
        "distribution_stats": distribution_stats,
        "top20_matched_pairs": top20,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Cross-head SAE feature overlap ===", flush=True)
    print(f"  d_features = {W_ar.shape[0]}", flush=True)
    print(f"  AR→diff best-cosine: mean={distribution_stats['ar_to_diff_mean']:.3f} "
          f"median={distribution_stats['ar_to_diff_median']:.3f} "
          f"max={distribution_stats['ar_to_diff_max']:.3f}",
          flush=True)
    print(f"  Fraction of AR features with sibling cosine ≥ X:", flush=True)
    print(f"    0.5: {headline['ar_to_diff_share_at_0.5']*100:.1f}%", flush=True)
    print(f"    0.7: {headline['ar_to_diff_share_at_0.7']*100:.1f}%", flush=True)
    print(f"    0.9: {headline['ar_to_diff_share_at_0.9']*100:.1f}%", flush=True)
    print(f"  Symmetric (diff→ar): 0.7 sibling share = "
          f"{headline['diff_to_ar_share_at_0.7']*100:.1f}%", flush=True)
    print(f"\n[overlap] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
