"""Phase P.2 by depth — cross-head SAE feature overlap, per layer.

Runs the cross_head_overlap analysis for blocks 6..15 (the divergence
zone identified in P.0) and produces a single specialization-by-depth
table.

Headline question for the paper: does the 0.169 mean cosine result
from `ln_f` hold uniformly across all 10 layers, or does the
specialisation vary with depth?

ENV:
  BLOCKS=6,7,8,9,10,11,12,13,14,15   (override layer list)
  SAES_DIR=e5/interp/saes            (where the per-layer SAEs live)
  OUT=e5/interp/results/cross_head_overlap_by_depth.json
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


def compute_overlap(sae_ar_path: Path, sae_diff_path: Path):
    ar = torch.load(sae_ar_path, map_location="cpu", weights_only=False)
    di = torch.load(sae_diff_path, map_location="cpu", weights_only=False)
    W_ar = ar["state_dict"]["W_dec"]
    W_di = di["state_dict"]["W_dec"]
    W_ar_n = F.normalize(W_ar, dim=1)
    W_di_n = F.normalize(W_di, dim=1)

    chunk = 1024
    best_ad = []
    best_da = []
    for i in range(0, W_ar_n.shape[0], chunk):
        sims = W_ar_n[i:i+chunk] @ W_di_n.T
        best_ad.append(sims.max(dim=1).values)
    for i in range(0, W_di_n.shape[0], chunk):
        sims = W_di_n[i:i+chunk] @ W_ar_n.T
        best_da.append(sims.max(dim=1).values)
    best_ad = torch.cat(best_ad)
    best_da = torch.cat(best_da)

    return {
        "ar_to_diff_mean": float(best_ad.mean()),
        "ar_to_diff_median": float(best_ad.median()),
        "ar_to_diff_max": float(best_ad.max()),
        "ar_to_diff_p90": float(best_ad.quantile(0.90)),
        "ar_to_diff_share_at_0.5": float((best_ad >= 0.5).float().mean()),
        "ar_to_diff_share_at_0.7": float((best_ad >= 0.7).float().mean()),
        "ar_to_diff_share_at_0.9": float((best_ad >= 0.9).float().mean()),
        "diff_to_ar_mean": float(best_da.mean()),
        "diff_to_ar_share_at_0.7": float((best_da >= 0.7).float().mean()),
        "d_features": int(W_ar.shape[0]),
        "d_in": int(W_ar.shape[1]),
    }


def main():
    saes_dir = Path(os.environ.get("SAES_DIR",
        str(REPO_ROOT / "e5/interp/saes")))
    blocks_env = os.environ.get("BLOCKS", "6,7,8,9,10,11,12,13,14,15")
    blocks = [int(x) for x in blocks_env.split(",") if x.strip()]
    out_path = Path(os.environ.get("OUT",
        str(REPO_ROOT / "e5/interp/results/cross_head_overlap_by_depth.json")))

    by_depth = {}
    for L in blocks:
        ar_p = saes_dir / f"block_{L}_ar" / "sae.pt"
        di_p = saes_dir / f"block_{L}_diff" / "sae.pt"
        if not ar_p.exists() or not di_p.exists():
            print(f"[skip] block.{L}: missing pair (ar={ar_p.exists()}, diff={di_p.exists()})")
            continue
        print(f"[compute] block.{L} ...", flush=True)
        stats = compute_overlap(ar_p, di_p)
        by_depth[f"block_{L}"] = stats
        print(f"  mean={stats['ar_to_diff_mean']:.3f}  median={stats['ar_to_diff_median']:.3f}  "
              f"max={stats['ar_to_diff_max']:.3f}  share>0.7={stats['ar_to_diff_share_at_0.7']*100:.1f}%",
              flush=True)

    # Also pre-head (ln_f) for direct comparability with the original P.2
    ln_ar = saes_dir / "ln_f_ar" / "sae.pt"
    ln_di = saes_dir / "ln_f_diff" / "sae.pt"
    if ln_ar.exists() and ln_di.exists():
        print("[compute] ln_f ...", flush=True)
        stats = compute_overlap(ln_ar, ln_di)
        by_depth["ln_f"] = stats
        print(f"  mean={stats['ar_to_diff_mean']:.3f}  median={stats['ar_to_diff_median']:.3f}  "
              f"max={stats['ar_to_diff_max']:.3f}  share>0.7={stats['ar_to_diff_share_at_0.7']*100:.1f}%",
              flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"by_depth": by_depth}, f, indent=2)

    print("\n=== Specialisation by depth (P.2 expanded) ===", flush=True)
    print(f"{'layer':>10s} {'mean_cos':>10s} {'median':>10s} {'max':>8s} {'share≥0.7':>11s}", flush=True)
    for name, s in by_depth.items():
        print(f"{name:>10s} {s['ar_to_diff_mean']:>10.3f} {s['ar_to_diff_median']:>10.3f} "
              f"{s['ar_to_diff_max']:>8.3f} {s['ar_to_diff_share_at_0.7']*100:>10.1f}%", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
