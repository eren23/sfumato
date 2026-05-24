"""Phase P.1 visualisations.

Given a trained TopK SAE, run F10 on a batch of GSM8K + FineWeb prompts,
capture activations at the SAE's hookpoint, encode through the SAE, and
for the top-N most-firing features show their top-activating-token
contexts.

Outputs:
  e5/interp/results/p1_viz/{hookpoint}/training_curve.png
  e5/interp/results/p1_viz/{hookpoint}/feature_firing_summary.md
"""
from __future__ import annotations

import json
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

from e5.interp.load_model import load_composite_for_interp
from e5.interp.topk_sae import TopKSAE
from e5.interp.train_saes import attach_hook, resolve_hookpoint


def main():
    sae_path = Path(os.environ.get("SAE",
        str(REPO_ROOT / "e5/interp/saes/block_6_ar/sae.pt")))
    ckpt_path = os.environ.get("CKPT",
        str(REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt"))
    out_dir = Path(os.environ.get("OUT_DIR", str(sae_path.parent.parent.parent / "results" / "p1_viz" / sae_path.parent.name)))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_top_features = int(os.environ.get("N_TOP_FEATURES", "5"))
    n_contexts = int(os.environ.get("N_CONTEXTS", "8"))
    context_radius = int(os.environ.get("CONTEXT_RADIUS", "4"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[viz] device={device}", flush=True)

    # Load SAE
    sae_ck = torch.load(sae_path, map_location=device, weights_only=False)
    cfg = sae_ck["config"]
    sae = TopKSAE(d_in=cfg["d_in"], d_features=cfg["d_features"], k=cfg["k"])
    sae.load_state_dict(sae_ck["state_dict"])
    sae.to(device).train(False)
    hookpoint = cfg["hookpoint"]
    print(f"[viz] SAE: hookpoint={hookpoint}  d_feat={cfg['d_features']}  k={cfg['k']}",
          flush=True)

    # Load F10
    nn_model, raw_model, ccfg = load_composite_for_interp(ckpt_path, device=device)
    print(f"[viz] F10 loaded: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M",
          flush=True)

    # Plot 1: training curve from train_log.jsonl
    log_path = sae_path.parent / "train_log.jsonl"
    if log_path.exists():
        rows = [json.loads(l) for l in open(log_path)]
        if rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            steps = [r["step"] for r in rows]
            ax.plot(steps, [r["delta_loss_ratio"] for r in rows],
                    "-o", label="heldout delta-loss ratio", color="#3b82f6")
            ax.set_xlabel("step")
            ax.set_ylabel("delta_loss_ratio (1 = no recon)")
            ax.set_title(f"SAE training curve — {hookpoint}")
            ax.axhline(0.1, ls="--", color="#888", label="target")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "training_curve.png", dpi=160)
            plt.close(fig)
            print(f"[viz] wrote training_curve.png", flush=True)

    # Plot 2: feature firing on a small prompt corpus
    prompts = [
        # GSM8K-style
        "Question: Janet has 16 ducks. She feeds them three times a day.\nAnswer:",
        "Question: A baker makes 30 cookies. He sells 12 of them. How many remain?\nAnswer:",
        "Question: There are 5 boxes each with 8 apples. How many apples total?\nAnswer:",
        # Prose
        "The wind blew through the trees as the storm approached the village.",
        "Once upon a time, in a kingdom far away, there lived a young princess",
        "In machine learning, gradient descent updates parameters by computing",
        # Math-coded
        "Let x = 3 + 4 * 2. Then x =",
        "The total cost is 50 - 12 + 7 = ",
    ]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    hook_target, mode = resolve_hookpoint(hookpoint)

    # Capture activations on each prompt
    all_acts = []  # list of (T, d_in) per prompt
    all_token_ids = []
    store = {"act": None}
    handle = attach_hook(raw_model, hook_target, store)
    try:
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=False)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                raw_model(idx, mode=mode)
            acts = store["act"][0].detach().float()  # (T, d_in)
            all_acts.append(acts)
            all_token_ids.append(ids)
    finally:
        handle.remove()

    # Encode through SAE; for each (prompt, token, feature) get activation
    # Find features that fire most often across the corpus
    feature_fire_counts = torch.zeros(cfg["d_features"], device=device)
    feature_total_act = torch.zeros(cfg["d_features"], device=device)
    per_token_z = []   # list of (T, d_features) per prompt
    with torch.no_grad():
        for acts in all_acts:
            z = sae.encode(acts)  # (T, d_features) sparse
            per_token_z.append(z.cpu())
            mask = (z > 0).float()
            feature_fire_counts += mask.sum(dim=0)
            feature_total_act += z.sum(dim=0)

    # Pick top-N features by total activation
    top_features = feature_total_act.topk(n_top_features).indices.cpu().tolist()
    print(f"[viz] top features: {top_features}", flush=True)

    # For each top feature, find top firing token contexts
    summary_lines = [f"# SAE feature firing — {hookpoint}\n",
                     f"SAE: `{sae_path}`\n",
                     f"d_features={cfg['d_features']}  k={cfg['k']}  n_prompts={len(prompts)}\n",
                     ""]

    for feat_idx in top_features:
        # Find top firing positions across corpus
        candidates = []  # list of (act_val, prompt_idx, token_pos)
        for p_i, z in enumerate(per_token_z):
            col = z[:, feat_idx]   # (T,)
            for t_i, val in enumerate(col.tolist()):
                if val > 0:
                    candidates.append((val, p_i, t_i))
        candidates.sort(key=lambda x: -x[0])
        contexts = candidates[:n_contexts]

        summary_lines.append(f"\n## Feature #{feat_idx}  "
                             f"(total_act={float(feature_total_act[feat_idx]):.2f}, "
                             f"fired in {int(feature_fire_counts[feat_idx])} positions)\n")
        for val, p_i, t_i in contexts:
            ids = all_token_ids[p_i]
            lo = max(0, t_i - context_radius)
            hi = min(len(ids), t_i + context_radius + 1)
            ctx_tokens = ids[lo:hi]
            mid_idx_in_ctx = t_i - lo
            decoded = []
            for ji, tid in enumerate(ctx_tokens):
                tok_str = tok.decode([tid]).replace("\n", "\\n")
                if ji == mid_idx_in_ctx:
                    decoded.append(f"**[{tok_str}]**")
                else:
                    decoded.append(tok_str)
            summary_lines.append(f"- act={val:.2f}  ctx: `{' '.join(decoded)}`")

    summary_path = out_dir / "feature_firing_summary.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[viz] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
