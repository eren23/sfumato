"""Phase P.4 demo — visualize feature activations on real prompts.

For each of N top features (from P.3 analysis), produce a matplotlib
heatmap showing per-token activation strength on a battery of prompts.
Gives a visual answer to "what does this feature actually detect?"

Outputs:
  e5/interp/results/p4_demo/feature_heatmap_{mode}_{feat_id}.png
  e5/interp/results/p4_demo/README.md  (gallery index)
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

from e5.interp.load_model import load_composite_for_interp
from e5.interp.topk_sae import TopKSAE
from e5.interp.p3_per_prompt_analysis import load_sae, collect_ln_f


def heatmap_feature(prompts: list[tuple[str, str]],
                    sae: TopKSAE, raw_model, tok, mode: str,
                    feat_id: int, out_path: Path, title: str,
                    device: str):
    """Stack the per-token activation strength of one feature across
    multiple prompts as a heatmap row per prompt."""
    rows = []
    labels = []
    token_strings = []
    for lbl, text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        act = collect_ln_f(raw_model, idx, mode)         # (T, d)
        z = sae.encode(act).cpu().detach()                # (T, d_features)
        feat_col = z[:, feat_id].numpy()
        rows.append(feat_col)
        labels.append(lbl)
        token_strings.append([tok.decode([i]).replace(" ", "▁") for i in ids])

    max_T = max(len(r) for r in rows)
    padded = np.zeros((len(rows), max_T), dtype=np.float32)
    for i, r in enumerate(rows):
        padded[i, : len(r)] = r

    fig_h = 0.55 * len(rows) + 1.0
    fig_w = 0.20 * max_T + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(padded, aspect="auto", cmap="magma",
                   interpolation="nearest", vmin=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("token position →")
    ax.set_title(f"{title} (mode={mode}, feat#{feat_id})", fontsize=11)
    # X-tick: show token strings for one representative prompt (the longest)
    longest_i = max(range(len(rows)), key=lambda i: len(rows[i]))
    longest_tokens = token_strings[longest_i]
    ax.set_xticks(range(len(longest_tokens)))
    ax.set_xticklabels(longest_tokens, rotation=70, fontsize=6.5, ha="right")
    fig.colorbar(im, ax=ax, label="feature activation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(REPO_ROOT / "e5/interp/results/p4_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p4-demo] device={device}", flush=True)

    nn_model, raw_model, _ = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt",
        device=device)
    sae_ar, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt", device)
    sae_diff, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt", device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    prompts = [
        ("janet", "Question: Janet's ducks lay 16 eggs per day. She sells them at $2 each. How much does she make?\nAnswer:"),
        ("baker", "Question: A baker makes 240 cookies, packs in boxes of 12. How many boxes?\nAnswer:"),
        ("apples", "Question: 96 apples among 8 children equally. Each gets how many?\nAnswer:"),
        ("history", "The Roman Empire reached its peak under Trajan in 117 AD, from Britain to Mesopotamia."),
        ("math_chain", "Let x = 3 + 4 * 2. Then x = "),
        ("counting", "John has 15 marbles. He gives 4 to Mary and 3 to Tom. 15 - 4 - 3 = "),
    ]

    # Pick a few interesting features per mode from P.3 G1 ducks analysis
    # AR side: #9304 (peaked at " 16" = number), #6191 (" sells"), #8435 (" day" bridge)
    # diff side: #14810 (" eggs" bridge), #14787 (" much"), #6082 (" make")
    ar_features = [
        (9304, "candidate: number/quantity"),
        (6191, "candidate: action verb"),
        (8435, "BRIDGE: ' day' / time"),
        (10109, "candidate: ' eats' / verb"),
    ]
    diff_features = [
        (14810, "BRIDGE: ' eggs' / object"),
        (14787, "candidate: ' much' / question"),
        (6082, "candidate: ' make' / question end"),
        (15540, "candidate: ' day' / time"),
    ]

    gallery_lines = [
        "# P.4 feature firing demo (visual)\n",
        "Per-feature heatmaps across 6 prompts. Brighter = stronger activation.",
        "Bridge features were identified in P.2 as the high-cosine matched",
        "pairs between AR and diff (the ~20 universal features the modes",
        "share). Candidates are heads-only top firing features from P.3.",
        "",
        "## AR-head features\n",
    ]
    for feat_id, desc in ar_features:
        out_path = out_dir / f"ar_feat_{feat_id}.png"
        heatmap_feature(prompts, sae_ar, raw_model, tok, "ar", feat_id,
                        out_path, desc, device)
        gallery_lines.append(f"![ar #{feat_id} — {desc}]({out_path.name})\n")
        print(f"[p4-demo] wrote {out_path.name}", flush=True)

    gallery_lines.append("\n## diff-head features\n")
    for feat_id, desc in diff_features:
        out_path = out_dir / f"diff_feat_{feat_id}.png"
        heatmap_feature(prompts, sae_diff, raw_model, tok, "diff", feat_id,
                        out_path, desc, device)
        gallery_lines.append(f"![diff #{feat_id} — {desc}]({out_path.name})\n")
        print(f"[p4-demo] wrote {out_path.name}", flush=True)

    (out_dir / "README.md").write_text("\n".join(gallery_lines))
    print(f"\n[p4-demo] gallery: {out_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
