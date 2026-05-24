"""Phase P.5 visual — Neuronpedia-style HTML feature dashboard.

For each interesting feature, generates an HTML page showing:
  - Feature ID, hookpoint, total activation in corpus
  - Top-20 firing contexts with token-level color heatmap
  - Activation bar chart for the top-firing tokens
  - Cross-head pairing info (bridge feature?)

Output:
  e5/interp/results/p5_dashboard/index.html  (gallery)
  e5/interp/results/p5_dashboard/feat_{mode}_{id}.html
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from e5.interp.load_model import load_composite_for_interp
from e5.interp.p3_per_prompt_analysis import load_sae, collect_ln_f


def colour_for(val, vmax):
    """Map activation 0..vmax to an inline CSS rgba background."""
    if vmax <= 0 or val <= 0:
        return "rgba(255,255,255,0)"
    t = min(val / vmax, 1.0)
    # white → yellow → red ramp
    if t < 0.5:
        a = t * 2
        r, g, b = 255, int(255 * (1 - 0.1 * a)), int(255 * (1 - 0.85 * a))
    else:
        a = (t - 0.5) * 2
        r = 255
        g = int(255 * (0.9 - 0.7 * a))
        b = int(255 * (0.15 - 0.15 * a))
    return f"rgba({r},{g},{b},{0.85*t:.2f})"


def render_context(token_strs, acts, peak_idx, vmax, radius=8):
    lo = max(0, peak_idx - radius)
    hi = min(len(token_strs), peak_idx + radius + 1)
    pieces = []
    for j in range(lo, hi):
        tok = html.escape(token_strs[j]).replace("\n", "↵")
        bg = colour_for(acts[j], vmax)
        bold = " font-weight:600" if j == peak_idx else ""
        pieces.append(
            f'<span style="background:{bg};padding:1px 3px;border-radius:2px;{bold}" '
            f'title="act={acts[j]:.2f}">{tok}</span>'
        )
    return "<span style='font-family:monospace;font-size:13px;'>" + "".join(pieces) + "</span>"


def build_feature_page(feat_id, mode, sae, raw_model, tok, prompts, bridge_pairs,
                       out_path: Path):
    rows = []
    for label, text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=sae.W_dec.device)
        act = collect_ln_f(raw_model, idx, mode)
        z = sae.encode(act).cpu().detach().numpy()
        feat_col = z[:, feat_id]
        peak_idx = int(np.argmax(feat_col))
        peak_act = float(feat_col[peak_idx])
        token_strs = [tok.decode([i]) for i in ids]
        rows.append({
            "label": label,
            "peak_act": peak_act,
            "peak_idx": peak_idx,
            "token_strs": token_strs,
            "acts": feat_col.tolist(),
        })

    rows.sort(key=lambda r: -r["peak_act"])
    rows = [r for r in rows if r["peak_act"] > 0][:20]
    vmax = max(r["peak_act"] for r in rows) if rows else 1.0

    bridge_info = ""
    for p in bridge_pairs:
        if (mode == "ar" and p["ar_feature"] == feat_id) or \
           (mode == "diff" and p["diff_feature"] == feat_id):
            other_mode = "diff" if mode == "ar" else "ar"
            other_id = p["diff_feature"] if mode == "ar" else p["ar_feature"]
            bridge_info = (f' <span style="background:#7c3aed22;color:#7c3aed;'
                           f'padding:2px 6px;border-radius:4px;font-size:12px;">'
                           f'BRIDGE → {other_mode} #{other_id} (cos={p["cosine"]:.3f})</span>')
            break

    parts = [
        f'<!doctype html><html><head><meta charset="utf-8"><title>{mode}/#{feat_id}</title>',
        '<style>body{font-family:-apple-system,Helvetica,sans-serif;max-width:920px;margin:30px auto;padding:0 18px;color:#111;}'
        'h1{font-size:22px;margin-bottom:0;}h2{font-size:15px;color:#666;margin-top:4px;}'
        '.row{margin:14px 0;padding:8px 10px;border:1px solid #eee;border-radius:6px;background:#fafafa;}'
        '.row .label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;}'
        '.row .peak{font-size:11px;color:#555;margin-left:10px;}'
        '.bar{background:#3b82f6;height:5px;border-radius:2px;display:block;margin:4px 0 6px;}'
        '</style></head><body>',
        f'<h1>{mode}-head SAE feature #{feat_id}{bridge_info}</h1>',
        f'<h2>Top contexts ranked by peak activation. Higher = more red.</h2>',
    ]
    for r in rows:
        bar_w = int(100 * r["peak_act"] / vmax)
        parts.append('<div class="row">')
        parts.append(
            f'<div><span class="label">{html.escape(r["label"])}</span>'
            f'<span class="peak">peak={r["peak_act"]:.2f} at pos {r["peak_idx"]}</span></div>'
        )
        parts.append(f'<div class="bar" style="width:{bar_w}%"></div>')
        parts.append(render_context(r["token_strs"], r["acts"], r["peak_idx"], vmax))
        parts.append('</div>')
    parts.append('</body></html>')

    out_path.write_text("\n".join(parts))


def main():
    out_dir = Path(REPO_ROOT / "e5/interp/results/p5_dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p5] device={device}", flush=True)

    nn_model, raw_model, _ = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt",
        device=device)
    sae_ar, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt", device)
    sae_diff, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt", device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    overlap = json.loads((REPO_ROOT / "e5/interp/results/cross_head_overlap.json").read_text())
    bridge_pairs = overlap["top20_matched_pairs"]

    # Expanded prompt corpus for richer top contexts
    prompts = [
        ("janet", "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 each. How much does she make daily?\nAnswer:"),
        ("baker", "Question: A baker makes 240 cookies and packs them in boxes of 12 each. How many boxes does he need?\nAnswer:"),
        ("apples", "Question: 96 apples are shared among 8 children equally. How many apples does each child get?\nAnswer:"),
        ("age", "Question: Alice is 12 years old. Her brother is 4 years younger. In 5 years, how old will Alice's brother be?\nAnswer:"),
        ("history", "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 AD, stretching from Britain in the north to Mesopotamia in the east."),
        ("science", "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll molecules in their leaves."),
        ("geography", "The Amazon rainforest produces more than 20 percent of the world's oxygen supply through its dense vegetation across South America."),
        ("narrative", "She closed the book slowly and looked out at the rain blurring the garden lights."),
        ("math1", "The total cost is 50 - 12 + 7 = "),
        ("math2", "Let x = 3 + 4 * 2. Then x = "),
        ("math3", "John has 15 marbles. He gives 4 to Mary and 3 to Tom. 15 - 4 - 3 = "),
        ("math4", "If a = 5 and b = a + 2, then b = 7. So a + b = 5 + 7 = "),
        ("cs_dijkstra", "Dijkstra's algorithm finds the shortest path by relaxing edges in non-decreasing order of distance."),
        ("cs_recursion", "A recursive function calls itself with a smaller subproblem until it reaches the base case."),
        ("recipe", "Mix the flour, sugar, and eggs in a large bowl, then bake at 350 degrees for 25 minutes."),
        ("dialogue", "\"Where did you go?\" she asked. He shrugged and said \"Nowhere in particular.\""),
    ]

    # Pick: 6 top AR features + 6 top diff features (using global activation
    # mass across the corpus to choose).
    print("[p5] scanning for top features across corpus...", flush=True)
    total_ar = torch.zeros(sae_ar.d_features)
    total_diff = torch.zeros(sae_diff.d_features)
    for _lbl, text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        a_ar = collect_ln_f(raw_model, idx, "ar")
        a_di = collect_ln_f(raw_model, idx, "diff")
        total_ar += sae_ar.encode(a_ar).sum(dim=0).cpu().detach()
        total_diff += sae_diff.encode(a_di).sum(dim=0).cpu().detach()

    top_ar = total_ar.topk(6).indices.tolist()
    top_diff = total_diff.topk(6).indices.tolist()
    # Also include a couple of bridge features explicitly
    for p in bridge_pairs[:2]:
        if p["ar_feature"] not in top_ar:
            top_ar.append(p["ar_feature"])
        if p["diff_feature"] not in top_diff:
            top_diff.append(p["diff_feature"])
    print(f"[p5] AR features to render: {top_ar}", flush=True)
    print(f"[p5] diff features to render: {top_diff}", flush=True)

    gallery = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Sfumato SAE Dashboard</title>',
        '<style>body{font-family:-apple-system,Helvetica,sans-serif;max-width:880px;margin:30px auto;padding:0 18px;}'
        'h1{font-size:24px;}h2{font-size:18px;color:#555;border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:30px;}'
        'a{display:block;padding:10px 14px;margin:4px 0;background:#f3f4f6;border-radius:6px;text-decoration:none;color:#111;}'
        'a:hover{background:#e5e7eb;}'
        '.badge{font-size:11px;padding:2px 6px;border-radius:4px;margin-left:8px;}'
        '</style></head><body>',
        '<h1>Sfumato F10 SAE feature dashboard</h1>',
        '<p>Per-feature pages with token-level activation highlighting on 16 prompts'
        ' (GSM8K, FineWeb prose, math, code, recipe, dialogue). Trained on 200M FineWeb'
        ' tokens with TopK k=64, d_features=16384.</p>',
        '<p><b>BRIDGE</b> tags mark features that have a high-cosine cross-head sibling'
        ' (P.2 top-20 matched pairs, cosine ≥ 0.85). These are universal content features'
        ' both heads use.</p>',
        '<h2>AR-head features</h2>',
    ]
    bridge_ar_ids = {p["ar_feature"] for p in bridge_pairs}
    bridge_diff_ids = {p["diff_feature"] for p in bridge_pairs}
    for fid in top_ar:
        page = out_dir / f"feat_ar_{fid}.html"
        build_feature_page(fid, "ar", sae_ar, raw_model, tok, prompts, bridge_pairs, page)
        is_bridge = fid in bridge_ar_ids
        badge = '<span class="badge" style="background:#7c3aed;color:white;">BRIDGE</span>' if is_bridge else ""
        gallery.append(f'<a href="{page.name}">AR feature #{fid}{badge}</a>')
        print(f"[p5] wrote {page.name}", flush=True)

    gallery.append('<h2>diff-head features</h2>')
    for fid in top_diff:
        page = out_dir / f"feat_diff_{fid}.html"
        build_feature_page(fid, "diff", sae_diff, raw_model, tok, prompts, bridge_pairs, page)
        is_bridge = fid in bridge_diff_ids
        badge = '<span class="badge" style="background:#7c3aed;color:white;">BRIDGE</span>' if is_bridge else ""
        gallery.append(f'<a href="{page.name}">diff feature #{fid}{badge}</a>')
        print(f"[p5] wrote {page.name}", flush=True)

    gallery.append('</body></html>')
    (out_dir / "index.html").write_text("\n".join(gallery))
    print(f"\n[p5] OPEN: file://{out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
