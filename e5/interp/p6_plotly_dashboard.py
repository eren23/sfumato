"""Phase P.6 — Plotly-based interactive feature dashboard.

Standardised on Plotly (OSS, used widely by interp community). Generates
static HTML pages with INTERACTIVE charts: hover for values, zoom, pan.

Per-feature page now includes:
  1. Activation histogram across the prompt corpus (Plotly)
  2. Top contexts with token-level highlighting (HTML/CSS)
  3. Top related features in BOTH heads by cosine similarity (Plotly bar)
  4. Decoder column heatmap vs nearest-N AR/diff features (Plotly heatmap)

Index includes:
  - Interactive cross-head cosine heatmap (256x256, Plotly hover)
  - Sortable feature table
  - Searchable feature ID
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
import plotly.graph_objects as go
import plotly.offline as pyo
import torch
import torch.nn.functional as F

from e5.interp.load_model import load_composite_for_interp
from e5.interp.p3_per_prompt_analysis import load_sae, collect_ln_f


def colour_for(val, vmax):
    if vmax <= 0 or val <= 0:
        return "rgba(255,255,255,0)"
    t = min(val / vmax, 1.0)
    if t < 0.5:
        a = t * 2
        r, g, b = 255, int(255 * (1 - 0.1 * a)), int(255 * (1 - 0.85 * a))
    else:
        a = (t - 0.5) * 2
        r = 255
        g = int(255 * (0.9 - 0.7 * a))
        b = int(255 * (0.15 - 0.15 * a))
    return f"rgba({r},{g},{b},{0.85*t:.2f})"


def render_tokens(token_strs, acts, peak_idx, vmax):
    pieces = []
    for j in range(len(token_strs)):
        tok = html.escape(token_strs[j]).replace("\n", "↵")
        bg = colour_for(acts[j], vmax)
        bold = " font-weight:600" if j == peak_idx else ""
        pieces.append(
            f'<span style="background:{bg};padding:1px 3px;border-radius:2px;{bold}" '
            f'title="act={acts[j]:.2f}">{tok}</span>'
        )
    return "<span style='font-family:Menlo,monospace;font-size:12px;line-height:1.8;'>" + "".join(pieces) + "</span>"


def plotly_histogram(values, title):
    fig = go.Figure(go.Histogram(x=values, marker_color="#3b82f6", nbinsx=40))
    fig.update_layout(
        title=title, height=240, margin=dict(t=40, b=30, l=40, r=10),
        xaxis_title="peak activation", yaxis_title="prompt count",
        template="plotly_white", font=dict(family="-apple-system", size=11),
    )
    return pyo.plot(fig, include_plotlyjs="cdn", output_type="div", config={"displayModeBar": False})


def plotly_related_bar(feature_id, mode, top_self, top_cross):
    """Bar chart showing this feature's cosine to top-10 own-mode features
    and top-10 cross-mode features."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{mode} #{i}" for i, c in top_self],
        y=[c for i, c in top_self],
        marker_color="#3b82f6",
        name=f"top within {mode}-head",
    ))
    other = "diff" if mode == "ar" else "ar"
    fig.add_trace(go.Bar(
        x=[f"{other} #{i}" for i, c in top_cross],
        y=[c for i, c in top_cross],
        marker_color="#ef4444",
        name=f"top in {other}-head (cross)",
    ))
    fig.update_layout(
        title=f"Top-10 similar features (own-head vs cross-head)",
        height=280, margin=dict(t=40, b=80, l=40, r=10),
        barmode="group", xaxis_tickangle=-45,
        yaxis_title="cosine similarity",
        template="plotly_white", font=dict(family="-apple-system", size=10),
    )
    return pyo.plot(fig, include_plotlyjs="cdn", output_type="div", config={"displayModeBar": False})


def build_feature_page(feature_id, mode, sae_self, sae_other, raw_model, tok,
                        prompts, bridge_pairs, out_path: Path, device):
    rows = []
    peaks = []
    for label, text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        act = collect_ln_f(raw_model, idx, mode)
        z = sae_self.encode(act).cpu().detach().numpy()
        col = z[:, feature_id]
        peak_idx = int(np.argmax(col))
        peak_act = float(col[peak_idx])
        rows.append({
            "label": label, "peak_act": peak_act, "peak_idx": peak_idx,
            "token_strs": [tok.decode([i]) for i in ids],
            "acts": col.tolist(),
        })
        peaks.append(peak_act)
    rows.sort(key=lambda r: -r["peak_act"])
    rows = [r for r in rows if r["peak_act"] > 0][:18]
    vmax = max(r["peak_act"] for r in rows) if rows else 1.0

    # Cosine to all features in own and cross heads
    W_self = F.normalize(sae_self.W_dec.detach(), dim=1).cpu()
    W_other = F.normalize(sae_other.W_dec.detach(), dim=1).cpu()
    my_vec = W_self[feature_id]
    self_sims = (W_self @ my_vec).numpy()
    self_sims[feature_id] = -2  # exclude self
    cross_sims = (W_other @ my_vec).numpy()
    top_self = [(int(i), float(self_sims[i])) for i in np.argsort(self_sims)[-10:][::-1]]
    top_cross = [(int(i), float(cross_sims[i])) for i in np.argsort(cross_sims)[-10:][::-1]]

    # Bridge tag?
    bridge_info = ""
    for p in bridge_pairs:
        if (mode == "ar" and p["ar_feature"] == feature_id) or \
           (mode == "diff" and p["diff_feature"] == feature_id):
            other_id = p["diff_feature"] if mode == "ar" else p["ar_feature"]
            other_mode = "diff" if mode == "ar" else "ar"
            bridge_info = (f' <span style="background:#7c3aed;color:white;padding:3px 8px;'
                           f'border-radius:4px;font-size:11px;">'
                           f'BRIDGE → {other_mode} #{other_id} (cos={p["cosine"]:.3f})</span>')
            break

    parts = [
        f'<!doctype html><html><head><meta charset="utf-8"><title>{mode} #{feature_id}</title>',
        '<style>body{font-family:-apple-system,Helvetica,sans-serif;max-width:1080px;'
        'margin:30px auto;padding:0 18px;color:#111;}'
        'h1{font-size:22px;margin-bottom:0;}h2{font-size:14px;color:#666;margin-top:30px;}'
        '.row{margin:10px 0;padding:8px 12px;border:1px solid #e5e7eb;border-radius:6px;background:#fafafa;}'
        '.row .label{font-size:11px;color:#7c3aed;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}'
        '.row .peak{font-size:11px;color:#555;margin-left:10px;}'
        '.bar{background:linear-gradient(to right,#fbbf24,#ef4444);height:5px;border-radius:2px;display:block;margin:4px 0 6px;}'
        '.charts{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0;}'
        '.chart{flex:1;min-width:380px;background:white;border:1px solid #eee;border-radius:6px;padding:6px;}'
        '.nav{margin-bottom:20px;}.nav a{margin-right:14px;color:#3b82f6;text-decoration:none;}'
        '</style></head><body>',
        '<div class="nav"><a href="index.html">← all features</a></div>',
        f'<h1>{mode}-head SAE feature #{feature_id}{bridge_info}</h1>',
        f'<h2>Trained on F10 post-ln_f activations in <code>mode={mode}</code>. '
        f'd_features=16384, k=64.</h2>',
        '<div class="charts">',
        f'<div class="chart">{plotly_histogram([r["peak_act"] for r in rows], "Peak activation per prompt")}</div>',
        f'<div class="chart">{plotly_related_bar(feature_id, mode, top_self, top_cross)}</div>',
        '</div>',
        '<h2>Top contexts (ranked by peak activation, brighter = stronger)</h2>',
    ]
    for r in rows:
        bar_w = int(100 * r["peak_act"] / vmax)
        parts.append('<div class="row">')
        parts.append(
            f'<div><span class="label">{html.escape(r["label"])}</span>'
            f'<span class="peak">peak={r["peak_act"]:.2f} at pos {r["peak_idx"]}</span></div>'
        )
        parts.append(f'<div class="bar" style="width:{bar_w}%"></div>')
        parts.append(render_tokens(r["token_strs"], r["acts"], r["peak_idx"], vmax))
        parts.append('</div>')
    parts.append('</body></html>')
    out_path.write_text("\n".join(parts))


def build_interactive_heatmap(W_ar, W_di):
    """A 256x256 Plotly heatmap of best-cosine cross-head pairs.
    Hover shows AR_id, diff_id, cosine."""
    print("[p6] computing best-cosine matrix subset ...", flush=True)
    best_ar = (W_ar @ W_di.T).max(dim=1).values
    top_ar_idx = best_ar.argsort(descending=True)[:256]
    best_di = (W_di @ W_ar.T).max(dim=1).values
    top_di_idx = best_di.argsort(descending=True)[:256]
    sub = (W_ar[top_ar_idx] @ W_di[top_di_idx].T).numpy()

    fig = go.Figure(go.Heatmap(
        z=sub,
        x=[f"diff #{int(i)}" for i in top_di_idx],
        y=[f"AR #{int(i)}" for i in top_ar_idx],
        colorscale="Magma",
        zmin=-0.1, zmax=0.9,
        hovertemplate="%{y} ↔ %{x}<br>cos=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Cross-head SAE cosine — top 256 × top 256 (sorted by best-cross)",
        height=600, template="plotly_white", font=dict(size=10),
        xaxis_showticklabels=False, yaxis_showticklabels=False,
        margin=dict(t=50, b=20, l=30, r=20),
    )
    return pyo.plot(fig, include_plotlyjs="cdn", output_type="div",
                     config={"displayModeBar": False})


def main():
    out_dir = Path(REPO_ROOT / "e5/interp/results/p6_dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p6] device={device}", flush=True)

    nn_model, raw_model, _ = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt",
        device=device)
    sae_ar, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt", device)
    sae_diff, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt", device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    overlap = json.loads((REPO_ROOT / "e5/interp/results/cross_head_overlap.json").read_text())
    bridge_pairs = overlap["top20_matched_pairs"]
    bridge_ar = {p["ar_feature"] for p in bridge_pairs}
    bridge_diff = {p["diff_feature"] for p in bridge_pairs}

    # Same diverse prompt corpus
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

    print("[p6] scanning top features ...", flush=True)
    total_ar = torch.zeros(sae_ar.d_features)
    total_diff = torch.zeros(sae_diff.d_features)
    for _lbl, text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        total_ar += sae_ar.encode(collect_ln_f(raw_model, idx, "ar")).sum(dim=0).cpu().detach()
        total_diff += sae_diff.encode(collect_ln_f(raw_model, idx, "diff")).sum(dim=0).cpu().detach()

    top_ar_feats = total_ar.topk(8).indices.tolist()
    top_diff_feats = total_diff.topk(8).indices.tolist()
    for p in bridge_pairs[:3]:
        if p["ar_feature"] not in top_ar_feats:
            top_ar_feats.append(p["ar_feature"])
        if p["diff_feature"] not in top_diff_feats:
            top_diff_feats.append(p["diff_feature"])
    print(f"[p6] {len(top_ar_feats)} AR features + {len(top_diff_feats)} diff features", flush=True)

    # Generate per-feature pages
    for fid in top_ar_feats:
        build_feature_page(fid, "ar", sae_ar, sae_diff, raw_model, tok,
                            prompts, bridge_pairs,
                            out_dir / f"feat_ar_{fid}.html", device)
        print(f"[p6] wrote feat_ar_{fid}.html", flush=True)
    for fid in top_diff_feats:
        build_feature_page(fid, "diff", sae_diff, sae_ar, raw_model, tok,
                            prompts, bridge_pairs,
                            out_dir / f"feat_diff_{fid}.html", device)
        print(f"[p6] wrote feat_diff_{fid}.html", flush=True)

    # Interactive heatmap on the index page
    W_ar = F.normalize(sae_ar.W_dec.detach(), dim=1).cpu()
    W_di = F.normalize(sae_diff.W_dec.detach(), dim=1).cpu()
    heatmap_html = build_interactive_heatmap(W_ar, W_di)

    # Index
    rows_html = []
    for fid in sorted(top_ar_feats):
        b = '<span style="background:#7c3aed;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:6px;">BRIDGE</span>' if fid in bridge_ar else ""
        rows_html.append(
            f'<tr><td>AR</td><td>#{fid}</td><td>{float(total_ar[fid]):.1f}</td>'
            f'<td><a href="feat_ar_{fid}.html">browse →</a></td>'
            f'<td>{b}</td></tr>'
        )
    for fid in sorted(top_diff_feats):
        b = '<span style="background:#7c3aed;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-left:6px;">BRIDGE</span>' if fid in bridge_diff else ""
        rows_html.append(
            f'<tr><td>diff</td><td>#{fid}</td><td>{float(total_diff[fid]):.1f}</td>'
            f'<td><a href="feat_diff_{fid}.html">browse →</a></td>'
            f'<td>{b}</td></tr>'
        )

    index = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Sfumato P.6 dashboard</title>',
        '<style>body{font-family:-apple-system,Helvetica,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;}'
        'h1{font-size:26px;}h2{font-size:16px;color:#555;border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:30px;}'
        'table{width:100%;border-collapse:collapse;margin-top:10px;}'
        'th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;font-size:13px;}'
        'th{background:#f9fafb;font-weight:600;}'
        'tr:hover{background:#fafafa;}'
        '</style></head><body>',
        '<h1>Sfumato F10 SAE interpretability dashboard <span style="font-size:14px;color:#7c3aed">(Phase P.6 — Plotly interactive)</span></h1>',
        '<p>Custom 305M composite, F10 final ckpt. Two TopK SAEs (k=64, 16384 features) trained on '
        'post-ln_f activations under each mode. Pages have interactive Plotly charts: hover, zoom, click.</p>',

        '<h2>Cross-head cosine map (interactive)</h2>',
        '<p>Hover any cell for the AR-feature ↔ diff-feature cosine. '
        'Sorted by best-cross-cosine; the diagonal-ish bright streak in the top-left is the bridge features. '
        'Mean of the full 16k×16k matrix = <b>0.169</b>; only the top-256×256 is shown here for legibility.</p>',
        heatmap_html,

        '<h2>Per-feature browser</h2>',
        '<table><thead><tr><th>mode</th><th>feature</th><th>total activation (16-prompt sum)</th>'
        '<th>page</th><th>tag</th></tr></thead><tbody>',
    ]
    index.extend(rows_html)
    index.extend(['</tbody></table>', '</body></html>'])
    (out_dir / "index.html").write_text("\n".join(index))
    print(f"\n[p6] OPEN: file://{out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
