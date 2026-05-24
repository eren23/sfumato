"""Phase P.7 — attribution-graph-style visualisation.

Inspired by coder-interp-tap / Goodfire / Anthropic circuit-tracer style:
  - Y axis: layers, stacked top-to-bottom (deep → shallow)
  - X axis: token positions in the prompt, columns
  - At each (layer, token) cell: small circles for the top-N firing SAE
    features at that position. Circle size = activation magnitude.
    Colour = feature identity (consistent per feature across layers).
  - Edges: between same-feature circles across adjacent layers, so a
    persistent feature shows up as a vertical thread of dots connected
    by a faint line.

Each prompt gets its own SVG attribution graph rendered into an HTML
page. AR-mode-only for this first pass (we have 10 layer SAEs in
AR mode: block 6 through block 15).

Output:
  e5/interp/results/p7_attr/
    index.html
    attr_{prompt_label}.html
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
from e5.interp.topk_sae import TopKSAE
from e5.interp.p3_per_prompt_analysis import load_sae


def feature_colour(fid: int) -> str:
    """Stable colour per feature id (HSL hash → hex)."""
    h = (fid * 37) % 360
    s = 70
    l = 58
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def collect_layer_activation(raw_model, idx, layer_idx: int):
    """Forward F10 in AR mode, capture residual after block `layer_idx`."""
    store = {"act": None}
    h = raw_model.blocks[layer_idx].register_forward_hook(
        lambda m, i, o: store.__setitem__("act", o if not isinstance(o, tuple) else o[0])
    )
    try:
        with torch.no_grad():
            raw_model(idx, mode="ar")
    finally:
        h.remove()
    return store["act"][0].detach().float()  # (T, d)


def build_attribution_svg(prompt_label, text, raw_model, tok, layer_saes,
                          top_per_cell=4, max_features=12):
    """
    layer_saes: list of (layer_idx, sae) tuples, sorted by layer ascending.
    Returns SVG string + a feature legend dict.
    """
    device = next(raw_model.parameters()).device
    ids = tok.encode(text, add_special_tokens=False)
    # Clip long prompts for legibility
    if len(ids) > 36:
        ids = ids[:36]
    T = len(ids)
    idx_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    token_strs = [tok.decode([i]).replace("\n", "↵") for i in ids]

    # Compute per-layer per-token sparse activations
    per_layer = []
    feature_global_total = {}
    for layer_idx, sae in layer_saes:
        act = collect_layer_activation(raw_model, idx_tensor, layer_idx)  # (T, d)
        z = sae.encode(act).cpu().detach().numpy()  # (T, d_features)
        # Per-token, pick top-K non-zero feature ids
        cells = []
        for t in range(T):
            col = z[t]
            top_idx = np.argsort(col)[-top_per_cell:][::-1]
            cells.append([(int(fid), float(col[fid])) for fid in top_idx if col[fid] > 0])
            for fid, val in cells[-1]:
                feature_global_total[fid] = feature_global_total.get(fid, 0.0) + val
        per_layer.append({"layer": layer_idx, "cells": cells})

    # Pick the top-K globally-firing features to colour-code consistently
    top_features = sorted(feature_global_total.items(), key=lambda x: -x[1])[:max_features]
    top_feature_ids = {fid: i for i, (fid, _) in enumerate(top_features)}
    colour_palette = [
        "#f59e0b", "#22d3ee", "#a855f7", "#10b981", "#ef4444",
        "#fbbf24", "#3b82f6", "#ec4899", "#84cc16", "#06b6d4",
        "#f97316", "#8b5cf6",
    ]
    colour_by_fid = {fid: colour_palette[i % len(colour_palette)]
                     for i, (fid, _) in enumerate(top_features)}

    # SVG layout
    margin_top = 60
    margin_bottom = 70
    margin_left = 70
    margin_right = 30
    cell_w = max(64, 1100 // max(T, 1))
    cell_h = 56
    L = len(per_layer)
    width = margin_left + cell_w * T + margin_right
    height = margin_top + cell_h * L + margin_bottom

    svg = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="display:block;width:100%;max-width:{width}px;height:auto;">']

    # background
    svg.append(f'<rect width="{width}" height="{height}" fill="#0a0e1a"/>')

    # subtle column dividers
    for t in range(T + 1):
        x = margin_left + t * cell_w
        svg.append(f'<line x1="{x}" y1="{margin_top - 10}" x2="{x}" '
                   f'y2="{height - margin_bottom + 10}" stroke="#1a1f2e" stroke-width="1"/>')

    # token labels (top)
    for t in range(T):
        x = margin_left + t * cell_w + cell_w / 2
        tok_s = html.escape(token_strs[t]).strip() or "·"
        if len(tok_s) > 9:
            tok_s = tok_s[:8] + "…"
        svg.append(
            f'<text x="{x}" y="{margin_top - 18}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="11" '
            f'fill="#9ca3af">{tok_s}</text>'
        )

    # layer labels (left)
    for li, layer in enumerate(per_layer):
        y = margin_top + li * cell_h + cell_h / 2
        svg.append(
            f'<text x="{margin_left - 14}" y="{y + 4}" text-anchor="end" '
            f'font-family="JetBrains Mono, monospace" font-size="12" '
            f'fill="#9ca3af">L{layer["layer"]:02d}</text>'
        )

    # circles per (layer, token)
    # also: edges between adjacent layers when same feature appears
    circle_positions = {}  # (layer_i, token, fid) → (x, y)
    max_act = 0.0
    for li, layer in enumerate(per_layer):
        for t, cell in enumerate(layer["cells"]):
            for fid, val in cell:
                max_act = max(max_act, val)
    if max_act <= 0:
        max_act = 1.0

    # Edges first so they sit beneath circles
    for li in range(len(per_layer) - 1):
        for t in range(T):
            top_cell = per_layer[li]["cells"][t]
            bot_cell = per_layer[li + 1]["cells"][t]
            top_ids = {fid for fid, _ in top_cell}
            bot_ids = {fid for fid, _ in bot_cell}
            common = top_ids & bot_ids
            for fid in common:
                if fid not in colour_by_fid:
                    continue
                color = colour_by_fid[fid]
                y_top = margin_top + li * cell_h + cell_h / 2
                y_bot = margin_top + (li + 1) * cell_h + cell_h / 2
                x = margin_left + t * cell_w + cell_w / 2
                svg.append(
                    f'<line x1="{x}" y1="{y_top}" x2="{x}" y2="{y_bot}" '
                    f'stroke="{color}" stroke-width="1.2" stroke-opacity="0.32"/>'
                )

    # Now circles
    for li, layer in enumerate(per_layer):
        for t, cell in enumerate(layer["cells"]):
            n = len(cell)
            if n == 0:
                continue
            # Lay out 1..top_per_cell circles inside cell, offset slightly
            for ci, (fid, val) in enumerate(cell):
                r = 2 + 9 * (val / max_act) ** 0.6
                cx = margin_left + t * cell_w + (cell_w * (ci + 1)) / (n + 1)
                cy = margin_top + li * cell_h + cell_h / 2
                color = colour_by_fid.get(fid, "#3a4255")
                opacity = 0.85 if fid in top_feature_ids else 0.45
                title = f"L{layer['layer']} · pos {t} · #{fid} · act {val:.2f}"
                svg.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" '
                    f'fill-opacity="{opacity}" stroke="{color}" stroke-opacity="0.9" '
                    f'stroke-width="1"><title>{html.escape(title)}</title></circle>'
                )

    # legend (bottom)
    leg_y = height - 50
    svg.append(
        f'<text x="{margin_left}" y="{leg_y}" font-family="JetBrains Mono, monospace" '
        f'font-size="11" fill="#9ca3af">TOP FEATURES</text>'
    )
    for i, (fid, total) in enumerate(top_features[:8]):
        col_x = margin_left + (i % 4) * 260
        row_y = leg_y + 14 + (i // 4) * 16
        color = colour_by_fid[fid]
        svg.append(f'<circle cx="{col_x + 6}" cy="{row_y - 4}" r="5" '
                   f'fill="{color}" fill-opacity="0.9"/>')
        svg.append(
            f'<text x="{col_x + 18}" y="{row_y}" font-family="JetBrains Mono, monospace" '
            f'font-size="11" fill="#e5e7eb">#{fid} · Σ {total:.1f}</text>'
        )

    svg.append('</svg>')
    return "\n".join(svg), top_features, colour_by_fid


GRAPH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
:root {
  --bg: #0a0e1a; --surface: #131725; --surface-2: #1a1f2e;
  --border: #262b3d; --border-lit: #3a4255;
  --text: #e5e7eb; --text-dim: #9ca3af; --text-muted: #6b7280;
  --gold: #fbbf24;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 14px; line-height: 1.55; -webkit-font-smoothing: antialiased; }
.shell { max-width: 1240px; margin: 0 auto; padding: 48px 32px 80px; }
.topbar { display:flex; justify-content:space-between; padding-bottom:12px; border-bottom:1px solid var(--border); margin-bottom:36px; font-size:11px; letter-spacing:0.18em; text-transform:uppercase; color:var(--text-muted); }
.topbar a { color: var(--text-dim); text-decoration: none; }
.topbar a:hover { color: var(--gold); }
.logo { font-family: 'Fraunces', serif; font-style: italic; font-size: 18px; color: var(--gold); text-transform: none; letter-spacing: -0.01em; }
h1 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 36px; letter-spacing: -0.015em; margin: 0 0 6px; }
.subtitle { font-family: 'Fraunces', serif; font-style: italic; color: var(--text-dim); font-size: 17px; margin-bottom: 22px; }
.toolbar { display:flex; gap: 10px; flex-wrap: wrap; padding: 14px 16px; background: var(--surface); border: 1px solid var(--border); margin-bottom: 22px; }
.toolbar .lab { font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-muted); margin-right: 8px; align-self: center; }
.toolbar a {
  display: inline-block; padding: 7px 12px; font-size: 12px; font-family: 'JetBrains Mono', monospace;
  color: var(--text-dim); border: 1px solid var(--border); text-decoration: none;
  letter-spacing: 0.04em;
}
.toolbar a:hover { color: var(--gold); border-color: var(--border-lit); background: var(--surface-2); }
.toolbar a.active { color: var(--bg); background: var(--gold); border-color: var(--gold); }

.graph-card { background: var(--surface); border: 1px solid var(--border); padding: 24px 26px; margin-bottom: 22px; }
.graph-meta { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px; font-size: 11px; color: var(--text-muted); letter-spacing: 0.06em; }
.graph-meta b { color: var(--text-dim); }
.prompt-block { background: rgba(0,0,0,0.30); border-left: 2px solid var(--gold); padding: 12px 16px;
  font-family: 'Fraunces', serif; font-style: italic; font-size: 15px; color: var(--text-dim); margin-bottom: 22px; }
.legend-help { display: flex; gap: 24px; font-size: 11px; color: var(--text-muted); letter-spacing: 0.06em; margin-top: 16px; align-items: center; }
.legend-help .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.foot { margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-muted); letter-spacing: 0.14em; text-transform: uppercase; display: flex; justify-content: space-between; }
.foot a { color: var(--text-dim); text-decoration: none; }
</style>
"""


def main():
    out_dir = Path(REPO_ROOT / "e5/interp/results/p7_attr")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p7] device={device}", flush=True)

    nn_model, raw_model, _ = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt",
        device=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    layer_indices = list(range(6, 16))
    layer_saes = []
    for L in layer_indices:
        sae_path = REPO_ROOT / f"e5/interp/saes/block_{L}_ar/sae.pt"
        if not sae_path.exists():
            print(f"[p7] missing SAE for block.{L}.ar; skipping", flush=True)
            continue
        sae, _ = load_sae(sae_path, device)
        layer_saes.append((L, sae))
    print(f"[p7] loaded {len(layer_saes)} layer SAEs", flush=True)

    prompts = [
        ("janet", "Question: Janet's ducks lay 16 eggs per day. How much does she make daily?\nAnswer:"),
        ("baker", "Question: A baker makes 240 cookies, packs in boxes of 12 each. Boxes?\nAnswer:"),
        ("apples", "Question: 96 apples shared among 8 children equally. Each gets?\nAnswer:"),
        ("history", "The Roman Empire reached its peak under Trajan in 117 AD, from Britain to Mesopotamia."),
        ("science", "Photosynthesis uses sunlight and chlorophyll to synthesize food in plant leaves."),
        ("math_chain", "Let x = 3 + 4 * 2. Then x = "),
        ("counting", "John has 15 marbles. He gives 4 to Mary and 3 to Tom. 15 - 4 - 3 = "),
        ("recipe", "Mix the flour, sugar, and eggs in a large bowl, then bake at 350 degrees for 25 minutes."),
        ("cs_dijkstra", "Dijkstra's algorithm finds the shortest path by relaxing edges in non-decreasing order."),
        ("dialogue", "\"Where did you go?\" she asked. He shrugged and said \"Nowhere in particular.\""),
    ]

    nav_links = []
    for label, _ in prompts:
        nav_links.append(f'<a href="attr_{label}.html">{label}</a>')

    for label, text in prompts:
        print(f"[p7] building {label} ...", flush=True)
        svg, top_features, _ = build_attribution_svg(label, text, raw_model, tok, layer_saes)
        nav_html = ['<div class="toolbar"><span class="lab">prompts</span>']
        for nl in nav_links:
            nav_html.append(nl.replace('href="', f'href="').replace(
                f'attr_{label}.html', f'attr_{label}.html" class="active'
            ) if f'attr_{label}.html' in nl else nl)
        nav_html.append('</div>')

        # cleaner version of nav (with active class only on current)
        nav2 = ['<div class="toolbar"><span class="lab">prompts</span>']
        for plabel, _ in prompts:
            cls = ' class="active"' if plabel == label else ''
            nav2.append(f'<a href="attr_{plabel}.html"{cls}>{plabel}</a>')
        nav2.append('<a href="index.html" style="margin-left:auto; color:var(--gold);">↩ index</a>')
        nav2.append('</div>')

        body = [
            f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>attr · {label}</title>',
            GRAPH_CSS,
            '</head><body><div class="shell">',
            '<div class="topbar"><span class="logo">Sfumato</span>'
            '<span>Attribution graph · AR mode · layers 6-15</span></div>',
            f'<h1>Attribution graph: <span style="color:var(--gold);font-style:italic;">{html.escape(label)}</span></h1>',
            '<div class="subtitle">Top-firing SAE features per (layer, token). Circle size = activation strength. Threads = same feature persisting across adjacent layers.</div>',
            *nav2,
            '<div class="graph-card">',
            f'<div class="prompt-block">{html.escape(text)}</div>',
            f'<div class="graph-meta"><b>Hookpoint:</b> block.L.ar (post-residual) &nbsp;&nbsp;'
            f'<b>SAE:</b> TopK k=64 · d_features=16384 &nbsp;&nbsp;'
            f'<b>Layers shown:</b> {", ".join(f"L{L}" for L, _ in layer_saes)} &nbsp;&nbsp;'
            f'<b>Cells:</b> top-4 features per token</div>',
            svg,
            '<div class="legend-help">'
            '<span><span class="dot" style="background:#9ca3af"></span>circle size ∝ activation</span>'
            '<span><span class="dot" style="background:#fbbf24"></span>thread = same feature firing in next layer</span>'
            '<span>hover any circle for the feature id and activation</span>'
            '</div>',
            '</div>',
            '<div class="foot"><span>Sfumato · P.7 attribution graph</span>'
            '<a href="index.html">↩ all prompts</a></div>',
            '</div></body></html>'
        ]
        (out_dir / f"attr_{label}.html").write_text("\n".join(body))

    # Index
    cards = []
    for label, text in prompts:
        cards.append(
            f'<a href="attr_{label}.html" class="card">'
            f'<div class="card-label">{html.escape(label)}</div>'
            f'<div class="card-prompt">{html.escape(text[:120])}{"…" if len(text) > 120 else ""}</div>'
            f'</a>'
        )
    index_body = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>Sfumato · Attribution graphs</title>',
        GRAPH_CSS,
        '<style>'
        '.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; margin-top: 22px; }'
        '.card { display: block; background: var(--surface); border: 1px solid var(--border); padding: 18px 20px;'
        ' text-decoration: none; color: inherit; transition: border-color 0.15s, transform 0.15s, background 0.15s; }'
        '.card:hover { border-color: var(--gold); transform: translateY(-1px); background: var(--surface-2); }'
        '.card-label { font-family: "Fraunces", serif; font-weight: 600; font-size: 22px; color: var(--gold); margin-bottom: 6px; letter-spacing: -0.01em; }'
        '.card-prompt { font-family: "JetBrains Mono", monospace; font-size: 12px; color: var(--text-dim); line-height: 1.5; }'
        '</style>',
        '</head><body><div class="shell">',
        '<div class="topbar"><span class="logo">Sfumato</span>'
        '<span>Attribution graphs · field manual</span></div>',
        '<h1>Attribution graphs</h1>',
        '<div class="subtitle">A per-prompt circuit-tracer-style view. Layers stacked vertical, '
        'tokens horizontal, top-firing SAE features rendered as circles. Lines connect the same feature '
        'across adjacent layers, so persistent features form a vertical thread.</div>',
        '<div class="grid">',
        *cards,
        '</div>',
        '<div class="foot" style="margin-top:60px;"><span>Sfumato · P.7</span>'
        f'<span>{len(prompts)} prompts × {len(layer_saes)} layers</span>'
        '<a href="../p6_dashboard/index.html">↩ to dossier</a></div>',
        '</div></body></html>'
    ]
    (out_dir / "index.html").write_text("\n".join(index_body))
    print(f"\n[p7] OPEN: file://{out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
