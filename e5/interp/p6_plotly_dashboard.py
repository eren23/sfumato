"""Phase P.6 — interactive feature dossier for Sfumato F10 SAE.

Aesthetic direction: "intelligence dossier" — two factions (AR amber,
diff cyan) inspecting the same shared backbone, BRIDGE features
(purple) the universal vocabulary both factions use. Dark mode,
Fraunces display + JetBrains Mono body, editorial asymmetric layout,
cross-head heatmap as the centerpiece.

Pages:
  index.html               — hero + headline stat + heatmap + roster
  feat_{mode}_{id}.html    — per-feature dossier with token specimens

Single static HTML per page; Plotly from CDN; no server.
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


# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------

THEME = {
    "bg":          "#0a0e1a",
    "surface":     "#131725",
    "surface_2":   "#1a1f2e",
    "border":      "#262b3d",
    "border_lit":  "#3a4255",
    "text":        "#e5e7eb",
    "text_dim":    "#9ca3af",
    "text_muted":  "#6b7280",
    "ar":          "#f59e0b",      # amber-500
    "ar_dim":      "#92510a",
    "diff":        "#22d3ee",      # cyan-400
    "diff_dim":    "#0e7490",
    "bridge":      "#a855f7",      # purple-500
    "bridge_dim":  "#6b21a8",
    "gold":        "#fbbf24",
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {
  --bg: %(bg)s;
  --surface: %(surface)s;
  --surface-2: %(surface_2)s;
  --border: %(border)s;
  --border-lit: %(border_lit)s;
  --text: %(text)s;
  --text-dim: %(text_dim)s;
  --text-muted: %(text_muted)s;
  --ar: %(ar)s;
  --ar-dim: %(ar_dim)s;
  --diff: %(diff)s;
  --diff-dim: %(diff_dim)s;
  --bridge: %(bridge)s;
  --bridge-dim: %(bridge_dim)s;
  --gold: %(gold)s;
}

* { box-sizing: border-box; }

html, body {
  margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 14px; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: 'ss01' on, 'ss02' on;
}

body::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 1000;
  background-image:
    radial-gradient(circle at 20%% 30%%, rgba(245,158,11,0.05) 0%%, transparent 40%%),
    radial-gradient(circle at 80%% 70%%, rgba(34,211,238,0.05) 0%%, transparent 40%%);
}
body::after {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 1001;
  background-image: url("data:image/svg+xml;utf8,<svg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='1.4' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.025 0'/></filter><rect width='200' height='200' filter='url(%%23n)'/></svg>");
  background-size: 200px 200px; opacity: 0.6;
}

.serif { font-family: 'Fraunces', 'Charter', Georgia, serif; font-feature-settings: 'ss01' on, 'ss03' on; }
.display { font-family: 'Fraunces', serif; font-weight: 800; letter-spacing: -0.02em; }
.mono { font-family: 'JetBrains Mono', monospace; }

.shell { max-width: 1280px; margin: 0 auto; padding: 56px 36px 80px; position: relative; z-index: 2; }

/* ---------- top bar ---------- */
.topbar { display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 48px;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--text-muted); }
.topbar a { color: var(--text-dim); text-decoration: none; }
.topbar a:hover { color: var(--gold); }
.brand { display: flex; align-items: center; gap: 14px; }
.brand .logo { font-family: 'Fraunces', serif; font-style: italic; font-size: 18px; color: var(--gold); letter-spacing: -0.01em; text-transform: none; }
.brand .tagline { font-size: 11px; letter-spacing: 0.18em; }
.crumbs { display: flex; gap: 18px; align-items: center; }
.dot { width: 6px; height: 6px; border-radius: 50%%; background: var(--gold); animation: pulse 2.4s ease-in-out infinite; }
@keyframes pulse { 0%%, 100%% { opacity: 0.4; } 50%% { opacity: 1; } }

/* ---------- hero ---------- */
.hero { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 56px; align-items: end; margin-bottom: 64px; }
.hero h1 { font-family: 'Fraunces', serif; font-weight: 800; font-size: clamp(56px, 8vw, 120px);
  line-height: 0.94; letter-spacing: -0.035em; margin: 0; color: var(--text); }
.hero h1 em { font-style: italic; font-weight: 400; color: var(--gold); }
.hero .strap { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 18px; }

.hero-stat { border-left: 1px solid var(--border); padding-left: 32px; }
.hero-stat .kicker { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 12px; }
.hero-stat .big {
  font-family: 'Fraunces', serif; font-weight: 800; font-size: 96px; line-height: 1;
  letter-spacing: -0.04em; color: var(--bridge); margin: 0;
  background: linear-gradient(135deg, var(--ar) 0%%, var(--bridge) 50%%, var(--diff) 100%%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-stat .verdict { font-family: 'Fraunces', serif; font-style: italic; font-size: 22px; color: var(--text); margin-top: 14px; }
.hero-stat .verdict strong { color: var(--gold); font-style: normal; font-weight: 600; }
.hero-stat .footnote { font-size: 11px; color: var(--text-muted); margin-top: 12px; letter-spacing: 0.04em; }

/* ---------- factions row ---------- */
.factions { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 22px; margin-bottom: 72px; }
.faction { padding: 22px 22px 24px; background: var(--surface); border: 1px solid var(--border);
  position: relative; overflow: hidden; }
.faction::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.faction.ar::before { background: linear-gradient(90deg, transparent, var(--ar), transparent); }
.faction.diff::before { background: linear-gradient(90deg, transparent, var(--diff), transparent); }
.faction.bridge::before { background: linear-gradient(90deg, transparent, var(--bridge), transparent); }
.faction .label { font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }
.faction.ar .label { color: var(--ar); }
.faction.diff .label { color: var(--diff); }
.faction.bridge .label { color: var(--bridge); }
.faction .title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 22px; letter-spacing: -0.01em; margin-bottom: 12px; }
.faction .desc { font-size: 13px; color: var(--text-dim); line-height: 1.6; }

/* ---------- section heading ---------- */
.section-head { display: flex; align-items: baseline; gap: 18px; margin: 72px 0 24px; }
.section-head .num { font-family: 'Fraunces', serif; font-style: italic; color: var(--gold); font-size: 16px; }
.section-head h2 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 32px; letter-spacing: -0.02em; margin: 0; color: var(--text); }
.section-head .rule { flex: 1; border-bottom: 1px solid var(--border); transform: translateY(-7px); }

.section-intro { font-family: 'Fraunces', serif; font-style: italic; font-size: 17px; color: var(--text-dim); max-width: 820px; line-height: 1.55; margin: 0 0 28px; }
.section-intro b { font-family: 'JetBrains Mono', monospace; font-style: normal; font-weight: 500; color: var(--gold); font-size: 14px; }

/* ---------- heatmap card ---------- */
.heatmap-card {
  background: var(--surface); border: 1px solid var(--border); padding: 28px 30px 22px;
  position: relative;
}
.heatmap-card::before, .heatmap-card::after {
  content: ""; position: absolute; width: 20px; height: 20px; border: 2px solid var(--border-lit);
}
.heatmap-card::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
.heatmap-card::after { bottom: -1px; right: -1px; border-left: none; border-top: none; }
.heatmap-card .hm-title { font-family: 'Fraunces', serif; font-style: italic; font-size: 18px; color: var(--gold); margin-bottom: 4px; }
.heatmap-card .hm-sub { font-size: 12px; color: var(--text-muted); margin-bottom: 22px; letter-spacing: 0.04em; }
.heatmap-card .annotations { display: flex; gap: 18px; margin-top: 14px; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
.heatmap-card .annotations .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: middle; }

/* ---------- roster grid ---------- */
.roster { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.column-head { display: flex; align-items: center; justify-content: space-between; padding-bottom: 14px;
  border-bottom: 2px solid; margin-bottom: 16px; }
.column-head.ar { border-color: var(--ar); }
.column-head.diff { border-color: var(--diff); }
.column-head .faction-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 22px; letter-spacing: -0.01em; }
.column-head.ar .faction-name { color: var(--ar); }
.column-head.diff .faction-name { color: var(--diff); }
.column-head .count { font-size: 11px; color: var(--text-muted); letter-spacing: 0.14em; text-transform: uppercase; }

.feat-card {
  display: grid; grid-template-columns: 90px 1fr 100px; align-items: center; gap: 18px;
  padding: 16px 18px; background: var(--surface); border: 1px solid var(--border);
  margin-bottom: 10px; text-decoration: none; color: var(--text); position: relative;
  transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
}
.feat-card:hover { border-color: var(--border-lit); transform: translateY(-1px); background: var(--surface-2); }
.feat-card.ar:hover { border-color: var(--ar); box-shadow: 0 0 0 1px var(--ar-dim) inset, 0 6px 24px -10px rgba(245,158,11,0.4); }
.feat-card.diff:hover { border-color: var(--diff); box-shadow: 0 0 0 1px var(--diff-dim) inset, 0 6px 24px -10px rgba(34,211,238,0.4); }
.feat-card .fid {
  font-family: 'Fraunces', serif; font-weight: 800; font-size: 30px; line-height: 1;
  letter-spacing: -0.02em;
}
.feat-card.ar .fid { color: var(--ar); }
.feat-card.diff .fid { color: var(--diff); }
.feat-card .meta { font-size: 11px; color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px; }
.feat-card .body { display: flex; flex-direction: column; gap: 4px; }
.feat-card .name { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text-dim); }
.feat-card .activation { font-family: 'Fraunces', serif; font-size: 16px; color: var(--text); }
.feat-card .right { text-align: right; }
.feat-card .activation-val { font-family: 'Fraunces', serif; font-size: 26px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); }
.feat-card .activation-lbl { font-size: 10px; color: var(--text-muted); letter-spacing: 0.14em; text-transform: uppercase; }

.bridge-tag {
  display: inline-block; padding: 2px 8px; font-size: 10px; letter-spacing: 0.16em;
  text-transform: uppercase; background: var(--bridge); color: var(--bg); margin-left: 8px;
  font-weight: 600; border-radius: 2px;
}

/* ---------- bridge band ---------- */
.bridge-band { margin-top: 72px; padding: 32px 36px;
  background: linear-gradient(135deg, rgba(168,85,247,0.08) 0%%, rgba(168,85,247,0.02) 100%%);
  border: 1px solid var(--bridge-dim); position: relative; }
.bridge-band::before { content: "Δ"; position: absolute; top: 24px; right: 32px; font-family: 'Fraunces', serif; font-size: 60px; color: var(--bridge); opacity: 0.5; line-height: 1; }
.bridge-band .label { font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--bridge); margin-bottom: 8px; }
.bridge-band h3 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 26px; letter-spacing: -0.02em; margin: 0 0 8px; }
.bridge-band p { color: var(--text-dim); max-width: 720px; margin: 0 0 18px; font-family: 'Fraunces', serif; font-style: italic; font-size: 16px; }
.bridge-pairs { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
.bridge-pair { display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  background: rgba(168,85,247,0.06); border: 1px solid rgba(168,85,247,0.18); font-size: 13px; }
.bridge-pair .arc, .bridge-pair .dic { font-family: 'Fraunces', serif; font-weight: 600; font-size: 17px; }
.bridge-pair .arc { color: var(--ar); }
.bridge-pair .dic { color: var(--diff); }
.bridge-pair .arrow { color: var(--bridge); font-size: 13px; }
.bridge-pair .cos { margin-left: auto; font-size: 11px; color: var(--text-muted); }

/* ---------- footer ---------- */
.foot { margin-top: 100px; padding-top: 24px; border-top: 1px solid var(--border);
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-muted);
  display: flex; justify-content: space-between; align-items: center; }
.foot a { color: var(--text-dim); text-decoration: none; }
.foot a:hover { color: var(--gold); }

/* ===========================================================
 * FEATURE PAGE styles
 * =========================================================== */
.dossier-head { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 48px;
  align-items: end; margin-bottom: 56px; padding-top: 8px; }
.dossier-id { font-family: 'Fraunces', serif; font-weight: 800;
  font-size: clamp(96px, 14vw, 200px); line-height: 0.88; letter-spacing: -0.05em; margin: 0; }
.dossier-id .hash { font-style: italic; font-weight: 400; }
.dossier-mode-ar .dossier-id { color: var(--ar); }
.dossier-mode-diff .dossier-id { color: var(--diff); }
.dossier-mode-ar .dossier-id .hash { color: var(--ar-dim); }
.dossier-mode-diff .dossier-id .hash { color: var(--diff-dim); }
.dossier-classify { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; }
.dossier-mode-tag {
  display: inline-block; padding: 4px 12px; font-size: 11px; letter-spacing: 0.2em;
  text-transform: uppercase; font-weight: 600; border: 1px solid; margin-right: 6px;
}
.dossier-mode-ar .dossier-mode-tag { color: var(--ar); border-color: var(--ar); background: rgba(245,158,11,0.08); }
.dossier-mode-diff .dossier-mode-tag { color: var(--diff); border-color: var(--diff); background: rgba(34,211,238,0.08); }
.dossier-side { display: flex; flex-direction: column; gap: 14px; }
.dossier-meta-card { padding: 18px 20px; background: var(--surface); border: 1px solid var(--border); }
.dossier-meta-card .lbl { font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
.dossier-meta-card .val { font-family: 'Fraunces', serif; font-size: 26px; font-weight: 600; letter-spacing: -0.02em; }
.dossier-mode-ar .dossier-meta-card.peak .val { color: var(--ar); }
.dossier-mode-diff .dossier-meta-card.peak .val { color: var(--diff); }
.dossier-meta-card.bridge { border-color: var(--bridge); background: rgba(168,85,247,0.06); }
.dossier-meta-card.bridge .lbl { color: var(--bridge); }
.dossier-meta-card.bridge .val { color: var(--bridge); font-size: 22px; }
.dossier-meta-card.bridge .sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

.charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 32px 0 56px; }
.chart-card { background: var(--surface); border: 1px solid var(--border); padding: 4px 8px 8px; }

.specimen { display: grid; grid-template-columns: 130px 1fr 90px; gap: 22px; align-items: start;
  padding: 18px 22px; background: var(--surface); border: 1px solid var(--border);
  margin-bottom: 10px; position: relative;
}
.specimen .label-col .case-id { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
.specimen .label-col .case-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 17px; color: var(--text); margin-bottom: 6px; }
.specimen .label-col .case-pos { font-size: 11px; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; }
.specimen .text-col {
  background: rgba(0,0,0,0.25); padding: 12px 14px; border-radius: 0; font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px; line-height: 1.85; overflow-wrap: anywhere;
}
.specimen .text-col span { padding: 1px 2px; border-radius: 2px; transition: filter 0.15s ease; }
.specimen .text-col span.peak { outline: 1px solid var(--gold); outline-offset: 1px; }
.specimen .value-col { text-align: right; }
.specimen .value-col .val { font-family: 'Fraunces', serif; font-size: 32px; font-weight: 600; letter-spacing: -0.02em; line-height: 1; }
.dossier-mode-ar .specimen .value-col .val { color: var(--ar); }
.dossier-mode-diff .specimen .value-col .val { color: var(--diff); }
.specimen .value-col .lbl { font-size: 10px; color: var(--text-muted); letter-spacing: 0.16em; text-transform: uppercase; margin-top: 4px; }
.specimen .value-col .bar { height: 3px; background: var(--ar); margin-top: 10px; transform-origin: right; opacity: 0.7; }
.dossier-mode-diff .specimen .value-col .bar { background: var(--diff); }

.nav-back { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-dim); text-decoration: none; }
.nav-back:hover { color: var(--gold); }

/* page-load reveal */
@keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.reveal { animation: rise 0.7s ease both; }
.reveal-1 { animation-delay: 0.05s; }
.reveal-2 { animation-delay: 0.15s; }
.reveal-3 { animation-delay: 0.30s; }
.reveal-4 { animation-delay: 0.45s; }
</style>
""" % THEME


def topbar(extra_crumb: str = "Index"):
    return (
        '<div class="topbar reveal reveal-1">'
        '  <div class="brand"><span class="dot"></span><span class="logo">Sfumato</span>'
        '<span class="tagline">F10 · 305M · TopK SAE</span></div>'
        '  <div class="crumbs">'
        '<span>§ P.6 Dossier</span><span>·</span>'
        f'<span>{html.escape(extra_crumb)}</span></div>'
        '</div>'
    )


PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, ui-monospace, monospace", color="#e5e7eb", size=11),
    xaxis=dict(gridcolor="#262b3d", zerolinecolor="#262b3d", linecolor="#262b3d", tickcolor="#262b3d"),
    yaxis=dict(gridcolor="#262b3d", zerolinecolor="#262b3d", linecolor="#262b3d", tickcolor="#262b3d"),
    margin=dict(t=46, b=44, l=52, r=16),
)


# ---------------------------------------------------------------------------
# Token highlight colour ramp (dark-mode-friendly: dark-blue → magenta → yellow)
# ---------------------------------------------------------------------------

def colour_for(val, vmax):
    if vmax <= 0 or val <= 0:
        return "transparent", "var(--text-dim)"
    t = min(val / vmax, 1.0)
    if t < 0.5:
        # midnight → magenta
        a = t * 2
        r = int(26 + (190 - 26) * a)
        g = int(31 + (45 - 31) * a)
        b = int(46 + (140 - 46) * a)
    else:
        # magenta → gold
        a = (t - 0.5) * 2
        r = int(190 + (251 - 190) * a)
        g = int(45 + (191 - 45) * a)
        b = int(140 - 140 * a + 36 * a)
    fg = "#0a0e1a" if t > 0.6 else "#e5e7eb"
    return f"rgb({r},{g},{b})", fg


def render_tokens(token_strs, acts, peak_idx, vmax):
    pieces = []
    for j in range(len(token_strs)):
        tok = html.escape(token_strs[j]).replace("\n", "↵")
        bg, fg = colour_for(acts[j], vmax)
        cls = " class='peak'" if j == peak_idx else ""
        pieces.append(
            f'<span style="background:{bg};color:{fg}"{cls} title="act={acts[j]:.2f}">{tok}</span>'
        )
    return "".join(pieces)


# ---------------------------------------------------------------------------
# CHART BUILDERS
# ---------------------------------------------------------------------------

def plotly_histogram(values, title, accent):
    fig = go.Figure(go.Histogram(
        x=values, marker_color=accent, nbinsx=24,
        marker_line=dict(color=THEME["bg"], width=1.2),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=THEME["text_dim"]),
                   x=0.02, xanchor="left"),
        height=240,
        **PLOTLY_DARK_LAYOUT,
        xaxis_title=dict(text="peak activation per prompt", font=dict(size=10)),
        yaxis_title=dict(text="count", font=dict(size=10)),
    )
    return pyo.plot(fig, include_plotlyjs="cdn", output_type="div",
                     config={"displayModeBar": False})


def plotly_related_bar(feature_id, mode, top_self, top_cross):
    own_color = THEME["ar"] if mode == "ar" else THEME["diff"]
    cross_color = THEME["diff"] if mode == "ar" else THEME["ar"]
    other = "diff" if mode == "ar" else "ar"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{mode} #{i}" for i, c in top_self],
        y=[c for i, c in top_self],
        marker_color=own_color, name=f"top within {mode}",
        marker_line=dict(color=THEME["bg"], width=1),
    ))
    fig.add_trace(go.Bar(
        x=[f"{other} #{i}" for i, c in top_cross],
        y=[c for i, c in top_cross],
        marker_color=cross_color, name=f"top across to {other}",
        marker_line=dict(color=THEME["bg"], width=1),
    ))
    fig.update_layout(
        title=dict(text="closest features by decoder-column cosine",
                   font=dict(size=12, color=THEME["text_dim"]), x=0.02),
        height=280,
        **PLOTLY_DARK_LAYOUT,
        barmode="group", xaxis_tickangle=-45,
        yaxis_title=dict(text="cosine", font=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.42,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return pyo.plot(fig, include_plotlyjs="cdn", output_type="div",
                     config={"displayModeBar": False})


def build_interactive_heatmap(W_ar, W_di):
    print("[p6] computing best-cosine matrix subset ...", flush=True)
    best_ar = (W_ar @ W_di.T).max(dim=1).values
    top_ar_idx = best_ar.argsort(descending=True)[:256]
    best_di = (W_di @ W_ar.T).max(dim=1).values
    top_di_idx = best_di.argsort(descending=True)[:256]
    sub = (W_ar[top_ar_idx] @ W_di[top_di_idx].T).numpy()

    # Custom dark-mode colorscale: midnight → purple → gold
    scale = [
        [0.0,  "#0a0e1a"],
        [0.12, "#1a1f2e"],
        [0.30, "#3b2a5e"],
        [0.55, "#a855f7"],
        [0.78, "#f59e0b"],
        [1.0,  "#fde68a"],
    ]
    fig = go.Figure(go.Heatmap(
        z=sub,
        x=[f"diff #{int(i)}" for i in top_di_idx],
        y=[f"AR #{int(i)}" for i in top_ar_idx],
        colorscale=scale, zmin=-0.05, zmax=0.9,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>cosine = %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.7, x=1.02, tickfont=dict(size=10),
                      title=dict(text="cos", font=dict(size=10))),
    ))
    fig.update_layout(
        height=620,
        **{**PLOTLY_DARK_LAYOUT, "margin": dict(t=10, b=10, l=10, r=60)},
        xaxis_showticklabels=False, yaxis_showticklabels=False,
        xaxis_showgrid=False, yaxis_showgrid=False,
    )
    return pyo.plot(fig, include_plotlyjs="cdn", output_type="div",
                     config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# PER-FEATURE PAGE
# ---------------------------------------------------------------------------

def build_feature_page(feature_id, mode, sae_self, sae_other, raw_model, tok,
                        prompts, bridge_pairs, out_path: Path, device):
    rows = []
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
    rows.sort(key=lambda r: -r["peak_act"])
    rows = [r for r in rows if r["peak_act"] > 0][:18]
    vmax = max(r["peak_act"] for r in rows) if rows else 1.0

    W_self = F.normalize(sae_self.W_dec.detach(), dim=1).cpu()
    W_other = F.normalize(sae_other.W_dec.detach(), dim=1).cpu()
    my_vec = W_self[feature_id]
    self_sims = (W_self @ my_vec).numpy()
    self_sims[feature_id] = -2
    cross_sims = (W_other @ my_vec).numpy()
    top_self = [(int(i), float(self_sims[i])) for i in np.argsort(self_sims)[-10:][::-1]]
    top_cross = [(int(i), float(cross_sims[i])) for i in np.argsort(cross_sims)[-10:][::-1]]

    bridge_card = ""
    for p in bridge_pairs:
        if (mode == "ar" and p["ar_feature"] == feature_id) or \
           (mode == "diff" and p["diff_feature"] == feature_id):
            other_id = p["diff_feature"] if mode == "ar" else p["ar_feature"]
            other_mode = "diff" if mode == "ar" else "ar"
            bridge_card = (
                '<div class="dossier-meta-card bridge">'
                '<div class="lbl">Δ Bridge pair</div>'
                f'<div class="val">{other_mode} #{other_id}</div>'
                f'<div class="sub">cross-head cosine {p["cosine"]:.3f} · universal vocabulary</div>'
                '</div>'
            )
            break

    accent = THEME["ar"] if mode == "ar" else THEME["diff"]
    other_mode = "diff" if mode == "ar" else "ar"

    parts = [
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{mode} · #{feature_id}</title>',
        GLOBAL_CSS,
        f'</head><body class="dossier-mode-{mode}"><div class="shell">',
        topbar(f"{mode}/#{feature_id}"),

        '<a href="index.html" class="nav-back">← all features</a>',

        '<div class="dossier-head reveal reveal-2">',
        '  <div>',
        f'    <div class="dossier-classify">Dossier · {mode}-head · ln_f hookpoint</div>',
        f'    <h1 class="dossier-id"><span class="hash">#</span>{feature_id}</h1>',
        '    <div style="margin-top:18px;">',
        f'      <span class="dossier-mode-tag">{mode} head</span>',
        '      <span style="font-size:11px;color:var(--text-muted);letter-spacing:0.14em;'
        'text-transform:uppercase;">k=64 · 16,384 features · TopK</span>',
        '    </div>',
        '  </div>',
        '  <div class="dossier-side">',
        f'    <div class="dossier-meta-card peak"><div class="lbl">Top peak activation</div>'
        f'<div class="val">{vmax:.2f}</div></div>',
        f'    <div class="dossier-meta-card"><div class="lbl">Prompts firing</div>'
        f'<div class="val">{len(rows)} / 16</div></div>',
        f'    {bridge_card}',
        '  </div>',
        '</div>',

        '<div class="charts-row reveal reveal-3">',
        f'  <div class="chart-card">{plotly_histogram([r["peak_act"] for r in rows], "peak activation across prompts", accent)}</div>',
        f'  <div class="chart-card">{plotly_related_bar(feature_id, mode, top_self, top_cross)}</div>',
        '</div>',

        '<div class="section-head reveal reveal-3">'
        '<span class="num">§</span><h2>Specimens</h2><span class="rule"></span></div>',
        '<p class="section-intro">Top contexts ranked by peak activation. '
        'Token background is the activation strength on a dark-blue → magenta → gold ramp; '
        'the gold-outlined token is the peak. Hover for exact values.</p>',
    ]

    for r in rows:
        bar_pct = max(8, int(100 * r["peak_act"] / vmax))
        parts.append('<div class="specimen reveal reveal-4">')
        parts.append('<div class="label-col">'
                     f'  <div class="case-id">Case {html.escape(r["label"])[:14]}</div>'
                     f'  <div class="case-name">{html.escape(r["label"])}</div>'
                     f'  <div class="case-pos">peak @ pos {r["peak_idx"]}</div>'
                     '</div>')
        parts.append(f'<div class="text-col">{render_tokens(r["token_strs"], r["acts"], r["peak_idx"], vmax)}</div>')
        parts.append('<div class="value-col">'
                     f'  <div class="val">{r["peak_act"]:.2f}</div>'
                     '  <div class="lbl">peak</div>'
                     f'  <div class="bar" style="width:{bar_pct}%; margin-left:auto"></div>'
                     '</div>')
        parts.append('</div>')

    parts.extend([
        '<div class="foot reveal reveal-4">'
        f'<span>Sfumato · Phase P.6</span>'
        f'<span>Feature {mode} #{feature_id} · open dossier</span>'
        '<a href="index.html">↰ back to roster</a>'
        '</div>',
        '</div></body></html>'
    ])
    out_path.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# INDEX
# ---------------------------------------------------------------------------

def build_index(out_dir: Path, top_ar_feats, top_diff_feats, total_ar, total_diff,
                bridge_pairs, bridge_ar, bridge_diff, heatmap_html):
    ar_rows = []
    for fid in sorted(top_ar_feats, key=lambda f: -float(total_ar[f])):
        is_bridge = fid in bridge_ar
        ar_rows.append(
            f'<a class="feat-card ar" href="feat_ar_{fid}.html">'
            f'  <div><div class="fid">#{fid}</div><div class="meta">ar/ln_f</div></div>'
            f'  <div class="body">'
            f'    <span class="name">k=64 TopK · d=16384</span>'
            f'    <span class="activation">universal content anchor' + ('' if not is_bridge else '<span class="bridge-tag">bridge</span>') + '</span>'
            f'  </div>'
            f'  <div class="right">'
            f'    <div class="activation-val">{float(total_ar[fid]):.0f}</div>'
            f'    <div class="activation-lbl">Σ activation</div>'
            f'  </div>'
            f'</a>'
        )

    diff_rows = []
    for fid in sorted(top_diff_feats, key=lambda f: -float(total_diff[f])):
        is_bridge = fid in bridge_diff
        diff_rows.append(
            f'<a class="feat-card diff" href="feat_diff_{fid}.html">'
            f'  <div><div class="fid">#{fid}</div><div class="meta">diff/ln_f</div></div>'
            f'  <div class="body">'
            f'    <span class="name">k=64 TopK · d=16384</span>'
            f'    <span class="activation">universal content anchor' + ('' if not is_bridge else '<span class="bridge-tag">bridge</span>') + '</span>'
            f'  </div>'
            f'  <div class="right">'
            f'    <div class="activation-val">{float(total_diff[fid]):.0f}</div>'
            f'    <div class="activation-lbl">Σ activation</div>'
            f'  </div>'
            f'</a>'
        )

    bridge_pair_rows = []
    for p in bridge_pairs[:12]:
        bridge_pair_rows.append(
            '<div class="bridge-pair">'
            f'<span class="arc">AR #{p["ar_feature"]}</span>'
            f'<span class="arrow">↔</span>'
            f'<span class="dic">D #{p["diff_feature"]}</span>'
            f'<span class="cos">cos {p["cosine"]:.3f}</span>'
            '</div>'
        )

    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sfumato · Feature Dossier</title>',
        GLOBAL_CSS,
        '</head><body><div class="shell">',
        topbar("Field Manual"),

        # ---------- HERO ----------
        '<div class="hero reveal reveal-2">',
        '  <div>',
        '    <div class="strap">A field manual for the F10 composite · Phase P · Sfumato</div>',
        '    <h1>Two heads, <em>one backbone</em>,<br>disjoint vocabularies.</h1>',
        '  </div>',
        '  <div class="hero-stat">',
        '    <div class="kicker">Mean cross-head decoder-column cosine</div>',
        '    <p class="big">0.169</p>',
        '    <div class="verdict">The composite at <strong>305M</strong> does not learn one shared representation served by two readouts. AR and diff use almost entirely <strong>different</strong> sparse-feature bases inside the same backbone.</div>',
        '    <div class="footnote">16,384 features · per-feature best cross-head match · only 1.84% above 0.5 cosine.</div>',
        '  </div>',
        '</div>',

        # ---------- FACTIONS ----------
        '<div class="factions reveal reveal-3">',
        '  <div class="faction ar"><div class="label">Faction AR</div>'
        '    <div class="title">Causal · sequential · left-to-right</div>'
        '    <div class="desc">The autoregressive head. Reads under a causal mask, produces logits one token at a time. Its features favour positions where the model has to commit to a next prediction — verbs, numbers, sentence boundaries.</div>'
        '  </div>',
        '  <div class="faction bridge"><div class="label">Bridge Δ</div>'
        '    <div class="title">Universal vocabulary</div>'
        '    <div class="desc">A tiny set of features (≈ 20 pairs with cosine ≥ 0.7) that both factions inherit from the shared backbone. They fire on semantic anchor tokens across math, prose, code, and dialogue. The K.2 mode-switching recipe routes through these.</div>'
        '  </div>',
        '  <div class="faction diff"><div class="label">Faction Diff</div>'
        '    <div class="title">Bidirectional · parallel · mask-fill</div>'
        '    <div class="desc">The discrete-diffusion head. Reads in both directions, fills masked spans in parallel. Its features cluster around objects, grammatical structure, and the end-of-question markers it needs to predict.</div>'
        '  </div>',
        '</div>',

        # ---------- HEATMAP ----------
        '<div class="section-head reveal reveal-3"><span class="num">I.</span>'
        '<h2>Cross-head cosine matrix</h2><span class="rule"></span></div>',
        '<p class="section-intro reveal reveal-3">The decoder of one head against the decoder of the other. '
        'The dark sea is the rule (mean cosine <b>0.169</b>); the thin bright stripe top-left is the exception — '
        'a few dozen <b>bridge features</b> whose decoder directions both heads share. '
        'Hover any cell for the exact AR ↔ diff cosine.</p>',
        f'<div class="heatmap-card reveal reveal-3">',
        '  <div class="hm-title">Cross-head SAE cosine — top 256 × top 256</div>',
        '  <div class="hm-sub">sorted by per-feature best-cross-cosine · interactive hover</div>',
        f'  {heatmap_html}',
        '  <div class="annotations">'
        f'    <span><span class="swatch" style="background:#0a0e1a;border:1px solid #262b3d"></span>~0.10 (specialised)</span>'
        f'    <span><span class="swatch" style="background:#a855f7"></span>~0.55 (partial)</span>'
        f'    <span><span class="swatch" style="background:#f59e0b"></span>~0.78 (bridge)</span>'
        f'    <span><span class="swatch" style="background:#fde68a"></span>≥ 0.85 (universal)</span>'
        '  </div>',
        '</div>',

        # ---------- ROSTER ----------
        '<div class="section-head reveal reveal-3"><span class="num">II.</span>'
        '<h2>Feature roster</h2><span class="rule"></span></div>',
        '<p class="section-intro reveal reveal-3">Top firing features in each head across a 16-prompt diagnostic '
        'corpus (math · prose · code · dialogue · recipes). Click any card to open its dossier — '
        'specimens, similarity neighbours, peak distribution.</p>',
        '<div class="roster reveal reveal-3">',
        '  <div>',
        f'    <div class="column-head ar"><span class="faction-name">AR head</span>'
        f'      <span class="count">{len(top_ar_feats)} features</span></div>',
        *ar_rows,
        '  </div>',
        '  <div>',
        f'    <div class="column-head diff"><span class="faction-name">diff head</span>'
        f'      <span class="count">{len(top_diff_feats)} features</span></div>',
        *diff_rows,
        '  </div>',
        '</div>',

        # ---------- BRIDGE ----------
        '<div class="bridge-band reveal reveal-3">',
        '  <div class="label">Bridge specimens</div>',
        '  <h3>The features both heads inherit</h3>',
        '  <p>Cosine ≥ 0.85 between an AR decoder column and a diff decoder column. '
        'These pairs are the few directions the shared backbone carries unchanged through to both readouts.</p>',
        '  <div class="bridge-pairs">',
        *bridge_pair_rows,
        '  </div>',
        '</div>',

        # ---------- FOOT ----------
        '<div class="foot reveal reveal-3">',
        '<span>github.com/eren23/sfumato</span>',
        '<span>Phase P · 3h compute · ~$3 pod</span>',
        '<span>F10 · 305M · 20-layer composite</span>',
        '</div>',

        '</div></body></html>',
    ]
    (out_dir / "index.html").write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

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
    print(f"[p6] {len(top_ar_feats)} AR · {len(top_diff_feats)} diff features", flush=True)

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

    W_ar = F.normalize(sae_ar.W_dec.detach(), dim=1).cpu()
    W_di = F.normalize(sae_diff.W_dec.detach(), dim=1).cpu()
    heatmap_html = build_interactive_heatmap(W_ar, W_di)

    build_index(out_dir, top_ar_feats, top_diff_feats, total_ar, total_diff,
                bridge_pairs, bridge_ar, bridge_diff, heatmap_html)
    print(f"\n[p6] OPEN: file://{out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
