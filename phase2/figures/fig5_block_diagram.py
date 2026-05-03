"""Figure 5 (NEW) — LLaDA semi-AR sampler block diagram.

The LLaDA-8B-Instruct sampler at our cmaj/c2c settings emits a 128-token
response in 4 sub-blocks of 32 tokens each. This figure shows that block
structure plus the two commit-LoRA application regions:

  - commit-LoRA v2 fires only on the LAST sub-block (block 4) — answer
    span only. (n_blocks=1, late-block-only application.)
  - commit-LoRA v3 fires on sub-blocks 2-4 — full-response loss, the
    structural fix that drove c2c from 70.5% to 79.0%.

Authoring is dual:
  - phase2/figures/fig5_block_diagram.excalidraw   (round-trippable JSON)
  - phase2/figures/fig5_block_diagram.{svg,png}    (rendered fallback)

The Excalidraw JSON is the source-of-truth for editing in
excalidraw.com; the SVG/PNG are rendered from a matplotlib mirror so
build_all.py can regenerate without an external Excalidraw runtime.
Both versions encode the same layout — see ``LAYOUT`` below.

Outputs:
  phase2/figures/fig5_block_diagram.excalidraw
  phase2/figures/fig5_block_diagram.svg
  phase2/figures/fig5_block_diagram.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from palette import PALETTE  # noqa: E402

plt.style.use(str(HERE / "sfumato.mplstyle"))

OUT_DIR = HERE
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Single source-of-truth layout shared by both renderers.
#
# Coordinate system: matplotlib axes units, [0, 12] x [0, 6].
# Each sub-block is 2.4 wide. Tokens drawn as 32 narrow rectangles per block.
# ---------------------------------------------------------------------------
LAYOUT = {
    "canvas": {"x0": 0.0, "y0": 0.0, "x1": 13.4, "y1": 6.6},
    "blocks": [
        {"idx": 1, "x": 0.6,  "label": "Block 1\ntokens 1-32"},
        {"idx": 2, "x": 3.4,  "label": "Block 2\ntokens 33-64"},
        {"idx": 3, "x": 6.2,  "label": "Block 3\ntokens 65-96"},
        {"idx": 4, "x": 9.0,  "label": "Block 4\ntokens 97-128"},
    ],
    "block_width": 2.4,
    "block_y": 2.7,
    "block_height": 1.6,
    "tokens_per_block": 32,
    "v2_region": {"blocks": [4],          "color": PALETTE.warn,
                  "label": "commit-LoRA v2  —  last block only  (answer span, n_blocks=1)"},
    "v3_region": {"blocks": [2, 3, 4],    "color": PALETTE.v3,
                  "label": "commit-LoRA v3  —  blocks 2-4  (full response, n_blocks=3)"},
}


# ---------------------------------------------------------------------------
# Excalidraw JSON writer
#
# Schema reference (Excalidraw v2 file format,
# https://github.com/excalidraw/excalidraw/blob/master/dev-docs/spec/spec.md):
#   {
#     "type": "excalidraw",
#     "version": 2,
#     "source": "...",
#     "elements": [...],   # rectangles, text, arrows
#     "appState": {...},
#     "files": {}
#   }
#
# Each element: stable id (deterministic hash of label+coords), type,
# x, y, width, height, angle, strokeColor, backgroundColor, fillStyle,
# strokeWidth, strokeStyle, roundness, roughness, opacity, seed,
# versionNonce, isDeleted, boundElements, updated, link, locked.
# ---------------------------------------------------------------------------

EXCAL_SOURCE = "https://github.com/eren23/sfumato — phase2/figures/fig5_block_diagram.py"


def _stable_id(*parts: Any) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:21]   # Excalidraw uses 21-char ids


def _stable_seed(*parts: Any) -> int:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


# Excalidraw uses Y-down screen coords; convert from our Y-up matplotlib coords.
def _to_excal_y(y: float, height: float = 0.0) -> float:
    return (LAYOUT["canvas"]["y1"] - y - height) * 60.0   # 60 px per axes-unit


def _to_excal_x(x: float) -> float:
    return x * 60.0


def _excal_rect(label: str, x: float, y: float, w: float, h: float,
                stroke: str, fill: str, fill_style: str = "solid",
                stroke_style: str = "solid",
                roundness: dict | None = None, opacity: int = 100) -> dict:
    return {
        "id": _stable_id("rect", label, x, y, w, h, stroke, fill),
        "type": "rectangle",
        "x": _to_excal_x(x), "y": _to_excal_y(y, h),
        "width": w * 60.0, "height": h * 60.0,
        "angle": 0,
        "strokeColor": stroke, "backgroundColor": fill,
        "fillStyle": fill_style, "strokeWidth": 1,
        "strokeStyle": stroke_style, "roughness": 0,
        "opacity": opacity,
        "groupIds": [], "frameId": None, "index": None,
        "roundness": roundness or {"type": 3},
        "seed": _stable_seed("rect", label, x, y),
        "versionNonce": _stable_seed("nonce", label, x, y),
        "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
    }


def _excal_text(text: str, x: float, y: float, font_size: int = 16,
                color: str = "#111827", text_align: str = "center",
                width: float = 2.0, height: float = 0.5) -> dict:
    return {
        "id": _stable_id("text", text, x, y, font_size),
        "type": "text",
        "x": _to_excal_x(x), "y": _to_excal_y(y, height),
        "width": width * 60.0, "height": height * 60.0,
        "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1,
        "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "index": None,
        "roundness": None,
        "seed": _stable_seed("text", text, x, y),
        "versionNonce": _stable_seed("nonce-text", text, x, y),
        "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
        "fontSize": font_size, "fontFamily": 5,  # 5 = Helvetica/sans
        "text": text, "textAlign": text_align, "verticalAlign": "middle",
        "containerId": None, "originalText": text,
        "lineHeight": 1.25, "autoResize": True,
    }


def _excal_arrow(x0: float, y0: float, x1: float, y1: float,
                 color: str, label: str = "") -> dict:
    return {
        "id": _stable_id("arrow", x0, y0, x1, y1, color, label),
        "type": "arrow",
        "x": _to_excal_x(x0), "y": _to_excal_y(y0),
        "width": (x1 - x0) * 60.0, "height": -(y1 - y0) * 60.0,
        "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1.5,
        "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "index": None,
        "roundness": {"type": 2},
        "seed": _stable_seed("arrow", x0, y0, x1, y1),
        "versionNonce": _stable_seed("arrow-n", x0, y0, x1, y1),
        "isDeleted": False, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
        "points": [[0, 0], [(x1 - x0) * 60.0, -(y1 - y0) * 60.0]],
        "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
    }


def write_excalidraw() -> Path:
    elements: list[dict] = []

    # Title
    elements.append(_excal_text(
        "LLaDA semi-AR sampler  —  4 sub-blocks of 32 tokens",
        x=0.4, y=6.2, font_size=22, color=PALETTE.ink, text_align="left",
        width=11.0, height=0.55,
    ))
    elements.append(_excal_text(
        "Commit-LoRA application regions (v2 vs v3)",
        x=0.4, y=5.65, font_size=14, color=PALETTE.sub, text_align="left",
        width=11.0, height=0.45,
    ))

    # commit-LoRA region bands (drawn UNDER the blocks)
    v3_xs = [LAYOUT["blocks"][i - 1]["x"] for i in LAYOUT["v3_region"]["blocks"]]
    v3_x0 = min(v3_xs) - 0.12
    v3_x1 = max(v3_xs) + LAYOUT["block_width"] + 0.12
    elements.append(_excal_rect(
        "v3-band", v3_x0, LAYOUT["block_y"] - 0.45,
        v3_x1 - v3_x0, LAYOUT["block_height"] + 0.95,
        stroke=PALETTE.v3, fill=PALETTE.v3, fill_style="hachure", opacity=18,
        roundness={"type": 3},
    ))
    elements.append(_excal_text(
        LAYOUT["v3_region"]["label"],
        x=v3_x0, y=LAYOUT["block_y"] + LAYOUT["block_height"] + 0.65,
        font_size=13, color=PALETTE.v3,
        text_align="left", width=v3_x1 - v3_x0, height=0.4,
    ))

    v2_xs = [LAYOUT["blocks"][i - 1]["x"] for i in LAYOUT["v2_region"]["blocks"]]
    v2_x0 = min(v2_xs) - 0.06
    v2_x1 = max(v2_xs) + LAYOUT["block_width"] + 0.06
    elements.append(_excal_rect(
        "v2-band", v2_x0, LAYOUT["block_y"] - 0.25,
        v2_x1 - v2_x0, LAYOUT["block_height"] + 0.55,
        stroke=PALETTE.warn, fill=PALETTE.warn, fill_style="cross-hatch", opacity=22,
        roundness={"type": 3},
    ))
    elements.append(_excal_text(
        LAYOUT["v2_region"]["label"],
        x=v2_x0 - 5.7, y=LAYOUT["block_y"] - 0.85,
        font_size=13, color=PALETTE.warn,
        text_align="left", width=v2_x1 - v2_x0 + 5.7, height=0.4,
    ))

    # Per-block boxes + token grid
    for blk in LAYOUT["blocks"]:
        # Block frame
        elements.append(_excal_rect(
            f"blk{blk['idx']}",
            blk["x"], LAYOUT["block_y"],
            LAYOUT["block_width"], LAYOUT["block_height"],
            stroke=PALETTE.ink, fill="transparent", fill_style="solid",
            roundness={"type": 3},
        ))
        # Block label (above)
        elements.append(_excal_text(
            blk["label"],
            x=blk["x"], y=LAYOUT["block_y"] + LAYOUT["block_height"] + 0.18,
            font_size=13, color=PALETTE.ink,
            text_align="center", width=LAYOUT["block_width"], height=0.5,
        ))
        # Token grid: 32 small rectangles in 4 rows of 8
        token_w = (LAYOUT["block_width"] - 0.18) / 8.0
        token_h = (LAYOUT["block_height"] - 0.18) / 4.0
        for ti in range(LAYOUT["tokens_per_block"]):
            row = ti // 8
            col = ti % 8
            tx = blk["x"] + 0.09 + col * token_w
            ty = LAYOUT["block_y"] + LAYOUT["block_height"] - 0.09 - (row + 1) * token_h
            elements.append(_excal_rect(
                f"tok-{blk['idx']}-{ti}",
                tx, ty, token_w * 0.85, token_h * 0.7,
                stroke=PALETTE.sub, fill=PALETTE.rule, fill_style="solid",
                roundness={"type": 3}, opacity=70,
            ))

    # Diffusion-direction arrow
    elements.append(_excal_arrow(
        x0=0.4, y0=LAYOUT["block_y"] - 1.3,
        x1=12.0, y1=LAYOUT["block_y"] - 1.3,
        color=PALETTE.sub, label="time",
    ))
    elements.append(_excal_text(
        "diffusion timestep  (semi-AR: block N can read blocks 1..N-1)",
        x=0.4, y=LAYOUT["block_y"] - 1.65, font_size=12,
        color=PALETTE.sub, text_align="left", width=11.6, height=0.4,
    ))

    # Footer
    elements.append(_excal_text(
        "v2 fixes the answer; v3 fixes the trajectory.  See "
        "RESULTS_TRACK2.md \"Disentangling ablations\".",
        x=0.4, y=0.1, font_size=11, color=PALETTE.sub, text_align="left",
        width=12.6, height=0.4,
    ))

    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": EXCAL_SOURCE,
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "gridSize": None,
            "currentItemFontFamily": 5,
            "currentItemFontSize": 14,
        },
        "files": {},
    }
    out = OUT_DIR / "fig5_block_diagram.excalidraw"
    # Excalidraw expects a JSON file with stable formatting
    out.write_text(json.dumps(doc, indent=2, sort_keys=False))
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Matplotlib mirror — renders the same layout to SVG + PNG.
# ---------------------------------------------------------------------------
def render_static() -> None:
    cv = LAYOUT["canvas"]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    ax.set_xlim(cv["x0"], cv["x1"])
    ax.set_ylim(cv["y0"], cv["y1"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)

    # ---- Region bands (under blocks) -----------------------------------
    v3_xs = [LAYOUT["blocks"][i - 1]["x"] for i in LAYOUT["v3_region"]["blocks"]]
    v3_x0 = min(v3_xs) - 0.12
    v3_x1 = max(v3_xs) + LAYOUT["block_width"] + 0.12
    v3_band = FancyBboxPatch(
        (v3_x0, LAYOUT["block_y"] - 0.45),
        v3_x1 - v3_x0, LAYOUT["block_height"] + 0.95,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=PALETTE.v3, alpha=0.10,
        edgecolor=PALETTE.v3, linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(v3_band)
    ax.text(
        v3_x0 + 0.08, LAYOUT["block_y"] + LAYOUT["block_height"] + 0.78,
        LAYOUT["v3_region"]["label"],
        ha="left", va="top", fontsize=10, color=PALETTE.v3, fontweight="semibold",
    )

    v2_xs = [LAYOUT["blocks"][i - 1]["x"] for i in LAYOUT["v2_region"]["blocks"]]
    v2_x0 = min(v2_xs) - 0.06
    v2_x1 = max(v2_xs) + LAYOUT["block_width"] + 0.06
    v2_band = FancyBboxPatch(
        (v2_x0, LAYOUT["block_y"] - 0.25),
        v2_x1 - v2_x0, LAYOUT["block_height"] + 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=PALETTE.warn, alpha=0.18,
        edgecolor=PALETTE.warn, linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(v2_band)
    ax.text(
        v2_x0 - 0.10, LAYOUT["block_y"] - 0.55,
        LAYOUT["v2_region"]["label"],
        ha="right", va="top", fontsize=10, color=PALETTE.warn, fontweight="semibold",
    )

    # ---- Block boxes + token grids -------------------------------------
    token_w_full = (LAYOUT["block_width"] - 0.18) / 8.0
    token_h_full = (LAYOUT["block_height"] - 0.18) / 4.0
    token_w = token_w_full * 0.78
    token_h = token_h_full * 0.62
    for blk in LAYOUT["blocks"]:
        frame = FancyBboxPatch(
            (blk["x"], LAYOUT["block_y"]),
            LAYOUT["block_width"], LAYOUT["block_height"],
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor="white", edgecolor=PALETTE.ink, linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(frame)
        ax.text(
            blk["x"] + LAYOUT["block_width"] / 2,
            LAYOUT["block_y"] + LAYOUT["block_height"] + 0.18,
            blk["label"],
            ha="center", va="bottom", fontsize=10, color=PALETTE.ink,
        )
        for ti in range(LAYOUT["tokens_per_block"]):
            row = ti // 8
            col = ti % 8
            tx = blk["x"] + 0.09 + col * token_w_full + (token_w_full - token_w) / 2
            ty = (LAYOUT["block_y"] + LAYOUT["block_height"] - 0.09
                  - (row + 1) * token_h_full + (token_h_full - token_h) / 2)
            tok = Rectangle(
                (tx, ty), token_w, token_h,
                facecolor=PALETTE.rule, edgecolor=PALETTE.sub, linewidth=0.4,
                alpha=0.9, zorder=4,
            )
            ax.add_patch(tok)

    # ---- Diffusion-time arrow ------------------------------------------
    arr = FancyArrowPatch(
        (0.4, LAYOUT["block_y"] - 1.3),
        (12.0, LAYOUT["block_y"] - 1.3),
        arrowstyle="-|>", color=PALETTE.sub, lw=1.2,
        mutation_scale=12, zorder=2,
    )
    ax.add_patch(arr)
    ax.text(
        0.4, LAYOUT["block_y"] - 1.65,
        "diffusion timestep   (semi-AR: block N attends to blocks 1..N-1)",
        ha="left", va="top", fontsize=9.5, color=PALETTE.sub,
    )

    # Title
    ax.text(
        0.4, cv["y1"] - 0.25,
        "Figure 5   LLaDA semi-AR sampler  —  commit-LoRA v2 vs v3 application regions",
        ha="left", va="top", fontsize=12, fontweight="semibold", color=PALETTE.ink,
    )
    ax.text(
        0.4, cv["y1"] - 0.65,
        "v2 fires only on the last block (answer span); v3 fires on blocks 2-4 with "
        "full-response loss. RESULTS_TRACK2.md attributes +4 pp of the c2c lift to "
        "this block-coverage change.",
        ha="left", va="top", fontsize=9, color=PALETTE.sub,
    )

    fig.tight_layout(pad=0.4)
    for ext in ("svg", "png"):
        out = OUT_DIR / f"fig5_block_diagram.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> int:
    write_excalidraw()
    render_static()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
