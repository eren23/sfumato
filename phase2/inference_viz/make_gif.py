"""Render a STATUS-schema JSONL trace as an animated GIF + upload to W&B.

Frame N shows the cumulative state after sub-block N has been committed:
  - 4×32 token grid (sub-block × position)
  - Cell color encodes per-position Shannon entropy (cool blue = confident,
    warm amber = uncertain). Black cells = not yet generated.
  - Cell text = decoded token string (truncated)
  - Title shows: problem index, sub-block N, mechanism, manual_intervention

Usage:
    python phase2/inference_viz/make_gif.py phase2/inference_viz/traces/trace_real_p10_all_llada.jsonl
    python phase2/inference_viz/make_gif.py phase2/inference_viz/traces/*.jsonl --upload-wandb
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _color_for_entropy(e: float, max_e: float = 4.0) -> tuple[float, float, float]:
    """Cool blue (low entropy = confident) → warm amber (high entropy = uncertain)."""
    if e is None or not np.isfinite(e):
        return (0.15, 0.15, 0.15)  # dark grey for missing
    t = max(0.0, min(1.0, e / max_e))
    # interpolate blue (#1d4ed8) → amber (#b45309)
    blue = np.array([0.114, 0.306, 0.847])
    amber = np.array([0.706, 0.325, 0.035])
    rgb = (1 - t) * blue + t * amber
    return tuple(rgb.tolist())


def _short_token(s: str, maxlen: int = 6) -> str:
    if not s: return ""
    s = s.strip().replace("\n", "↵").replace("\t", "→")
    if len(s) > maxlen: return s[:maxlen-1] + "…"
    return s


def render_gif(jsonl_path: pathlib.Path, out_path: pathlib.Path | None = None,
               frame_ms: int = 800, dpi: int = 120) -> pathlib.Path:
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not records:
        raise ValueError(f"empty trace: {jsonl_path}")

    n_blocks = max(r["sub_block"] for r in records) + 1
    block_lens = [len(r.get("token_strings", [])) for r in records]
    block_len = max(block_lens) if block_lens else 32
    pi = records[0].get("problem_idx", "?")

    # Build the cumulative state per frame
    frames = []
    grid_tokens = [["" for _ in range(block_len)] for _ in range(n_blocks)]
    grid_ent = [[None for _ in range(block_len)] for _ in range(n_blocks)]
    interventions = ["" for _ in range(n_blocks)]
    mechanisms = ["" for _ in range(n_blocks)]
    for r in records:
        b = r["sub_block"]
        for i, (tok, ent) in enumerate(zip(r.get("token_strings", []), r.get("entropy", []))):
            if i < block_len:
                grid_tokens[b][i] = tok
                grid_ent[b][i] = float(ent) if ent is not None else None
        mi = r.get("manual_intervention") or {}
        interventions[b] = mi.get("directive", "continue_llada") if isinstance(mi, dict) else str(mi)
        mechanisms[b] = r.get("mechanism", "?")
        # snapshot
        frames.append({
            "tokens": [row[:] for row in grid_tokens],
            "ent": [row[:] for row in grid_ent],
            "interventions": interventions[:],
            "mechanisms": mechanisms[:],
            "active_block": b,
        })

    # Plot setup
    fig, ax = plt.subplots(figsize=(16, max(3, n_blocks * 1.0)), dpi=dpi)
    ax.set_xlim(0, block_len)
    ax.set_ylim(0, n_blocks)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([y + 0.5 for y in range(n_blocks)])
    ax.set_yticklabels([f"sub-block {i}" for i in range(n_blocks)], fontsize=9)
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa")
    title = ax.set_title("", color="#eee", fontsize=11, pad=10)

    # Pre-create rectangles + texts (mutable each frame)
    rects = [[None] * block_len for _ in range(n_blocks)]
    texts = [[None] * block_len for _ in range(n_blocks)]
    for r in range(n_blocks):
        for c in range(block_len):
            rect = Rectangle((c, r), 1, 1, facecolor="#0a0a0a", edgecolor="#222", linewidth=0.3)
            ax.add_patch(rect)
            rects[r][c] = rect
            texts[r][c] = ax.text(c + 0.5, r + 0.5, "", ha="center", va="center",
                                    fontsize=7, color="#fafafa", family="monospace")

    def update(frame_idx):
        f = frames[frame_idx]
        for r in range(n_blocks):
            for c in range(block_len):
                tok = f["tokens"][r][c]
                ent = f["ent"][r][c]
                if tok:
                    color = _color_for_entropy(ent)
                    rects[r][c].set_facecolor(color)
                    rects[r][c].set_edgecolor("#fff" if r == f["active_block"] else "#444")
                    rects[r][c].set_linewidth(1.0 if r == f["active_block"] else 0.3)
                    # Text color contrast
                    luminance = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                    texts[r][c].set_color("#0a0a0a" if luminance > 0.55 else "#fafafa")
                    texts[r][c].set_text(_short_token(tok))
                else:
                    rects[r][c].set_facecolor("#0a0a0a")
                    rects[r][c].set_edgecolor("#222")
                    rects[r][c].set_linewidth(0.3)
                    texts[r][c].set_text("")
        active = f["active_block"]
        intv = f["interventions"][active] if active < len(f["interventions"]) else ""
        mech = f["mechanisms"][active] if active < len(f["mechanisms"]) else ""
        title.set_text(
            f"problem_idx={pi}  ·  sub-block {active}/{n_blocks-1}  ·  mechanism={mech}  ·  intervention={intv}"
        )
        return [title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=frame_ms, blit=False)

    if out_path is None:
        out_path = jsonl_path.with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer="pillow", fps=max(1, 1000 // frame_ms))
    plt.close(fig)
    return out_path


def upload_wandb(jsonl_path: pathlib.Path, gif_path: pathlib.Path,
                 project: str = "sfumato-e4", run_name_prefix: str = "viz-trace"):
    """Upload trace JSON + GIF to W&B. Each trace as its own short run."""
    import wandb
    name = f"{run_name_prefix}-{jsonl_path.stem}"
    run = wandb.init(project=project, name=name, job_type="trace-viz", reinit=True,
                     config={"jsonl_file": jsonl_path.name})
    # Log the GIF as an Image
    run.log({"inference_animation": wandb.Image(str(gif_path), caption=jsonl_path.stem)})
    # Upload JSON as artifact
    art = wandb.Artifact(name=f"trace-{jsonl_path.stem}", type="trace-jsonl")
    art.add_file(str(jsonl_path))
    art.add_file(str(gif_path), name=jsonl_path.stem + ".gif")
    run.log_artifact(art)
    run.finish()
    print(f"  wandb: uploaded {name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSONL trace files (globs ok)")
    ap.add_argument("--frame-ms", type=int, default=800)
    ap.add_argument("--upload-wandb", action="store_true")
    ap.add_argument("--project", default="sfumato-e4")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)))
    files = [pathlib.Path(f) for f in files if pathlib.Path(f).exists()]
    if not files:
        print("no files matched"); return 1
    print(f"rendering {len(files)} GIF(s)", flush=True)

    for jp in files:
        try:
            gif = render_gif(jp, frame_ms=args.frame_ms)
            print(f"  {jp.name} → {gif.name} ({gif.stat().st_size//1024} KB)", flush=True)
            if args.upload_wandb:
                upload_wandb(jp, gif, project=args.project)
        except Exception as e:
            print(f"  FAIL {jp.name}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
