"""Side-by-side comparison GIFs for visualizing the night-3 diagnosis.

Two trace flavors supported:

A. **Same-problem cmaj-correct vs cmaj-wrong** — for visualizing the
   "wrong-setup-with-right-arithmetic" failure mode. Pulls two branches of
   the same problem from rich_substrate_n500.jsonl: one whose extracted
   answer == gold, one whose ≠ gold. Renders both stacked vertically with
   shared axis, so you can read across to compare the entropy commit
   patterns of right vs wrong.

B. **all-LLaDA vs AR-hybrid** — for showing the AR-splicing mechanism. Takes
   two v2 trace JSONLs (one all_llada, one ar_at_*) and renders side-by-side.
   The AR-grafted cells are highlighted in the right pane.

Usage:
    # mode A: cmaj-failure visualization on problem 1071 (gold=251, cmaj=371)
    python make_comparison_gif.py --substrate-pid 1071

    # mode B: all-LLaDA vs AR-hybrid
    python make_comparison_gif.py --left traces/trace_v2_p10_all_llada.jsonl \
                                   --right traces/trace_v2_p30_ar_at_0_n4.jsonl
"""
from __future__ import annotations
import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRACES_DIR = REPO_ROOT / "phase2/inference_viz/traces"
COMP_DIR = TRACES_DIR / "comparisons"
SUBSTRATE_PATH = REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate_n500.jsonl"
if not SUBSTRATE_PATH.exists():
    SUBSTRATE_PATH = pathlib.Path("/tmp/rich_substrate_n500.jsonl")


def _color_for_entropy(e, max_e=4.0):
    if e is None or not np.isfinite(e):
        return (0.15, 0.15, 0.15)
    t = max(0.0, min(1.0, e / max_e))
    blue = np.array([0.114, 0.306, 0.847])
    amber = np.array([0.706, 0.325, 0.035])
    return tuple(((1 - t) * blue + t * amber).tolist())


def _short(s, maxlen=6):
    if not s: return ""
    s = s.strip().replace("\n", "↵").replace("\t", "→")
    return s[:maxlen-1] + "…" if len(s) > maxlen else s


def trace_records_from_v2_jsonl(path: pathlib.Path):
    """v2 JSONL: each line = one sub-block step record."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def trace_records_from_substrate_branch(branch: dict):
    """Substrate branch: dict with embedded list of per-step records."""
    return branch.get("records", [])


def build_grid(records, n_blocks_max=4, block_len_max=32):
    """Convert per-step records → (n_blocks, block_len) entropy + token grids,
    plus list of AR-grafted (block, position) pairs."""
    n_blocks = max(n_blocks_max, max((r.get("sub_block", 0) for r in records), default=0) + 1)
    block_len = max(block_len_max, max((len(r.get("token_strings", [])) for r in records), default=0))
    tokens = [["" for _ in range(block_len)] for _ in range(n_blocks)]
    ent = [[None for _ in range(block_len)] for _ in range(n_blocks)]
    interventions = [""] * n_blocks
    ar_grafted_cells = []  # list of (block_idx, position_in_block)
    for r in records:
        b = r.get("sub_block", 0)
        for i, (tok, e) in enumerate(zip(r.get("token_strings", []), r.get("entropy", []))):
            if i < block_len:
                tokens[b][i] = tok
                ent[b][i] = float(e) if e is not None else None
        mi = r.get("manual_intervention") or {}
        interventions[b] = mi.get("directive", "continue_llada") if isinstance(mi, dict) else ""
        ar_payload = r.get("ar_extend") or {}
        n_grafted = len(ar_payload.get("tokens_grafted", []) or [])
        # Mark first n_grafted positions of the *next* block (or this block) as grafted
        for j in range(n_grafted):
            if b < n_blocks - 1:
                ar_grafted_cells.append((b + 1, j))
    return tokens, ent, interventions, ar_grafted_cells


def render_pane(ax, tokens, ent, interventions, ar_cells, title, active_block):
    n_blocks = len(tokens)
    block_len = len(tokens[0]) if tokens else 32
    ax.clear()
    ax.set_xlim(0, block_len)
    ax.set_ylim(0, n_blocks)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([y + 0.5 for y in range(n_blocks)])
    ax.set_yticklabels([f"sub-block {i}" for i in range(n_blocks)], fontsize=8)
    ax.set_facecolor("#0a0a0a")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.set_title(title, color="#eee", fontsize=10, pad=6)
    ar_cells_set = set(ar_cells)
    for r in range(n_blocks):
        for c in range(block_len):
            tok = tokens[r][c]
            e = ent[r][c]
            if tok:
                color = _color_for_entropy(e)
                edge = "#fff" if r == active_block else "#444"
                lw = 1.0 if r == active_block else 0.3
                if (r, c) in ar_cells_set:
                    edge = "#22ee22"  # AR-grafted = bright green border
                    lw = 1.5
                rect = Rectangle((c, r), 1, 1, facecolor=color, edgecolor=edge, linewidth=lw)
                ax.add_patch(rect)
                lum = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                ax.text(c + 0.5, r + 0.5, _short(tok), ha="center", va="center",
                        fontsize=6, color=("#0a0a0a" if lum > 0.55 else "#fafafa"),
                        family="monospace")
            else:
                rect = Rectangle((c, r), 1, 1, facecolor="#0a0a0a", edgecolor="#222", linewidth=0.3)
                ax.add_patch(rect)


def render_comparison(left_records, right_records, left_title, right_title,
                       overall_title, out_path, frame_ms=900):
    lt, le, li, lar = build_grid(left_records)
    rt, re, ri, rar = build_grid(right_records)
    n_frames = max(len(lt), len(rt))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, max(3, len(lt) * 1.0)),
                                      dpi=110, gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor("#0a0a0a")
    sup = fig.suptitle(overall_title, color="#eee", fontsize=12, y=0.98)

    # Pre-build cumulative views
    def cum_view(tokens, ent, up_to_block):
        ct = [["" for _ in row] for row in tokens]
        ce = [[None for _ in row] for row in ent]
        for b in range(min(up_to_block + 1, len(tokens))):
            ct[b] = list(tokens[b])
            ce[b] = list(ent[b])
        return ct, ce

    def update(idx):
        ct_l, ce_l = cum_view(lt, le, idx)
        ct_r, ce_r = cum_view(rt, re, idx)
        render_pane(ax_l, ct_l, ce_l, li, lar, left_title, idx)
        render_pane(ax_r, ct_r, ce_r, ri, rar, right_title, idx)
        return [sup]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=frame_ms, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer="pillow", fps=max(1, 1000 // frame_ms))
    plt.close(fig)
    return out_path


def comparison_substrate_pid(pid: str):
    """Find one cmaj-correct + one cmaj-wrong branch of `pid` in substrate."""
    if not SUBSTRATE_PATH.exists():
        raise FileNotFoundError(f"substrate {SUBSTRATE_PATH} missing")
    branches = []
    with SUBSTRATE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if str(r.get("problem_id")) == str(pid):
                branches.append(r)
    if len(branches) < 2:
        raise ValueError(f"pid={pid} has only {len(branches)} branches")
    correct = next((b for b in branches if b["correct"]), None)
    wrong = next((b for b in branches if not b["correct"]), None)
    if correct is None or wrong is None:
        raise ValueError(f"pid={pid}: missing correct or wrong branch")
    gold = branches[0]["gold"]
    out_path = COMP_DIR / f"compare_substrate_p{pid}_correct_vs_wrong.gif"
    overall = (f"problem_id={pid}  ·  gold={gold}  ·  same problem, two branches  "
               f"·  ✅ correct (left) vs ❌ wrong (right)")
    return render_comparison(
        trace_records_from_substrate_branch(correct),
        trace_records_from_substrate_branch(wrong),
        f"branch_{correct['branch_idx']} ✅ extracted={correct['extracted']}",
        f"branch_{wrong['branch_idx']} ❌ extracted={wrong['extracted']}",
        overall, out_path,
    )


def comparison_v2_jsonls(left_path: pathlib.Path, right_path: pathlib.Path):
    """Side-by-side of two v2 trace JSONLs (different problems is fine)."""
    lr = trace_records_from_v2_jsonl(left_path)
    rr = trace_records_from_v2_jsonl(right_path)
    out_path = COMP_DIR / f"compare_{left_path.stem}__VS__{right_path.stem}.gif"
    overall = f"{left_path.stem}  vs  {right_path.stem}  (AR-grafted cells = green border)"
    return render_comparison(lr, rr, left_path.stem, right_path.stem, overall, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate-pid", help="problem_id from rich_substrate; emits cmaj-correct vs cmaj-wrong GIF")
    ap.add_argument("--left", type=pathlib.Path, help="left trace JSONL (v2 schema)")
    ap.add_argument("--right", type=pathlib.Path, help="right trace JSONL (v2 schema)")
    args = ap.parse_args()

    if args.substrate_pid:
        out = comparison_substrate_pid(args.substrate_pid)
        print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    elif args.left and args.right:
        out = comparison_v2_jsonls(args.left, args.right)
        print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    else:
        ap.error("provide --substrate-pid OR (--left AND --right)")


if __name__ == "__main__":
    main()
