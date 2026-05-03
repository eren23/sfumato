"""Gradio frontend for the step-by-step inference visualizer.

Talks to ``server.py`` over HTTP. Renders a 4×32 token grid (one row per
LLaDA sub-block, one cell per token), color-codes cells by entropy, and
exposes mid-generation directive controls: Continue LLaDA, AR-extend,
Cmaj-branch, Stop. Live trace pane on the right; "Save Trace" button at
the bottom dumps the current session's JSONL into
``phase2/inference_viz/traces/``.

Run locally
-----------
    BACKEND_URL=http://127.0.0.1:8765 \
        python3 phase2/inference_viz/app.py --port 7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = REPO_ROOT / "phase2" / "inference_viz" / "traces"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Minimal HTTP client ───────────────────────────────────────────────────


class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.problem: dict = {}
        self.cfg: dict = {}
        self.steps: list[dict] = []
        self.done: bool = False
        self.final_text: str = ""
        self.error: Optional[str] = None

    def health(self) -> dict:
        try:
            return requests.get(f"{self.base_url}/healthz", timeout=5).json()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def start(self, **kwargs) -> dict:
        r = requests.post(f"{self.base_url}/session/start", json=kwargs, timeout=120)
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        self.problem = data["problem"]
        self.cfg = data["cfg"]
        self.steps = []
        self.done = False
        self.final_text = ""
        self.error = None
        return data

    def step(
        self,
        directive: str = "continue_llada",
        n_tokens: int = 6,
        model_name: str = "",
        b: int = 5,
    ) -> dict:
        if not self.session_id:
            return {"error": "no session"}
        body = {
            "directive": directive,
            "n_tokens": n_tokens,
            "model_name": model_name,
            "b": b,
        }
        r = requests.post(
            f"{self.base_url}/session/{self.session_id}/step", json=body, timeout=180
        )
        r.raise_for_status()
        data = r.json()
        if data.get("done"):
            self.done = True
            self.final_text = data.get("final_text", "")
            self.error = data.get("error")
        elif "step" in data:
            self.steps.append(data["step"])
        return data

    def trace(self) -> str:
        if not self.session_id:
            return ""
        r = requests.get(f"{self.base_url}/session/{self.session_id}/trace", timeout=30)
        r.raise_for_status()
        return r.text


# ── Rendering helpers ─────────────────────────────────────────────────────


def _entropy_to_color(ent: float) -> str:
    """Map Shannon entropy (nats) -> hex color along blue-700 -> amber-700.

    Low entropy = confident = cool; high entropy = uncertain = warm.
    Stripe-Press cool/warm ramp from STATUS.md palette (matches Workstream A).
    """
    # Clamp into [0, 2.0] (typical CoT range; > 2 is rare for committed tokens)
    e = max(0.0, min(2.0, float(ent)))
    t = e / 2.0
    # Linear blend in RGB between #1d4ed8 and #b45309.
    c0 = (0x1d, 0x4e, 0xd8)
    c1 = (0xb4, 0x53, 0x09)
    rgb = tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _grid_html(steps: list[dict], num_blocks: int = 4, block_len: int = 32) -> str:
    """Render the 4×32 token grid as HTML."""
    # Build a (block, slot) -> record map.
    cells: dict[tuple[int, int], dict] = {}
    for rec in steps:
        block = int(rec.get("sub_block", 0))
        positions = rec.get("positions", [])
        toks = rec.get("token_strings", [])
        ents = rec.get("entropy", [])
        topks = rec.get("top_k_logits", [])
        commit = bool(rec.get("commit_lora_active", False))
        mech = rec.get("mechanism", "llada")
        intervention = rec.get("manual_intervention")
        if not positions:
            continue
        prompt_offset = positions[0] - (block * block_len)
        for j, pos in enumerate(positions):
            slot = pos - prompt_offset - block * block_len
            if 0 <= slot < block_len:
                cells[(block, slot)] = {
                    "tok": toks[j] if j < len(toks) else "?",
                    "ent": ents[j] if j < len(ents) else 0.0,
                    "topk": topks[j] if j < len(topks) else [],
                    "commit": commit,
                    "mech": mech,
                    "intervention": intervention,
                }

    rows_html = []
    for b in range(num_blocks):
        row_cells = []
        for s in range(block_len):
            cell = cells.get((b, s))
            if cell is None:
                row_cells.append(
                    "<div style='width:34px;height:30px;border:1px solid #e5e7eb;"
                    "background:#fafafa;display:inline-block;margin:1px;"
                    "border-radius:3px;'></div>"
                )
                continue
            color = _entropy_to_color(cell["ent"])
            border = "2px solid #065f46" if cell["commit"] else "1px solid #d1d5db"
            mech_marker = ""
            if cell["mech"] == "ar_extend":
                mech_marker = "<sup style='color:#b45309;font-size:8px'>AR</sup>"
            elif cell["mech"] == "cmaj_branch":
                mech_marker = "<sup style='color:#7c3aed;font-size:8px'>CM</sup>"
            raw = (cell["tok"] or "").strip()
            # Strip mock-only `<m{b}_{i}>` wrapper so cells show the index, not "&lt;".
            if raw.startswith("<") and raw.endswith(">") and len(raw) > 2:
                raw = raw[1:-1]
            display_raw = raw[:4] if raw else "·"
            display = display_raw.replace("<", "&lt;").replace(">", "&gt;")
            tok = raw.replace("<", "&lt;").replace(">", "&gt;")
            tt_lines = [
                f"token: {tok!r}",
                f"entropy: {cell['ent']:.3f} nats",
                f"mechanism: {cell['mech']}",
                f"commit-LoRA: {cell['commit']}",
            ]
            if cell["topk"]:
                tt_lines.append("top-5:")
                for tid, prob in cell["topk"][:5]:
                    tt_lines.append(f"  {tid}: {prob:.3f}")
            tooltip = "\n".join(tt_lines).replace('"', "&quot;")
            row_cells.append(
                f"<div title=\"{tooltip}\" "
                f"style='width:34px;height:30px;background:{color};color:white;"
                f"font-family:monospace;font-size:10px;display:inline-flex;"
                f"align-items:center;justify-content:center;margin:1px;"
                f"border:{border};border-radius:3px;cursor:help;'>"
                f"{display}{mech_marker}</div>"
            )
        rows_html.append(
            f"<div style='white-space:nowrap;'>"
            f"<span style='display:inline-block;width:60px;color:#6b7280;"
            f"font-family:sans-serif;font-size:11px;'>blk {b}</span>"
            f"{''.join(row_cells)}</div>"
        )

    legend = (
        "<div style='margin-top:12px;font-family:sans-serif;font-size:11px;color:#374151;'>"
        "<b>Legend:</b> color = entropy (cool=confident, warm=uncertain) · "
        "<span style='border:2px solid #065f46;padding:1px 4px;'>green border</span> "
        "= commit-LoRA active · "
        "<sup style='color:#b45309'>AR</sup> = AR-extend graft · "
        "<sup style='color:#7c3aed'>CM</sup> = cmaj branch token"
        "</div>"
    )
    return (
        "<div style='font-family:sans-serif;'>"
        + "".join(rows_html)
        + legend
        + "</div>"
    )


def _state_summary(client: BackendClient) -> str:
    if not client.session_id:
        return "_(no session)_"
    n_steps = len(client.steps)
    finished = client.done or n_steps >= 4
    parts = [
        f"**session_id:** `{client.session_id}`",
        f"**problem_idx:** {client.problem.get('idx', '?')}",
        f"**question:** {client.problem.get('question', '')[:200]}",
        f"**gold:** {client.problem.get('answer', '?')}",
        f"**steps so far:** {n_steps} / 4",
        f"**done:** {finished}"
        + ("  _(all sub-blocks committed)_" if finished and not client.done else ""),
    ]
    if client.error:
        parts.append(f"**ERROR:** `{client.error}`")
    if client.final_text:
        parts.append(f"\n**final text:**\n```\n{client.final_text[:600]}\n```")
    return "\n\n".join(parts)


def _trace_pane(client: BackendClient) -> str:
    if not client.steps:
        return "_(no trace yet)_"
    lines = []
    for r in client.steps:
        mech = r.get("mechanism", "?")
        commit = "yes" if r.get("commit_lora_active") else "no"
        n = len(r.get("tokens_committed", []))
        ent = r.get("entropy", [])
        ent_avg = sum(ent) / max(len(ent), 1)
        intervention = r.get("manual_intervention")
        intv = (
            f" intv={intervention.get('directive')}"
            if intervention
            else ""
        )
        lines.append(
            f"- block {r.get('sub_block')}: mech={mech} n_committed={n} "
            f"avg_entropy={ent_avg:.2f} commit_lora={commit}{intv}"
        )
    return "\n".join(lines)


def _save_trace(client: BackendClient, label: str) -> str:
    if not client.session_id:
        return "no session"
    body = client.trace()
    if not body.strip():
        return "trace empty"
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in label) or "trace"
    out = TRACES_DIR / f"{safe}.jsonl"
    out.write_text(body)
    return f"saved -> {out.relative_to(REPO_ROOT)} ({len(body)} bytes)"


# ── Gradio UI ─────────────────────────────────────────────────────────────


def build_ui(default_backend: str, default_problem_idx: int):
    import gradio as gr  # type: ignore

    with gr.Blocks(title="Sfumato Inference Visualizer", theme=gr.themes.Default()) as demo:
        state = gr.State(value=BackendClient(default_backend))
        gr.Markdown(
            "# Sfumato — step-by-step LLaDA + commit-LoRA + AR-extend\n"
            "_Workstream C, Phase 2._\n\n"
            "**What you're looking at.** LLaDA generates a 128-token answer in **4 sub-blocks of 32 tokens** "
            "(`gen_length=128`, `sub_block_length=32`). Each row of the grid below is one sub-block; "
            "each cell is one committed token. Generation goes left→right, top→bottom.\n\n"
            "**Color = per-token Shannon entropy.** Cool blue = the model was confident "
            "(low entropy, sharp distribution); warm amber = uncertain (high entropy). "
            "Watch how later blocks get cooler — the model commits fewer alternatives "
            "as the answer crystallizes.\n\n"
            "**Green border = commit-LoRA active on that block.** With `commit_n_blocks=3` (the v3 "
            "headline setting), commit-LoRA fires on the LAST 3 sub-blocks (rows 1,2,3). "
            "In this mock you'll see those rows visibly cooler than block 0 — that's the "
            "trained adapter sharpening the answer-span logits.\n\n"
            "**Hover any cell** for the raw token, top-5 alternatives, and which mechanism "
            "produced it. **Auto-run** below = one click, watch all 4 blocks fill in. "
            "**Manual mode** = stop at any boundary and pivot (AR-extend grafts Qwen tokens; "
            "Cmaj-branch forks into _b_ parallel LLaDA branches and majority-votes).\n\n"
            "_Note: with `MOCK_MODELS=1` the **problem** is a synthetic placeholder "
            "(`Mock problem N: 2 + N = ?`, gold = `2+N`) and the **cells** are generic "
            "reasoning words — neither comes from a real model or a real GSM8K item. "
            "The point of mock is to exercise the UI/HTTP/queue plumbing end-to-end. "
            "For real LLaDA-8B + v3 LoRA tokens (and real GSM8K problems), run on a "
            "4090 pod — see `LOCAL_QUICKSTART.md` Option B._"
        )

        with gr.Row():
            with gr.Column(scale=2):
                problem_idx = gr.Number(
                    value=default_problem_idx, label="problem_idx", precision=0
                )
                k_steps = gr.Number(value=32, label="k_steps (diffusion)", precision=0)
                seed = gr.Number(value=0, label="seed", precision=0)
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.0, step=0.05, label="temperature"
                )
                apply_commit = gr.Checkbox(value=True, label="apply commit-LoRA")
                commit_n_blocks = gr.Slider(
                    minimum=1, maximum=4, value=3, step=1, label="commit_n_blocks (last N)"
                )
                mock_chk = gr.Checkbox(
                    value=bool(int(os.environ.get("MOCK_MODELS", "1"))),
                    label="MOCK_MODELS (no GPU)",
                )
                start_btn = gr.Button("Start session", variant="primary")
                health_md = gr.Markdown()
            with gr.Column(scale=4):
                grid_html = gr.HTML(label="Token grid (4×32)")
                summary_md = gr.Markdown(label="Session state")

        gr.Markdown(
            "### Auto mode — one click, runs all 4 sub-blocks"
        )
        with gr.Row():
            auto_btn = gr.Button("Auto-run (all blocks)", variant="primary")
            auto_delay = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.4, step=0.1,
                label="delay between blocks (s) — slower = easier to watch",
            )

        gr.Markdown("### Manual mode — choose a directive for the next sub-block")
        with gr.Row():
            cont_btn = gr.Button("Continue LLaDA")
            with gr.Column():
                ar_n = gr.Slider(minimum=1, maximum=24, value=6, step=1, label="AR n_tokens")
                ar_btn = gr.Button("Switch to AR-extend")
            with gr.Column():
                br_b = gr.Slider(minimum=2, maximum=8, value=5, step=1, label="cmaj b")
                br_btn = gr.Button("Cmaj branch")
            stop_btn = gr.Button("Stop")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Trace (live)")
                trace_md = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Save trace")
                label = gr.Textbox(value="trace_session", label="filename label")
                save_btn = gr.Button("Save trace JSONL")
                save_msg = gr.Markdown()

        # ── Handlers ──────────────────────────────────────────────────────

        def on_start(client, problem_idx, k_steps, seed, temperature,
                     apply_commit, commit_n_blocks, mock):
            client.base_url = client.base_url  # keep
            try:
                resp = client.start(
                    problem_idx=int(problem_idx),
                    k_steps=int(k_steps),
                    seed=int(seed),
                    temperature=float(temperature),
                    apply_commit=bool(apply_commit),
                    commit_n_blocks=int(commit_n_blocks),
                    mock=bool(mock),
                )
                # Auto-fetch the first sub-block so the grid shows immediately.
                client.step(directive="continue_llada")
                health = client.health()
                health_str = f"backend ok={health.get('ok')} mock={health.get('mock')}"
            except Exception as exc:
                health_str = f"start failed: {exc!r}"
            return (
                client,
                _grid_html(client.steps),
                _state_summary(client),
                _trace_pane(client),
                health_str,
            )

        def on_step(client, directive, n_tokens, model_name, b):
            try:
                client.step(directive=directive, n_tokens=int(n_tokens),
                            model_name=model_name, b=int(b))
            except Exception as exc:
                client.error = repr(exc)
            return (
                client,
                _grid_html(client.steps),
                _state_summary(client),
                _trace_pane(client),
            )

        def on_continue(client):
            return on_step(client, "continue_llada", 6, "", 5)

        def on_ar(client, n):
            return on_step(client, "switch_to_ar", n, "", 5)

        def on_branch(client, b):
            return on_step(client, "branch_cmaj", 6, "", b)

        def on_stop(client):
            return on_step(client, "stop", 6, "", 5)

        def on_auto(client, delay):
            """Auto-advance: continue_llada until session done or 4 blocks reached.

            Generator handler so Gradio streams each sub-block to the UI as
            it lands, rather than blocking until the whole sequence finishes.
            """
            if not client.session_id:
                client.error = "no session — click Start first"
                yield (
                    client, _grid_html(client.steps),
                    _state_summary(client), _trace_pane(client),
                )
                return
            max_blocks = 4
            while not client.done and len(client.steps) < max_blocks:
                try:
                    client.step(directive="continue_llada")
                except Exception as exc:
                    client.error = repr(exc)
                    break
                yield (
                    client, _grid_html(client.steps),
                    _state_summary(client), _trace_pane(client),
                )
                if client.done:
                    break
                time.sleep(max(0.0, float(delay)))

        def on_save(client, lbl):
            try:
                msg = _save_trace(client, lbl)
            except Exception as exc:
                msg = f"save failed: {exc!r}"
            return msg

        start_btn.click(
            on_start,
            inputs=[state, problem_idx, k_steps, seed, temperature,
                    apply_commit, commit_n_blocks, mock_chk],
            outputs=[state, grid_html, summary_md, trace_md, health_md],
        )
        auto_btn.click(on_auto, [state, auto_delay],
                       [state, grid_html, summary_md, trace_md])
        cont_btn.click(on_continue, [state],
                       [state, grid_html, summary_md, trace_md])
        ar_btn.click(on_ar, [state, ar_n],
                     [state, grid_html, summary_md, trace_md])
        br_btn.click(on_branch, [state, br_b],
                     [state, grid_html, summary_md, trace_md])
        stop_btn.click(on_stop, [state],
                       [state, grid_html, summary_md, trace_md])
        save_btn.click(on_save, [state, label], [save_msg])

    return demo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8765"))
    p.add_argument("--problem-idx", type=int, default=42)
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    try:
        import gradio  # type: ignore  # noqa: F401
    except ImportError:
        print(
            "gradio + requests required. Install with: pip install gradio requests",
            file=sys.stderr,
        )
        return 1

    demo = build_ui(default_backend=args.backend, default_problem_idx=args.problem_idx)
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
