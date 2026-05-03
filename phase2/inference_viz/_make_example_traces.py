"""Drive the live visualizer server to produce the three canonical example
traces required by the Phase-2 Workstream-C done criteria.

Usage:
    MOCK_MODELS=1 python3 phase2/inference_viz/server.py --port 8765 &
    python3 phase2/inference_viz/_make_example_traces.py

Writes:
    phase2/inference_viz/traces/trace_all_llada.jsonl
    phase2/inference_viz/traces/trace_mid_ar_handoff.jsonl
    phase2/inference_viz/traces/trace_cmaj_branching.jsonl

These are produced via the same HTTP endpoints the Gradio app uses, so
they're a faithful exercise of the production flow (not synthesized by hand).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = REPO_ROOT / "phase2" / "inference_viz" / "traces"
BACKEND = "http://127.0.0.1:8765"


def _start(**cfg) -> str:
    r = requests.post(f"{BACKEND}/session/start", json=cfg, timeout=60)
    r.raise_for_status()
    return r.json()["session_id"]


def _step(sid: str, **kw) -> dict:
    r = requests.post(f"{BACKEND}/session/{sid}/step", json=kw, timeout=60)
    r.raise_for_status()
    return r.json()


def _save(sid: str, name: str) -> Path:
    r = requests.get(f"{BACKEND}/session/{sid}/trace", timeout=30)
    r.raise_for_status()
    out = TRACES_DIR / name
    out.write_text(r.text)
    return out


def _drive(sid: str, directives: list[dict]) -> None:
    """Pull the first state, then alternate directive -> state."""
    # First step has no directive payload meaning — server treats first call
    # as "just hand me the first sub-block state" and ignores directive.
    _step(sid, directive="continue_llada")
    for d in directives:
        out = _step(sid, **d)
        if out.get("done"):
            return


def main() -> int:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = dict(
        problem_idx=42, k_steps=32, seed=0, temperature=0.0,
        apply_commit=True, commit_n_blocks=3, mock=True,
    )

    # 1) all-LLaDA: continue at every boundary (4 sub-blocks).
    sid = _start(**base_cfg)
    _drive(sid, [
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
    ])
    p1 = _save(sid, "trace_all_llada.jsonl")
    print(f"  wrote {p1.relative_to(REPO_ROOT)}")

    # 2) mid-AR handoff: AR-extend at block 0, then continue LLaDA.
    sid = _start(**base_cfg)
    _drive(sid, [
        dict(directive="switch_to_ar", n_tokens=6, model_name="Qwen/Qwen2.5-0.5B-Instruct"),
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
    ])
    p2 = _save(sid, "trace_mid_ar_handoff.jsonl")
    print(f"  wrote {p2.relative_to(REPO_ROOT)}")

    # 3) cmaj branching at block 3 (ends generation early — that's the point).
    sid = _start(**base_cfg)
    _drive(sid, [
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
        dict(directive="continue_llada"),
        dict(directive="branch_cmaj", b=5),
    ])
    p3 = _save(sid, "trace_cmaj_branching.jsonl")
    print(f"  wrote {p3.relative_to(REPO_ROOT)}")

    print("[traces] all 3 example traces generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
