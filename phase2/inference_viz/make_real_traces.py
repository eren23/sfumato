"""Standalone real-mode STATUS-schema trace generator (no FastAPI, no HTTP).

Imports `e4.diff_llada` directly with a `step_callback` that records each
sub-block StepState as a JSONL record matching the schema pinned in
`phase2/STATUS.md` "Workstream C → Trace JSONL schema".

For each (problem_idx, planned_directive_sequence), the callback:
  1. Records the StepState (the mechanism the sampler is about to run = "llada").
  2. Adds a `manual_intervention` field with the directive that a hypothetical
     policy / human would have chosen at this boundary (from the planned
     sequence). This is the substrate D1 mode-router needs.
  3. Returns `continue_llada()` to the sampler so generation completes
     consistently. (Actually injecting AR-extend / cmaj-branch mid-flight is
     out of scope for batch trace generation — that's what the interactive
     server.py + Gradio app exists for.)

Output: one JSONL per (problem, sequence_tag) under
`phase2/inference_viz/traces/trace_real_p{idx}_{tag}.jsonl`.

Usage on pod:
    /workspace/sfumato/.venv/bin/python phase2/inference_viz/make_real_traces.py
"""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e4 import diff_llada  # noqa: E402
from e4 import runner  # noqa: E402

TRACES_DIR = REPO_ROOT / "phase2" / "inference_viz" / "traces"
DEV_INDICES_PATH = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"

# 7 problems × planned directive sequences (each sequence has 4 entries — one per sub-block boundary)
RUNS = [
    (10, "all_llada", ["continue_llada"] * 4),
    (20, "all_llada", ["continue_llada"] * 4),
    (30, "ar_handoff", ["switch_to_ar:6", "continue_llada", "continue_llada", "continue_llada"]),
    (40, "cmaj_branch", ["continue_llada", "continue_llada", "continue_llada", "branch_cmaj:5"]),
    (50, "mid_ar_handoff", ["continue_llada", "switch_to_ar:12", "continue_llada", "continue_llada"]),
    (60, "all_llada", ["continue_llada"] * 4),
    (70, "early_cmaj", ["continue_llada", "continue_llada", "branch_cmaj:3", "continue_llada"]),
]

LORA_PATH = "eren23/sfumato-llada-prefix-robust-v3"
COMMIT_LORA_PATH = "eren23/sfumato-llada-commit-v3"
DIFF_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
K_STEPS = 64
SEED = 0
TEMPERATURE = 0.0
APPLY_COMMIT = True
COMMIT_N_BLOCKS = 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_to_record(
    session_id: str,
    problem_idx: int,
    state: diff_llada.StepState,
    intervention: dict | None,
) -> dict:
    """Mirror the schema in phase2/inference_viz/server.py:_state_to_record."""
    return {
        "session_id": session_id,
        "problem_idx": problem_idx,
        "step_idx": state.step_idx,
        "sub_block": state.sub_block,
        "mechanism": state.mechanism,  # always "llada" in batch mode
        "tokens_committed": state.tokens_committed,
        "token_strings": state.token_strings,
        "positions": state.positions,
        "entropy": state.entropy,
        "top_k_logits": [
            [[int(t), float(p)] for (t, p) in row] for row in state.top_k_logits
        ],
        "commit_lora_active": bool(state.commit_lora_active),
        "logit_shift_norm": state.logit_shift_norm,
        "temperature": float(state.temperature),
        "steps_per_block": int(state.steps_per_block),
        "manual_intervention": intervention,
        "ar_extend": None,    # not injected in batch mode
        "cmaj_branch": None,  # not injected in batch mode
        "wallclock_ms": int(state.wallclock_ms),
        "timestamp": _utc_now_iso(),
    }


def _parse_directive_str(s: str) -> dict:
    """`continue_llada` -> {kind:..}; `switch_to_ar:6` -> {kind:.., n_tokens:6}"""
    if ":" not in s:
        return {"kind": s}
    kind, arg = s.split(":", 1)
    if kind == "switch_to_ar":
        return {"kind": kind, "n_tokens": int(arg), "model_name": "Qwen/Qwen2.5-0.5B-Instruct"}
    if kind == "branch_cmaj":
        return {"kind": kind, "b": int(arg)}
    return {"kind": kind}


def _load_problem(idx: int) -> dict:
    """Use runner.load_problems to read the frozen index → real GSM8K problem."""
    # load_problems takes n=count and returns first n; we want a specific idx.
    # Easiest: load enough problems and return the matching one by id.
    problems = runner.load_problems(idx + 1, DEV_INDICES_PATH)
    for p in problems:
        if p["id"] == str(idx):
            return p
    # Fallback: positional
    if idx < len(problems):
        return problems[idx]
    raise ValueError(f"problem_idx={idx} not found in dev indices")


def main() -> int:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[trace] loading model (mock=False)...", flush=True)
    t0 = time.time()
    model = diff_llada.load(DIFF_MODEL, mock=False, lora_path=LORA_PATH, commit_lora_path=COMMIT_LORA_PATH)
    print(f"[trace] model loaded in {time.time() - t0:.1f}s", flush=True)

    summary = []
    for run_i, (problem_idx, tag, directive_strs) in enumerate(RUNS):
        print(f"\n[trace] {run_i + 1}/{len(RUNS)} problem_idx={problem_idx} tag={tag}", flush=True)
        try:
            problem = _load_problem(problem_idx)
        except Exception as e:
            print(f"  FAIL load_problem: {e}", flush=True)
            summary.append({"problem_idx": problem_idx, "tag": tag, "error": f"load: {e}"})
            continue

        directives = [_parse_directive_str(s) for s in directive_strs]
        session_id = f"batch-{uuid.uuid4().hex[:10]}"
        records: list[dict] = []
        sub_block_seen = {"i": 0}

        def cb(state: diff_llada.StepState) -> diff_llada.StepDirective:
            i = sub_block_seen["i"]
            planned = directives[i] if i < len(directives) else {"kind": "continue_llada"}
            intervention = {
                "directive": planned["kind"],
                "args": {k: v for k, v in planned.items() if k != "kind"},
            }
            records.append(_state_to_record(session_id, problem_idx, state, intervention))
            sub_block_seen["i"] += 1
            # Always actually continue LLaDA — we only RECORD the intent, don't inject.
            # (Actual mid-flight AR-extend / cmaj-branch is interactive-server-only.)
            return diff_llada.StepDirective.continue_llada()

        t1 = time.time()
        try:
            text, _used = model.denoise_block(
                prompt=problem["question"],
                k_steps=K_STEPS,
                seed=SEED,
                temperature=TEMPERATURE,
                apply_commit=APPLY_COMMIT,
                commit_n_blocks=COMMIT_N_BLOCKS,
                step_callback=cb,
            )
        except Exception as e:
            print(f"  FAIL sampler: {e}", flush=True)
            summary.append({"problem_idx": problem_idx, "tag": tag, "error": f"sampler: {e}"})
            continue
        wall = time.time() - t1

        out_path = TRACES_DIR / f"trace_real_p{problem_idx}_{tag}.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"  wrote {out_path.name} ({len(records)} records, {wall:.1f}s)", flush=True)
        summary.append({
            "problem_idx": problem_idx, "tag": tag,
            "records": len(records), "wall_s": round(wall, 2),
            "size_bytes": out_path.stat().st_size,
            "final_text_tail": text[-100:] if text else "",
        })

    summary_path = TRACES_DIR / "make_real_traces_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[trace] done — wrote {summary_path.name}", flush=True)
    n_ok = sum(1 for s in summary if "records" in s)
    print(f"[trace] {n_ok}/{len(RUNS)} traces succeeded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
