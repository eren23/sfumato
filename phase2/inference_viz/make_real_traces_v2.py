"""make_real_traces v2: ACTUALLY executes switch_to_ar (grafts AR tokens into x).

v1 (make_real_traces.py) just RECORDED the planned directive but always returned
continue_llada(). v2 ports `_ar_extend_inject` from server.py so AR-extend
genuinely happens — the resulting trace has AR-grafted tokens visible in the
sub-block records, and the GIF can show them in a distinct color.

cmaj-branch is harder (needs separate denoise calls from scratch with deep-copy
of x); for v2 we still record it but execute as continue_llada. AR-extend is
the headline mechanism for the user's "see it in action" ask.

Output: phase2/inference_viz/traces/trace_v2_p{idx}_{tag}.jsonl
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

from e4 import ar_qwen, diff_llada, runner  # noqa: E402

TRACES_DIR = REPO_ROOT / "phase2" / "inference_viz" / "traces"
DEV_INDICES_PATH = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"

DIFF_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LORA_PATH = "eren23/sfumato-llada-prefix-robust-v3"
COMMIT_LORA_PATH = "eren23/sfumato-llada-commit-v3"
AR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

K_STEPS = 64
SEED = 0
TEMPERATURE = 0.0
APPLY_COMMIT = True
COMMIT_N_BLOCKS = 3

# 30 problems × varied directive sequences with REAL AR-execution
RUNS = []
# 10 plain-LLaDA traces
for i in range(10):
    RUNS.append((10 + i, "all_llada", ["continue_llada"] * 4))
# 10 AR-handoff traces (varied position, varied N tokens)
for i in range(10):
    pi = 30 + i
    n_tokens = [4, 6, 8, 12][i % 4]
    pos = i % 4  # which sub-block boundary to inject AR
    seq = ["continue_llada"] * 4
    seq[pos] = f"switch_to_ar:{n_tokens}"
    RUNS.append((pi, f"ar_at_{pos}_n{n_tokens}", seq))
# 10 cmaj-branch traces (recorded only; not actually executed)
for i in range(10):
    pi = 60 + i
    pos = (i + 2) % 4
    b = 3 if i % 2 else 5
    seq = ["continue_llada"] * 4
    seq[pos] = f"branch_cmaj:{b}"
    RUNS.append((pi, f"cmaj_at_{pos}_b{b}", seq))


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_to_record(
    session_id: str, problem_idx: int, state: diff_llada.StepState,
    intervention: dict | None, ar_payload: dict | None, cmaj_payload: dict | None,
) -> dict:
    return {
        "session_id": session_id, "problem_idx": problem_idx,
        "step_idx": state.step_idx, "sub_block": state.sub_block,
        "mechanism": state.mechanism if intervention is None else (
            "ar_extend" if intervention.get("directive") == "switch_to_ar"
            else "cmaj_branch" if intervention.get("directive") == "branch_cmaj"
            else state.mechanism
        ),
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
        "ar_extend": ar_payload,
        "cmaj_branch": cmaj_payload,
        "wallclock_ms": int(state.wallclock_ms),
        "timestamp": _utc(),
    }


def _ar_extend_inject_real(state: diff_llada.StepState, n_tokens: int,
                           model_name: str, problem_question: str, seed: int):
    """Port of server.py:_ar_extend_inject — ACTUALLY grafts AR tokens into x."""
    ar_real = ar_qwen.load(model_name, mock=False)
    text, _flops = ar_real.extend_cot(
        question=problem_question, plan=problem_question, cot="", seed=seed,
    )
    written: list[int] = []
    if state.x_handle is not None:
        try:
            import torch  # noqa
            from e4.diff_llada import _LLADA_MASK_ID
            ar_real._ensure_loaded()  # type: ignore[attr-defined]
            tok = ar_real._tokenizer  # type: ignore[attr-defined]
            ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()[:n_tokens]
            x = state.x_handle
            mask_positions = (x[0] == _LLADA_MASK_ID).nonzero(as_tuple=False).flatten().tolist()
            mask_positions = [p for p in mask_positions if p >= state.block_start]
            for tid, pos in zip(ids, mask_positions[:n_tokens]):
                x[0, pos] = int(tid)
                written.append(int(tid))
            print(f"      [AR-extend] grafted {len(written)} tokens at block boundary {state.sub_block}", flush=True)
        except Exception as e:
            print(f"      [AR-extend] graft FAIL ({e}); recorded but not injected", flush=True)
    return {"model_name": model_name, "n_tokens": n_tokens, "text": text[:200],
            "tokens_grafted": written}


def _parse(s: str) -> dict:
    if ":" not in s: return {"kind": s}
    k, v = s.split(":", 1)
    if k == "switch_to_ar": return {"kind": k, "n_tokens": int(v), "model_name": AR_MODEL}
    if k == "branch_cmaj": return {"kind": k, "b": int(v)}
    return {"kind": k}


def _load_problem(idx: int) -> dict:
    problems = runner.load_problems(idx + 1, DEV_INDICES_PATH)
    for p in problems:
        if p["id"] == str(idx):
            return p
    if idx < len(problems): return problems[idx]
    raise ValueError(f"problem_idx={idx} not found")


def main() -> int:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v2] loading models (mock=False)...", flush=True)
    t0 = time.time()
    model = diff_llada.load(DIFF_MODEL, mock=False, lora_path=LORA_PATH,
                             commit_lora_path=COMMIT_LORA_PATH)
    # Pre-load AR model so first AR-extend call doesn't pay the load cost
    ar_qwen.load(AR_MODEL, mock=False)
    print(f"[v2] loaded in {time.time()-t0:.1f}s", flush=True)

    summary = []
    for i, (problem_idx, tag, dir_strs) in enumerate(RUNS):
        print(f"\n[v2] {i+1}/{len(RUNS)} idx={problem_idx} tag={tag}", flush=True)
        try:
            problem = _load_problem(problem_idx)
        except Exception as e:
            print(f"  load fail: {e}", flush=True)
            summary.append({"problem_idx": problem_idx, "tag": tag, "error": f"load: {e}"})
            continue
        directives = [_parse(s) for s in dir_strs]
        session_id = f"v2-{uuid.uuid4().hex[:10]}"
        records: list[dict] = []
        seen = {"i": 0}

        def cb(state: diff_llada.StepState) -> diff_llada.StepDirective:
            j = seen["i"]
            planned = directives[j] if j < len(directives) else {"kind": "continue_llada"}
            intervention = {"directive": planned["kind"],
                             "args": {k: v for k, v in planned.items() if k != "kind"}}
            ar_payload = None
            cmaj_payload = None
            # ACTUALLY execute switch_to_ar
            if planned["kind"] == "switch_to_ar":
                ar_payload = _ar_extend_inject_real(
                    state, n_tokens=planned.get("n_tokens", 6),
                    model_name=planned.get("model_name", AR_MODEL),
                    problem_question=problem["question"], seed=SEED,
                )
            # cmaj_branch is recorded but not executed (needs deep-copy + parallel)
            elif planned["kind"] == "branch_cmaj":
                cmaj_payload = {"b": planned.get("b", 5),
                                 "executed": False, "note": "recorded only"}
            records.append(_state_to_record(
                session_id, problem_idx, state, intervention, ar_payload, cmaj_payload
            ))
            seen["i"] += 1
            return diff_llada.StepDirective.continue_llada()

        t1 = time.time()
        try:
            text, _used = model.denoise_block(
                prompt=problem["question"], k_steps=K_STEPS, seed=SEED,
                temperature=TEMPERATURE, apply_commit=APPLY_COMMIT,
                commit_n_blocks=COMMIT_N_BLOCKS, step_callback=cb,
            )
        except Exception as e:
            print(f"  sampler fail: {e}", flush=True)
            summary.append({"problem_idx": problem_idx, "tag": tag, "error": f"sampler: {e}"})
            continue
        wall = time.time() - t1
        out_path = TRACES_DIR / f"trace_v2_p{problem_idx}_{tag}.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        n_ar_grafted = sum(len(r.get("ar_extend", {}).get("tokens_grafted", []) or [])
                            for r in records if r.get("ar_extend"))
        print(f"  wrote {out_path.name} ({len(records)} records, {wall:.1f}s, "
              f"{n_ar_grafted} AR-grafted tokens)", flush=True)
        summary.append({
            "problem_idx": problem_idx, "tag": tag,
            "records": len(records), "wall_s": round(wall, 2),
            "ar_grafted_tokens": n_ar_grafted,
            "size_bytes": out_path.stat().st_size,
            "final_text_tail": text[-100:] if text else "",
        })

    summary_path = TRACES_DIR / "make_real_traces_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if "records" in s)
    print(f"\n[v2] done: {n_ok}/{len(RUNS)} traces, {sum(s.get('ar_grafted_tokens',0) for s in summary)} total AR-grafted tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
