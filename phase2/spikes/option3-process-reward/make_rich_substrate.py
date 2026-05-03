"""Rich substrate generator: cmaj N=50 b=5 with per-step recording per branch.

Combines what the substrate gives us (per-branch correctness labels) with what
make_real_traces.py gives us (per-step StepStates including entropy, top-k
logits, commit-LoRA status, mechanism source). Output is the substrate D3.5
option-3 (process-reward verifier) needs.

For each problem × branch:
  - run diff_llada.denoise_block with step_callback that records each StepState
  - extract final numeric answer
  - label = (extracted == gold)

Output: phase2/spikes/option3-process-reward/rich_substrate.jsonl
  one record per (problem, branch) with: {problem_idx, branch_idx, gold,
  extracted, correct, records: [StepState, ...]}

Cost: ~10 min on RTX 4090 (50 × 5 × ~2.5s).
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from e4 import diff_llada, runner  # noqa: E402

OUT_PATH = REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate.jsonl"
DEV_INDICES_PATH = REPO_ROOT / "e4/data/gsm8k_dev_200.json"

LORA_PATH = "eren23/sfumato-llada-prefix-robust-v3"
COMMIT_LORA_PATH = "eren23/sfumato-llada-commit-v3"
DIFF_MODEL = "GSAI-ML/LLaDA-8B-Instruct"

N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "50"))
PROBLEM_OFFSET = int(os.environ.get("PROBLEM_OFFSET", "0"))
B = int(os.environ.get("BRANCHES", "5"))
K_STEPS = int(os.environ.get("K_STEPS", "64"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
COMMIT_N_BLOCKS = 3

_ANSWER_RE = re.compile(r"Answer\s*[:=]\s*([\-\+]?\d[\d,]*\.?\d*)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"([\-\+]?\d[\d,]*\.?\d*)")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_answer(text: str) -> str:
    if not text: return ""
    m = _ANSWER_RE.search(text)
    if m: return m.group(1).replace(",", "").rstrip(".")
    nums = _LAST_NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def _state_to_record(state: diff_llada.StepState) -> dict:
    return {
        "step_idx": state.step_idx,
        "sub_block": state.sub_block,
        "mechanism": state.mechanism,
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
        "wallclock_ms": int(state.wallclock_ms),
    }


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rich] loading {DIFF_MODEL} (real models)", flush=True)
    t0 = time.time()
    model = diff_llada.load(DIFF_MODEL, mock=False, lora_path=LORA_PATH, commit_lora_path=COMMIT_LORA_PATH)
    print(f"[rich] loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[rich] loading {N_PROBLEMS} problems (offset={PROBLEM_OFFSET})...", flush=True)
    all_problems = runner.load_problems(N_PROBLEMS + PROBLEM_OFFSET, DEV_INDICES_PATH)
    problems = all_problems[PROBLEM_OFFSET:PROBLEM_OFFSET + N_PROBLEMS]
    print(f"[rich] {len(problems)} problems loaded (slice [{PROBLEM_OFFSET}:{PROBLEM_OFFSET+N_PROBLEMS}])", flush=True)

    n_correct_total = 0
    n_branches_total = 0
    with OUT_PATH.open("w") as f:
        for pi, problem in enumerate(problems):
            for b_idx in range(B):
                records: list[dict] = []
                def cb(state, _records=records):
                    _records.append(_state_to_record(state))
                    return diff_llada.StepDirective.continue_llada()
                t1 = time.time()
                try:
                    text, _used = model.denoise_block(
                        prompt=problem["question"],
                        k_steps=K_STEPS,
                        seed=b_idx,
                        temperature=TEMPERATURE,
                        apply_commit=True,
                        commit_n_blocks=COMMIT_N_BLOCKS,
                        step_callback=cb,
                    )
                except Exception as e:
                    print(f"  problem {pi} branch {b_idx} FAIL: {e}", flush=True)
                    text = ""
                wall = time.time() - t1
                extracted = extract_answer(text)
                gold = str(problem["answer"]).strip()
                correct = (extracted == gold)
                if correct: n_correct_total += 1
                n_branches_total += 1
                rec = {
                    "problem_id": problem["id"],
                    "problem_idx": pi,
                    "branch_idx": b_idx,
                    "gold": gold,
                    "extracted": extracted,
                    "correct": correct,
                    "branch_text": text,
                    "wall_s": round(wall, 2),
                    "records": records,
                    "ts": _utc(),
                }
                f.write(json.dumps(rec) + "\n")
                if (pi * B + b_idx) % 25 == 0:
                    print(f"[rich] {pi*B+b_idx+1}/{N_PROBLEMS*B} problem={pi} branch={b_idx} ext={extracted!r} gold={gold!r} {wall:.1f}s", flush=True)
    bar_a1 = n_correct_total / n_branches_total
    print(f"[rich] DONE: {n_correct_total}/{n_branches_total} per-branch correct = {bar_a1:.1%}", flush=True)
    print(f"[rich] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
