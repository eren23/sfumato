"""E4 runner: AR vs diffusion vs hybrid CoT on GSM8K.

Crucible-friendly. Reads config from env vars; emits stdout in the
LM-training-contract format so Crucible's OutputParser can parse it.

Conditions (env CONDITION):
    c1: pure AR (Qwen)
    c2: pure diffusion (LLaDA)
    c3: AR plan -> diffusion CoT -> AR answer (hybrid)
    c4: c3 + one extra (AR-extend, diffuse-again) round

Usage (local):
    MOCK_MODELS=1 CONDITION=c3 N_PROBLEMS=5 python e4/runner.py

Crucible parses lines like:
    step:{i}/{N} train_loss:{1-acc} val_loss:{1-acc} val_bpb:{flops}
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from e4 import ar_qwen, diff_llada, flops as flops_mod, grade  # noqa: E402


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, str(int(default)))
    return raw.lower() in {"1", "true", "yes"}


def load_problems(n: int, dev_indices_path: Path) -> list[dict]:
    """Load the first n problems from the frozen dev set.

    Tries the local pinned indices first; falls back to streaming GSM8K from HF
    if the dataset is reachable. The mock path returns synthetic problems.
    """
    if env_bool("MOCK_MODELS", False):
        return [
            {
                "id": f"mock-{i}",
                "question": f"Mock problem {i}: 2 + {i} = ?",
                "answer": f"{2 + i}",
            }
            for i in range(n)
        ]

    if not dev_indices_path.exists():
        raise FileNotFoundError(
            f"Frozen dev indices missing at {dev_indices_path}. "
            "Run scripts/freeze_gsm8k.py first."
        )

    with dev_indices_path.open() as f:
        spec = json.load(f)

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(spec["dataset"], split=spec["split"])
    return [
        {
            "id": str(i),
            "question": ds[i]["question"],
            "answer": ds[i]["answer"].split("####")[-1].strip(),
        }
        for i in spec["indices"][:n]
    ]


def run_condition(
    problem: dict,
    condition: str,
    k_steps: int,
    ar_model,
    diff_model,
    seed: int,
) -> tuple[str, int]:
    """Returns (predicted_answer, flops_used)."""
    q = problem["question"]
    if condition == "c1":
        cot, used = ar_model.generate_cot_and_answer(q, seed=seed)
        return grade.extract_answer(cot), used
    if condition == "c2":
        text, used = diff_model.denoise_block(prompt=q, k_steps=k_steps, seed=seed)
        return grade.extract_answer(text), used
    if condition == "c3":
        plan, f1 = ar_model.generate_plan(q, max_tokens=32, seed=seed)
        cot, f2 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan, k_steps=k_steps, seed=seed
        )
        ans, f3 = ar_model.finalize_answer(question=q, plan=plan, cot=cot, seed=seed)
        return ans, f1 + f2 + f3
    if condition == "c4":
        plan, f1 = ar_model.generate_plan(q, max_tokens=32, seed=seed)
        cot1, f2 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan, k_steps=k_steps, seed=seed
        )
        extension, f3 = ar_model.extend_cot(question=q, plan=plan, cot=cot1, seed=seed)
        cot2, f4 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan + "\n\nDraft: " + cot1 + extension,
            k_steps=k_steps,
            seed=seed,
        )
        ans, f5 = ar_model.finalize_answer(question=q, plan=plan, cot=cot2, seed=seed)
        return ans, f1 + f2 + f3 + f4 + f5
    raise ValueError(f"Unknown condition: {condition}")


def main() -> int:
    condition = env_str("CONDITION", "c1")
    k_steps = env_int("K_STEPS", 0)
    n_problems = env_int("N_PROBLEMS", 5)
    seed = env_int("SEED", 0)
    ar_model_name = env_str("AR_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    diff_model_name = env_str("DIFF_MODEL", "GSAI-ML/LLaDA-8B-Instruct")
    mock = env_bool("MOCK_MODELS", False)

    random.seed(seed)
    dev_indices = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"
    problems = load_problems(n_problems, dev_indices)

    ar_model = ar_qwen.load(ar_model_name, mock=mock)
    diff_model = diff_llada.load(diff_model_name, mock=mock)

    n_correct = 0
    total_flops = 0
    rows: list[dict] = []
    t0 = time.time()
    for i, prob in enumerate(problems):
        pred, used = run_condition(
            prob, condition, k_steps, ar_model, diff_model, seed=seed
        )
        correct = grade.is_correct(pred, prob["answer"])
        n_correct += int(correct)
        total_flops += used
        rows.append(
            {
                "idx": i,
                "id": prob["id"],
                "condition": condition,
                "k_steps": k_steps,
                "seed": seed,
                "pred": pred,
                "gold": prob["answer"],
                "correct": correct,
                "flops": used,
            }
        )
        running_acc = n_correct / (i + 1)
        running_loss = 1.0 - running_acc
        # Crucible LM-training-contract stdout. val_bpb piggybacks for FLOPs.
        print(
            f"step:{i+1}/{n_problems} "
            f"train_loss:{running_loss:.4f} "
            f"val_loss:{running_loss:.4f} "
            f"val_bpb:{total_flops:.3e}",
            flush=True,
        )

    elapsed = time.time() - t0
    accuracy = n_correct / max(n_problems, 1)
    print(
        f"\n[E4] condition={condition} k={k_steps} seed={seed} "
        f"n={n_problems} accuracy={accuracy:.4f} flops={total_flops:.3e} "
        f"wallclock={elapsed:.1f}s",
        flush=True,
    )

    out_path = (
        REPO_ROOT
        / "e4"
        / "results"
        / f"raw_{condition}_k{k_steps}_seed{seed}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[E4] rows -> {out_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
