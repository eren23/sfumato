"""Build the Track 2 consensus training dataset for E2.

For each GSM8K-train problem, run cmaj b=5 t=0.7 (5 LLaDA-Instruct samples,
majority vote) plus base-greedy LLaDA, and keep rows where majority is
correct AND greedy is wrong — those are the cases where a "commit
adapter" distilled from cmaj has positive teaching signal beyond the base.

Reuses e4.diff_llada (real path) and e4.grade. Resume-safe: pass
--resume_from to skip already-processed source_idx values; raw rows are
appended to e2/data/consensus_raw.jsonl as we go, so a preempted run
loses at most one branch.

Usage (smoke / mock locally, no GPU needed):
    python scripts/build_consensus_dataset.py --n_problems 4 --push False --mock

Real run (on pod, GPU, resumable):
    HF_TOKEN=... python scripts/build_consensus_dataset.py \\
        --n_problems 1500 --max_problems_per_run 200 \\
        --resume_from e2/data/consensus_raw.jsonl --push True
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "e2" / "data"
RAW_PATH = DATA_DIR / "consensus_raw.jsonl"

# Make e4.* importable when running this script directly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _split_gsm8k_answer(answer_field: str) -> tuple[str, str]:
    if "####" in answer_field:
        rationale, numeric = answer_field.rsplit("####", 1)
        return rationale.strip(), numeric.strip()
    return answer_field.strip(), ""


def _load_resume(path: Path) -> dict[int, dict]:
    """Load existing rows keyed by source_idx; tolerate truncated last line."""
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a half-flushed final line
            if "source_idx" in row:
                rows[int(row["source_idx"])] = row
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_problems", type=int, default=1500)
    parser.add_argument("--k_steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--branches", type=int, default=5)
    parser.add_argument("--push", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument(
        "--repo_id", type=str, default="eren23/sfumato-consensus-gsm8k"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_problems_per_run",
        type=int,
        default=0,
        help="If >0, stop after generating rows for this many *new* "
        "problems (preemption-friendly).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=str(RAW_PATH),
        help="Path to a jsonl of previously-emitted rows; their "
        "source_idx values will be skipped.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the diff_llada Mock backend (deterministic strings; no GPU).",
    )
    parser.add_argument(
        "--diff_model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
    )
    args = parser.parse_args()

    from datasets import Dataset, load_dataset  # type: ignore

    from e4 import diff_llada, grade  # type: ignore

    resume_path = Path(args.resume_from)
    existing = _load_resume(resume_path)
    print(f"[consensus] resume: {len(existing)} rows already in {resume_path}")

    print("[consensus] loading gsm8k/main:train", flush=True)
    ds = load_dataset("gsm8k", "main", split="train")
    n = min(args.n_problems, len(ds))

    print(f"[consensus] loading diffusion model: {args.diff_model}", flush=True)
    diff_model = diff_llada.load(args.diff_model, mock=args.mock)

    new_count = 0
    try:
        for i in range(n):
            if i in existing:
                continue
            ex = ds[i]
            question = ex["question"]
            _, gold_answer = _split_gsm8k_answer(ex["answer"])

            # Greedy reference (no temperature) so we know whether branches
            # actually beat the deterministic baseline.
            greedy_text, _ = diff_model.denoise_block(
                prompt=question,
                k_steps=args.k_steps,
                seed=0,
                temperature=0.0,
            )
            greedy_answer = grade.extract_answer(greedy_text)

            branches: list[str] = []
            branch_answers: list[str] = []
            for b in range(args.branches):
                cot, _ = diff_model.denoise_block(
                    prompt=question,
                    k_steps=args.k_steps,
                    seed=args.seed * 100 + b,
                    temperature=args.temperature,
                )
                branches.append(cot)
                branch_answers.append(grade.extract_answer(cot))

            counts = Counter(a for a in branch_answers if a)
            if counts:
                majority_answer = counts.most_common(1)[0][0]
            else:
                majority_answer = branch_answers[0] or ""

            try:
                consensus_correct = bool(majority_answer) and (
                    float(majority_answer) == float(gold_answer)
                )
            except ValueError:
                consensus_correct = (
                    majority_answer.strip() == gold_answer.strip()
                )

            row = {
                "question": question,
                "gold_answer": gold_answer,
                "greedy_output": greedy_text,
                "greedy_answer": greedy_answer,
                "branches": branches,
                "branch_answers": branch_answers,
                "majority_answer": majority_answer,
                "consensus_correct": consensus_correct,
                "source_idx": i,
            }
            _append_jsonl(resume_path, row)
            existing[i] = row
            new_count += 1

            if new_count % 10 == 0:
                kept = sum(
                    1
                    for r in existing.values()
                    if r.get("consensus_correct")
                    and r.get("greedy_answer") != r.get("gold_answer")
                )
                print(
                    f"[consensus] processed={new_count} "
                    f"total={len(existing)} kept={kept}",
                    flush=True,
                )

            if (
                args.max_problems_per_run
                and new_count >= args.max_problems_per_run
            ):
                print(
                    f"[consensus] hit --max_problems_per_run={args.max_problems_per_run}; "
                    "stopping run for resumability",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print("[consensus] interrupted; jsonl is preserved", flush=True)

    # Filter to commit-adapter-positive rows.
    keep_rows = []
    for r in existing.values():
        if not r.get("consensus_correct"):
            continue
        if r.get("greedy_answer") == r.get("gold_answer"):
            continue
        keep_rows.append(
            {
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "greedy_output": r["greedy_output"],
                "branches": r["branches"],
                "branch_answers": r["branch_answers"],
                "majority_answer": r["majority_answer"],
                "consensus_correct": True,
                "source_idx": r["source_idx"],
            }
        )

    print(
        f"[consensus] kept {len(keep_rows)} rows "
        f"(majority correct AND greedy wrong) out of {len(existing)} processed",
        flush=True,
    )

    if not keep_rows:
        print("[consensus] no rows to emit; raw jsonl preserved on disk")
        return 0

    dataset = Dataset.from_list(keep_rows)

    if args.push:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            print(
                "[consensus] no HF token in env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN); cannot push",
                file=sys.stderr,
            )
            return 1
        print(f"[consensus] pushing to hub: {args.repo_id}", flush=True)
        dataset.push_to_hub(args.repo_id, token=token)
        print(f"[consensus] pushed {args.repo_id}", flush=True)
    else:
        out_path = DATA_DIR / "consensus_filtered.parquet"
        dataset.to_parquet(str(out_path))
        print(f"[consensus] wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
