"""Strong-judge sanity check via OpenRouter.

For each (problem, branch) in the rich substrate:
  - Send (problem_text, candidate_solution) to a frontier judge via OpenRouter
  - Ask: "Is the final numeric answer correct? Reply only YES or NO."
  - Aggregate per problem: pick branch with highest YES count (ties: take
    branch whose extracted answer is the cmaj pick).
  - Compare verifier acc vs cmaj.

This tests whether the gap is closable IN PRINCIPLE with a strictly stronger
judge than the LLaDA-8B that produced the substrate.

Reads:
  - rich_substrate_n500.jsonl (RICH_PATH env override)
  - e4/data/gsm8k_dev_500.json (for problem text — substrate doesn't store it)
  - OPENROUTER_API_KEY from .env

Writes one JSON per judge:
  phase2/spikes/strong-judge/results_<judge_tag>.json
matching the RANKING.md schema for direct comparison.

Cost: ~1000 branches * 3 judges * ~500 tok in / 5 tok out.
At openrouter prices (~$2-5 per million in tok for these tiers), total ~$1-3.

Usage:
  OPENROUTER_API_KEY=sk-or-... python phase2/spikes/strong-judge/judge_via_openrouter.py
  # or with --judges to limit:
  ... judge_via_openrouter.py --judges anthropic/claude-sonnet-4.5
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RICH = REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate_n500.jsonl"
if not DEFAULT_RICH.exists():
    DEFAULT_RICH = pathlib.Path("/tmp/rich_substrate_n500.jsonl")
RICH_PATH = pathlib.Path(os.environ.get("RICH_PATH", str(DEFAULT_RICH)))
DEV_INDICES_PATH = REPO_ROOT / "e4/data/gsm8k_dev_500.json"
OUT_DIR = REPO_ROOT / "phase2/spikes/strong-judge"

DEFAULT_JUDGES = [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5",
    "google/gemini-2.5-pro",
]

JUDGE_PROMPT = """Below is a math word problem and a candidate solution.
Decide if the candidate's FINAL numeric answer is exactly correct.

Problem:
{problem}

Candidate solution:
{solution}

Reply with EXACTLY one token: YES (the final answer is correct) or NO (it is wrong).
"""


def _safe_tag(judge: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", judge).strip("-")


def load_problem_texts() -> dict[str, str]:
    """Map problem_id (str) -> question text by pulling from HF GSM8K test."""
    spec = json.loads(DEV_INDICES_PATH.read_text())
    from datasets import load_dataset
    ds = load_dataset(spec["dataset"], spec.get("config", "main"), split=spec["split"])
    return {str(idx): ds[idx]["question"] for idx in spec["indices"]}


def call_judge(client, judge: str, problem: str, solution: str, max_tokens: int = 4) -> str:
    """Returns 'YES' or 'NO' (best-effort)."""
    resp = client.chat.completions.create(
        model=judge,
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(problem=problem, solution=solution)},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip().upper()
    if "YES" in txt[:5]:
        return "YES"
    if "NO" in txt[:4]:
        return "NO"
    return "?"


def judge_one(rows, judge: str, client, problem_texts: dict[str, str], max_branches: int | None):
    """Score every branch with this judge; pick argmax-yes-count per problem."""
    by_pid = defaultdict(list)
    for i, r in enumerate(rows):
        by_pid[r["problem_id"]].append(i)

    yes_count = {i: 0 for i in range(len(rows))}  # 1 if judge says YES, else 0
    n_calls = 0
    n_yes = 0
    n_err = 0
    t0 = time.time()
    for pid, idxs in by_pid.items():
        problem = problem_texts.get(pid, "(problem text unavailable)")
        for i in idxs[:max_branches] if max_branches else idxs:
            try:
                v = call_judge(client, judge, problem, rows[i].get("branch_text", ""))
                yes_count[i] = 1 if v == "YES" else 0
                if v == "YES": n_yes += 1
                if v == "?": n_err += 1
            except Exception as e:
                n_err += 1
                print(f"  [{judge}] err pid={pid} i={i}: {e}", flush=True)
            n_calls += 1
            if n_calls % 25 == 0:
                el = time.time() - t0
                print(f"  [{judge}] {n_calls}/{len(rows)} ({el:.0f}s, {n_yes} YES, {n_err} err)",
                      flush=True)
    return yes_count, {"n_calls": n_calls, "n_yes": n_yes, "n_err": n_err,
                       "wall_s": time.time() - t0}


def aggregate(rows, yes_count):
    """Per problem: pick branch with most YES votes; tiebreak by extracted-cmaj."""
    by_pid = defaultdict(list)
    for i, r in enumerate(rows):
        by_pid[r["problem_id"]].append(i)
    cmaj_correct = ver_correct = oracle_correct = recovers = 0
    n = 0
    per_problem = []
    for pid, idxs in by_pid.items():
        n += 1
        items = [rows[i] for i in idxs]
        gold = items[0]["gold"]
        votes = [r["extracted"] for r in items]
        cmaj_pick = Counter(votes).most_common(1)[0][0]
        cmaj_hit = (cmaj_pick == gold)
        if cmaj_hit: cmaj_correct += 1
        if any(v == gold for v in votes): oracle_correct += 1
        scored = [(yes_count.get(idxs[k], 0), votes[k], k) for k in range(len(items))]
        # Tiebreak: prefer branches whose extracted matches cmaj_pick
        scored.sort(key=lambda x: (x[0], int(x[1] == cmaj_pick)), reverse=True)
        ver_pick = items[scored[0][2]]["extracted"]
        if ver_pick == gold: ver_correct += 1
        if not cmaj_hit and ver_pick == gold: recovers += 1
        per_problem.append({
            "problem_id": pid, "gold": gold, "cmaj_pick": cmaj_pick,
            "verifier_pick": ver_pick, "yes_per_branch": [yes_count.get(idxs[k], 0) for k in range(len(items))],
            "votes": votes, "cmaj_hit": cmaj_hit, "verifier_hit": ver_pick == gold,
        })
    dpp = (ver_correct - cmaj_correct) / n * 100
    if (ver_correct/n) >= 0.89: dec = "WIN-DECISIVE"
    elif (ver_correct/n) >= 0.87: dec = "WIN-STRONG"
    elif (ver_correct/n) >= 0.83: dec = "WIN-MINOR"
    elif (ver_correct/n) >= (cmaj_correct/n) - 0.01: dec = "INCONCLUSIVE"
    else: dec = "LOSS"
    return {
        "n_problems": n,
        "mean_cmaj": cmaj_correct/n,
        "mean_verifier": ver_correct/n,
        "mean_oracle": oracle_correct/n,
        "delta_pp_vs_cmaj": dpp,
        "verifier_recovers": recovers,
        "decision": dec,
        "by_problem": per_problem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--max-branches-per-problem", type=int, default=None,
                    help="Cap calls per problem (e.g. 5 for cmaj b=5; None = use all)")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        # Try .env file
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(args.api_key_env + "="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        print(f"FATAL: {args.api_key_env} not set in env or .env", file=sys.stderr)
        sys.exit(1)

    if not RICH_PATH.exists():
        print(f"FATAL: substrate {RICH_PATH} missing"); sys.exit(1)
    rows = [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]
    print(f"[judge] loaded {len(rows)} branches from {RICH_PATH.name}", flush=True)

    print(f"[judge] loading problem texts from {DEV_INDICES_PATH.name}", flush=True)
    problem_texts = load_problem_texts()
    print(f"[judge] {len(problem_texts)} problem texts loaded", flush=True)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for judge in args.judges:
        tag = _safe_tag(judge)
        out_path = OUT_DIR / f"results_{tag}.json"
        if out_path.exists():
            print(f"[judge] skip {tag} (exists: {out_path})"); continue
        print(f"\n[judge] === {judge} ===", flush=True)
        yes_count, meta = judge_one(rows, judge, client, problem_texts,
                                     args.max_branches_per_problem)
        agg = aggregate(rows, yes_count)
        result = {
            "model_id": judge,
            "judge_tag": tag,
            "rich_path": str(RICH_PATH),
            "n_branches": len(rows),
            "mean_cmaj": agg["mean_cmaj"],
            "mean_verifier": agg["mean_verifier"],
            "mean_oracle": agg["mean_oracle"],
            "delta_pp_vs_cmaj": agg["delta_pp_vs_cmaj"],
            "verifier_recovers": agg["verifier_recovers"],
            "decision": agg["decision"],
            "n_problems": agg["n_problems"],
            "openrouter_meta": meta,
            "by_problem": agg["by_problem"][:30],  # cap to keep file small
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[judge] {tag}: cmaj={agg['mean_cmaj']:.1%} verifier={agg['mean_verifier']:.1%} "
              f"oracle={agg['mean_oracle']:.1%} dpp={agg['delta_pp_vs_cmaj']:+.2f} {agg['decision']}")
        print(f"[judge] wrote {out_path}")


if __name__ == "__main__":
    main()
