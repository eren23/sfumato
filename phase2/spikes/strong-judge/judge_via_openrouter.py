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

JUDGE_PROMPT_COT = """Below is a math word problem and a candidate solution.
Your job is to decide if the candidate's FINAL numeric answer is exactly correct.

Problem:
{problem}

Candidate solution:
{solution}

First, in 1-2 sentences, work out the correct answer yourself.
Then on the LAST line of your response write exactly one of:
VERDICT: YES
VERDICT: NO
"""


def _safe_tag(judge: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", judge).strip("-")


def load_problem_texts() -> dict[str, str]:
    """Map problem_id (str) -> question text by pulling from HF GSM8K test."""
    spec = json.loads(DEV_INDICES_PATH.read_text())
    from datasets import load_dataset
    ds = load_dataset(spec["dataset"], spec.get("config", "main"), split=spec["split"])
    return {str(idx): ds[idx]["question"] for idx in spec["indices"]}


def call_judge(client, judge: str, problem: str, solution: str,
               max_tokens: int = 32, timeout: float = 45.0,
               prompt_template: str = JUDGE_PROMPT) -> str:
    """Returns 'YES' or 'NO' (best-effort). Per-call timeout guards against hangs."""
    resp = client.with_options(timeout=timeout).chat.completions.create(
        model=judge,
        messages=[
            {"role": "user", "content": prompt_template.format(problem=problem, solution=solution)},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    txt = (resp.choices[0].message.content or "").strip().upper()
    # For CoT variant, look at last line for "VERDICT: YES/NO"
    last = txt.splitlines()[-1] if txt.splitlines() else txt
    if "VERDICT: YES" in last or "VERDICT:YES" in last: return "YES"
    if "VERDICT: NO" in last or "VERDICT:NO" in last: return "NO"
    # Fallback: simple YES/NO anywhere
    if "YES" in txt and "NO" not in txt[:txt.find("YES") if "YES" in txt else 0]:
        return "YES"
    if "NO" in txt:
        return "NO"
    if "YES" in txt:
        return "YES"
    return "?"


def judge_one(rows, judge: str, client, problem_texts: dict[str, str], max_branches: int | None,
              concurrency: int = 8, only_indices: set[int] | None = None,
              prior_yes: dict[int, int] | None = None,
              prior_status: dict[int, str] | None = None):
    """Score every branch with this judge; pick argmax-yes-count per problem.

    Concurrent via ThreadPoolExecutor so we don't get killed by one slow call.
    If `only_indices` is set, only judge those branches; carry prior YES from
    prior_yes for the rest.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    by_pid = defaultdict(list)
    for i, r in enumerate(rows):
        by_pid[r["problem_id"]].append(i)

    tasks = []
    for pid, idxs in by_pid.items():
        problem = problem_texts.get(pid, "(problem text unavailable)")
        for i in idxs[:max_branches] if max_branches else idxs:
            if only_indices is not None and i not in only_indices:
                continue
            tasks.append((i, pid, problem, rows[i].get("branch_text", "")))

    yes_count = dict(prior_yes) if prior_yes else {i: 0 for i in range(len(rows))}
    branch_status = dict(prior_status) if prior_status else {i: "unjudged" for i in range(len(rows))}
    n_yes = sum(1 for v in yes_count.values() if v == 1)
    n_err = sum(1 for s in branch_status.values() if s == "?")
    n_done = 0
    t0 = time.time()

    def _one(task):
        i, pid, problem, sol = task
        try:
            v = call_judge(client, judge, problem, sol,
                           max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "32")),
                           prompt_template=(JUDGE_PROMPT_COT if os.environ.get("JUDGE_COT") else JUDGE_PROMPT))
            return (i, v, None)
        except Exception as e:
            return (i, "?", str(e)[:100])

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tasks]):
            i, v, err = fut.result()
            # If retrying and we got an error this time, keep prior status
            prev = branch_status.get(i, "unjudged")
            if v == "?":
                # On retry, only overwrite YES/NO with err if no prior good answer
                if prev not in ("YES", "NO"):
                    branch_status[i] = "?"
                    if prev != "?": n_err += 1
            else:
                if prev == "?": n_err -= 1  # recovered an error
                branch_status[i] = v
                yes_count[i] = 1 if v == "YES" else 0
                if v == "YES" and prev != "YES": n_yes += 1
                if v != "YES" and prev == "YES": n_yes -= 1
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"  [{judge}] {n_done}/{len(tasks)} ({el:.0f}s, {n_yes} YES, {n_err} err)",
                      flush=True)
    return yes_count, branch_status, {"n_calls": n_done, "n_yes": n_yes, "n_err": n_err,
                       "wall_s": time.time() - t0, "concurrency": concurrency}


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
            "verifier_pick": ver_pick,
            "branch_indices": [int(j) for j in idxs],
            "yes_per_branch": [yes_count.get(idxs[k], 0) for k in range(len(items))],
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
    ap.add_argument("--api-key-env", default=None,
                    help="env var (default tries OPEN_ROUTER_API_KEY then OPENROUTER_API_KEY)")
    ap.add_argument("--retry-errors-from", default=None,
                    help="path to existing results JSON; only re-judge branches that errored")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    candidates = [args.api_key_env] if args.api_key_env else \
                 ["OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"]
    api_key = None
    chosen_key = None
    # Try env vars first
    for k in candidates:
        if k and os.environ.get(k):
            api_key = os.environ[k]; chosen_key = k; break
    # Then .env
    if not api_key:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                for k in candidates:
                    if k and line.strip().startswith(k + "="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        chosen_key = k
                        break
                if api_key: break
    if not api_key:
        print(f"FATAL: none of {candidates} set in env or .env", file=sys.stderr)
        sys.exit(1)
    print(f"[judge] using {chosen_key}", flush=True)

    if not RICH_PATH.exists():
        print(f"FATAL: substrate {RICH_PATH} missing"); sys.exit(1)
    rows = [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]
    print(f"[judge] loaded {len(rows)} branches from {RICH_PATH.name}", flush=True)

    print(f"[judge] loading problem texts from {DEV_INDICES_PATH.name}", flush=True)
    problem_texts = load_problem_texts()
    print(f"[judge] {len(problem_texts)} problem texts loaded", flush=True)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
                    timeout=30.0, max_retries=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Optional: load prior result to retry only errors
    prior_yes = None
    prior_status = None
    only_indices = None
    if args.retry_errors_from:
        prior = json.loads(pathlib.Path(args.retry_errors_from).read_text())
        prior_status = {}
        prior_yes = {}
        only_indices = set()
        bs = prior.get("branch_status", {})
        if not bs:
            print("[judge] WARN: prior has no branch_status; can't isolate errors", flush=True)
            prior_yes = None; prior_status = None; only_indices = None
        else:
            for k, v in bs.items():
                bi = int(k)
                prior_status[bi] = v
                prior_yes[bi] = 1 if v == "YES" else 0
                if v == "?":
                    only_indices.add(bi)
            print(f"[judge] resume from {args.retry_errors_from}: "
                  f"{len(only_indices)} errored branches to retry "
                  f"(prior had {sum(1 for v in bs.values() if v=='YES')} YES, "
                  f"{sum(1 for v in bs.values() if v=='NO')} NO, "
                  f"{sum(1 for v in bs.values() if v=='?')} ?)", flush=True)

    for judge in args.judges:
        tag = _safe_tag(judge)
        out_path = OUT_DIR / f"results_{tag}.json"
        if out_path.exists() and not args.retry_errors_from:
            print(f"[judge] skip {tag} (exists: {out_path})"); continue
        print(f"\n[judge] === {judge} ===", flush=True)
        yes_count, branch_status, meta = judge_one(
            rows, judge, client, problem_texts,
            args.max_branches_per_problem,
            concurrency=args.concurrency,
            only_indices=only_indices,
            prior_yes=prior_yes,
            prior_status=prior_status,
        )
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
            "by_problem": agg["by_problem"],  # full per-problem detail (needed for retry-only-errors)
            "branch_status": {str(k): v for k, v in branch_status.items()},
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[judge] {tag}: cmaj={agg['mean_cmaj']:.1%} verifier={agg['mean_verifier']:.1%} "
              f"oracle={agg['mean_oracle']:.1%} dpp={agg['delta_pp_vs_cmaj']:+.2f} {agg['decision']}")
        print(f"[judge] wrote {out_path}")


if __name__ == "__main__":
    main()
