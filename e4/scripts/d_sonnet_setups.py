"""D step 1 — Sonnet 4.5 problem-setup extraction.

For each problem in e4/data/gsm8k_dev_200.json, call Sonnet 4.5 via OpenRouter
with a setup-extraction prompt that produces a structured restatement
(givens, unknown, intermediate-quantity-names) WITHOUT solving the problem.

The structured restatement is then used as the prompt prefix for LLaDA in
step 2 (GPU run). The hypothesis: if "LLaDA's GSM8K failure mode is problem
comprehension, not arithmetic" (paper 1), scaffolding the comprehension step
with Sonnet should lift accuracy materially.

Reads:
  - e4/data/gsm8k_dev_200.json (frozen indices)
  - HF gsm8k dataset (cached locally)
  - OPENROUTER_API_KEY from env or .env

Writes:
  - e4/results/comprehension_scaffold/sonnet_setups.jsonl
  - e4/results/comprehension_scaffold/sonnet_meta.json

Usage:
  N_PROBLEMS=5 python e4/scripts/d_sonnet_setups.py          # smoke test
  python e4/scripts/d_sonnet_setups.py                       # full N=200
  N_PROBLEMS=200 RESUME=1 python e4/scripts/d_sonnet_setups.py  # resume from checkpoint
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV_INDICES_PATH = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"
OUT_DIR = REPO_ROOT / "e4" / "results" / "comprehension_scaffold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "sonnet_setups.jsonl"
OUT_META = OUT_DIR / "sonnet_meta.json"

JUDGE_MODEL = "anthropic/claude-sonnet-4.5"

SETUP_PROMPT = """You will receive a GSM8K math word problem. Your job is to restate it in a structured form WITHOUT solving it.

Output EXACTLY this format (no commentary, no numbers other than the givens):

GIVENS:
- <each numerical or factual given, one per line, named clearly>

UNKNOWN:
- <the single quantity the question asks for>

INTERMEDIATE QUANTITIES (in solving order):
- <name>: <one-sentence English description, no formulas, no numeric answer>
- <name>: ...

DO NOT compute any answers. DO NOT add numbers that weren't in the original problem. DO NOT solve. Just restate the structure.

Problem:
{problem}
"""


def load_problems() -> list[dict]:
    spec = json.loads(DEV_INDICES_PATH.read_text())
    from datasets import load_dataset
    ds = load_dataset(spec["dataset"], spec.get("config", "main"), split=spec["split"])
    out = []
    for idx in spec["indices"]:
        row = ds[idx]
        # GSM8K answer field is "#### N" at end; strip the explanation
        ans = row["answer"]
        gold = ans.split("####")[-1].strip().replace(",", "") if "####" in ans else ans.strip()
        out.append({"idx": idx, "question": row["question"], "gold": gold})
    return out


def get_api_key() -> str:
    for k in ("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            for k in ("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"):
                if line.strip().startswith(k + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found in env or .env")


def call_sonnet(client, problem: str, max_tokens: int = 600, timeout: float = 45.0) -> str:
    resp = client.with_options(timeout=timeout).chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": SETUP_PROMPT.format(problem=problem)}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def load_existing() -> dict[int, dict]:
    existing = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    existing[r["idx"]] = r
                except Exception:
                    continue
    return existing


def main() -> None:
    n_problems = int(os.environ.get("N_PROBLEMS", "200"))
    concurrency = int(os.environ.get("CONCURRENCY", "6"))
    resume = os.environ.get("RESUME", "0") == "1"

    api_key = get_api_key()
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=45.0,
        max_retries=2,
    )

    problems = load_problems()[:n_problems]
    print(f"loaded {len(problems)} problems")

    existing = load_existing() if resume else {}
    if existing:
        print(f"resume: found {len(existing)} existing setups; skipping those")
    problems_to_call = [p for p in problems if p["idx"] not in existing]
    print(f"calling Sonnet for {len(problems_to_call)} problems")

    if not problems_to_call:
        print("nothing to do; all setups present")
        return

    t0 = time.time()
    results = list(existing.values())
    n_err = 0

    def _one(p):
        try:
            setup = call_sonnet(client, p["question"])
            return {"idx": p["idx"], "question": p["question"], "gold": p["gold"], "sonnet_setup": setup, "model": JUDGE_MODEL, "error": None}
        except Exception as e:
            return {"idx": p["idx"], "question": p["question"], "gold": p["gold"], "sonnet_setup": None, "model": JUDGE_MODEL, "error": str(e)[:200]}

    n_done = 0
    fh = open(OUT_JSONL, "a")
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for fut in as_completed([ex.submit(_one, p) for p in problems_to_call]):
                r = fut.result()
                if r["error"]:
                    n_err += 1
                results.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(problems_to_call):
                    el = time.time() - t0
                    rate = n_done / el if el > 0 else 0
                    eta = (len(problems_to_call) - n_done) / rate if rate > 0 else 0
                    print(f"  [{n_done}/{len(problems_to_call)}] elapsed {el:.0f}s rate {rate:.2f}/s eta {eta:.0f}s errors {n_err}", flush=True)
    finally:
        fh.close()

    # Sort by idx and rewrite jsonl atomically
    results.sort(key=lambda r: r["idx"])
    OUT_JSONL.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    n_ok = sum(1 for r in results if r["sonnet_setup"])
    meta = {
        "model": JUDGE_MODEL,
        "n_problems": len(problems),
        "n_ok": n_ok,
        "n_error": len(results) - n_ok,
        "wall_s": round(time.time() - t0, 2),
        "concurrency": concurrency,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print()
    print(f"done: {n_ok}/{len(results)} setups extracted ({len(results) - n_ok} errors)")
    print(f"wrote {OUT_JSONL}")
    print(f"wrote {OUT_META}")

    if n_ok > 0:
        sample = [r for r in results if r["sonnet_setup"]][0]
        print()
        print(f"-- sample (idx={sample['idx']}, gold={sample['gold']}) --")
        print("question:")
        print(sample["question"])
        print()
        print("sonnet_setup:")
        print(sample["sonnet_setup"])


if __name__ == "__main__":
    main()
