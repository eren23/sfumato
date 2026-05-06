"""Phase A pilot: frontier-model GSM8K solve via OpenRouter (50 problems x 3 models).

Asks frontier models (GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro) to solve GSM8K
problems zero-shot CoT and writes per-problem JSONL outputs keyed by HF idx for
join compatibility with the sfumato cmajc-v3 substrate.

Reuses:
  - .env / env-var loading pattern from phase2/spikes/strong-judge/judge_via_openrouter.py
  - e4/grade.py:extract_final_answer for strict answer extraction

Usage:
  python phase2/frontier_compare/run_compare.py --n 50

Output:
  phase2/frontier_compare/results_50/{model_tag}.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
import time
from typing import Optional

import aiohttp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from e4.grade import extract_final_answer  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Map of CLI-friendly aliases to OpenRouter model IDs and short tags.
MODEL_ALIASES = {
    # Frontier tier
    "gpt-4o": ("openai/gpt-4o", "gpt4o"),
    "claude-sonnet-4.5": ("anthropic/claude-sonnet-4.5", "claude45"),
    "gemini-2.5-pro": ("google/gemini-2.5-pro", "gemini25"),
    # OSS tier (fair-fight)
    "deepseek-chat": ("deepseek/deepseek-chat", "deepseek_chat"),
    "qwen-2.5-72b": ("qwen/qwen-2.5-72b-instruct", "qwen25_72b"),
    "llama-3.3-70b": ("meta-llama/llama-3.3-70b-instruct", "llama33_70b"),
}

# Map of OpenRouter model IDs (as passed via --models) to canonical short tags.
# Used so users can pass either alias or full id and still get stable filenames.
MODEL_ID_TO_TAG = {
    "openai/gpt-4o": "gpt4o",
    "anthropic/claude-sonnet-4.5": "claude45",
    "google/gemini-2.5-pro": "gemini25",
    "deepseek/deepseek-chat": "deepseek_chat",
    "qwen/qwen-2.5-72b-instruct": "qwen25_72b",
    "meta-llama/llama-3.3-70b-instruct": "llama33_70b",
    # Phase A.2 — small + mid tier
    "meta-llama/llama-3.1-8b-instruct": "llama31_8b",
    "qwen/qwen-2.5-7b-instruct": "qwen25_7b",
    "mistralai/mistral-7b-instruct-v0.1": "mistral_7b_v01",
    "qwen/qwen3-30b-a3b-instruct-2507": "qwen3_30b_a3b",
}

# Conservative pricing snapshot for cost estimates only (USD per 1M tokens).
# Source: OpenRouter list prices, ballpark; only used for cost-cap accounting.
PRICE_PER_M = {
    # Frontier
    "openai/gpt-4o": (2.5, 10.0),                # in, out
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "google/gemini-2.5-pro": (1.25, 10.0),
    # OSS large (per OpenRouter list prices)
    "deepseek/deepseek-chat": (0.27, 1.10),
    "qwen/qwen-2.5-72b-instruct": (0.13, 0.40),
    "meta-llama/llama-3.3-70b-instruct": (0.13, 0.40),
    # Small (~7-8B, sfumato's weight class) — Phase A.2 extension
    "meta-llama/llama-3.1-8b-instruct": (0.05, 0.08),
    "qwen/qwen-2.5-7b-instruct": (0.10, 0.20),
    "mistralai/mistral-7b-instruct-v0.1": (0.11, 0.19),
    # Mid (~30B) — Phase A.2 extension (qwen3 30B-A3B MoE; substituted for
    # missing qwen-2.5-32b-instruct slug)
    "qwen/qwen3-30b-a3b-instruct-2507": (0.09, 0.30),
}

PROMPT_TMPL = (
    "Solve the following grade-school math problem. Show your reasoning step by "
    "step. End your response with the final numeric answer in exactly this "
    "format on its own line: `#### N` (replace N with the number).\n\n"
    "Problem: {question}"
)


def _load_api_key() -> Optional[str]:
    for k in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            for k in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
                if line.strip().startswith(k + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _safe_tag(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")


def _gold_int(answer: str) -> str:
    """GSM8K gold answer string -> the post-#### integer string."""
    if "####" in answer:
        return answer.split("####", 1)[1].strip().replace(",", "")
    return answer.strip()


def _correct(pred: str, gold: str) -> Optional[bool]:
    if pred is None or pred == "":
        return None
    try:
        return float(pred.replace(",", "")) == float(gold.replace(",", ""))
    except ValueError:
        return pred.strip() == gold.strip()


def _load_existing(path: pathlib.Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            done.add(int(r["idx"]))
        except Exception:
            continue
    return done


async def _call_openrouter(session: aiohttp.ClientSession, model: str, question: str,
                           api_key: str, timeout: float = 180.0):
    """Single completion call. Returns dict: {raw, latency_ms, usage, error}."""
    # Reasoning models (Gemini 2.5 Pro etc.) burn most of `max_tokens` on hidden
    # reasoning_tokens, leaving the visible answer truncated. Bump the cap for
    # the known reasoning models so they have room to finish with #### N.
    if "gemini-2.5-pro" in model:
        max_tokens = 4096
    else:
        max_tokens = 1024
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TMPL.format(question=question)}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/eren23/sfumato",
        "X-Title": "sfumato-frontier-compare",
    }
    t0 = time.time()
    try:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            text = await resp.text()
            latency_ms = int((time.time() - t0) * 1000)
            if resp.status != 200:
                return {"raw": "", "latency_ms": latency_ms,
                        "usage": {}, "error": f"HTTP {resp.status}: {text[:200]}"}
            data = json.loads(text)
            try:
                content = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                return {"raw": "", "latency_ms": latency_ms,
                        "usage": {}, "error": f"bad response: {str(e)[:80]} :: {text[:200]}"}
            usage = data.get("usage", {}) or {}
            return {"raw": content, "latency_ms": latency_ms,
                    "usage": usage, "error": None}
    except asyncio.TimeoutError:
        return {"raw": "", "latency_ms": int((time.time() - t0) * 1000),
                "usage": {}, "error": "timeout"}
    except Exception as e:
        return {"raw": "", "latency_ms": int((time.time() - t0) * 1000),
                "usage": {}, "error": f"{type(e).__name__}: {str(e)[:160]}"}


def _estimate_cost(model: str, usage: dict) -> float:
    pin, pout = PRICE_PER_M.get(model, (0.0, 0.0))
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    return (pt / 1e6) * pin + (ct / 1e6) * pout


async def _solve_one(sem, session, model, question, gold, idx, api_key,
                     max_attempts=2):
    last_err = None
    last_latency = 0
    last_usage: dict = {}
    last_raw = ""
    attempts = 0
    async with sem:
        for a in range(max_attempts):
            attempts = a + 1
            r = await _call_openrouter(session, model, question, api_key)
            last_latency = r["latency_ms"]
            last_usage = r["usage"]
            last_raw = r["raw"]
            if r["error"] is None and r["raw"].strip():
                pred = extract_final_answer(r["raw"])
                # Fallback: try last numeric token if no #### marker.
                if pred == "":
                    from e4.grade import extract_answer
                    pred = extract_answer(r["raw"])
                cor = _correct(pred, gold)
                return {
                    "idx": idx, "model": model, "pred": pred or None,
                    "correct": cor,
                    "raw_response": r["raw"],
                    "latency_ms": r["latency_ms"],
                    "attempts": attempts,
                    "usage": r["usage"],
                    "cost_estimate_usd": _estimate_cost(model, r["usage"]),
                    "error": None,
                }
            last_err = r["error"]
            await asyncio.sleep(1.0 + a * 1.5)
    return {
        "idx": idx, "model": model, "pred": None, "correct": None,
        "raw_response": last_raw, "latency_ms": last_latency,
        "attempts": attempts, "usage": last_usage,
        "cost_estimate_usd": _estimate_cost(model, last_usage),
        "error": last_err,
    }


async def run_for_model(model_id: str, model_tag: str, problems: list[dict],
                        out_path: pathlib.Path, api_key: str, concurrency: int,
                        budget_remaining: list[float]):
    done = _load_existing(out_path)
    todo = [p for p in problems if p["idx"] not in done]
    if not todo:
        print(f"[{model_tag}] all {len(problems)} already done; skipping", flush=True)
        return 0.0
    sem = asyncio.Semaphore(concurrency)
    spent = 0.0
    n_done = 0
    n_ok = 0
    n_correct = 0
    async with aiohttp.ClientSession() as session:
        coros = [_solve_one(sem, session, model_id, p["question"],
                            _gold_int(p["answer"]), p["idx"], api_key)
                 for p in todo]
        with out_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                # Reformat row to canonical schema.
                gold = next(_gold_int(p["answer"]) for p in todo if p["idx"] == row["idx"])
                question = next(p["question"] for p in todo if p["idx"] == row["idx"])
                out_row = {
                    "idx": row["idx"],
                    "question": question,
                    "gold": gold,
                    "pred": row["pred"],
                    "correct": row["correct"],
                    "model": model_id,
                    "raw_response": row["raw_response"],
                    "latency_ms": row["latency_ms"],
                    "attempts": row["attempts"],
                    "usage": row["usage"],
                    "cost_estimate_usd": row["cost_estimate_usd"],
                    "error": row["error"],
                }
                f.write(json.dumps(out_row) + "\n")
                f.flush()
                spent += row["cost_estimate_usd"]
                n_done += 1
                if row["error"] is None:
                    n_ok += 1
                if row["correct"]:
                    n_correct += 1
                if n_done % 10 == 0:
                    print(f"[{model_tag}] {n_done}/{len(todo)} ok={n_ok} "
                          f"correct={n_correct} ${spent:.4f} cum-budget=${budget_remaining[0]-spent:.4f}",
                          flush=True)
                if (budget_remaining[0] - spent) < 0:
                    print(f"[{model_tag}] BUDGET CAP HIT — stopping", flush=True)
                    break
    budget_remaining[0] -= spent
    print(f"[{model_tag}] DONE: {n_done} new rows, ok={n_ok}, correct={n_correct}, "
          f"spent=${spent:.4f}", flush=True)
    return spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of problems (idx 0..n-1)")
    ap.add_argument("--models", default="gpt-4o,claude-sonnet-4.5,gemini-2.5-pro")
    ap.add_argument("--out-dir", default="phase2/frontier_compare/results_50")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=1.00,
                    help="hard cap on total estimated OpenRouter spend")
    args = ap.parse_args()

    api_key = _load_api_key()
    if not api_key:
        miss = REPO_ROOT / "phase2/frontier_compare/MISSING_API_KEY.md"
        miss.parent.mkdir(parents=True, exist_ok=True)
        miss.write_text("OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY) not found in "
                         ".env or shell env. Aborting Phase A pilot.\n")
        print(f"FATAL: no API key; wrote {miss}", file=sys.stderr)
        sys.exit(1)

    out_dir = REPO_ROOT / args.out_dir if not pathlib.Path(args.out_dir).is_absolute() \
        else pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_compare] loading GSM8K test split via datasets...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    problems = [{"idx": i, "question": ds[i]["question"], "answer": ds[i]["answer"]}
                for i in range(args.n)]
    print(f"[run_compare] {len(problems)} problems loaded (idx 0..{args.n-1})", flush=True)

    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    resolved = []
    for a in aliases:
        if a in MODEL_ALIASES:
            mid, tag = MODEL_ALIASES[a]
        elif a in MODEL_ID_TO_TAG:
            mid = a
            tag = MODEL_ID_TO_TAG[a]
        else:
            mid = a
            tag = _safe_tag(a)
        resolved.append((mid, tag))
    print(f"[run_compare] models: {[t for _,t in resolved]}", flush=True)

    budget_remaining = [args.budget_usd]

    async def go():
        for mid, tag in resolved:
            if budget_remaining[0] <= 0:
                print(f"[run_compare] budget exhausted; skip {tag}", flush=True)
                continue
            out_path = out_dir / f"{tag}.jsonl"
            print(f"\n[run_compare] === {mid} -> {out_path.name} (budget left ${budget_remaining[0]:.4f}) ===",
                  flush=True)
            await run_for_model(mid, tag, problems, out_path, api_key,
                                args.concurrency, budget_remaining)

    asyncio.run(go())
    print(f"\n[run_compare] all done; remaining budget ${budget_remaining[0]:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
