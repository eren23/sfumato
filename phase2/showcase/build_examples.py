"""Build the sfumato showcase example bank.

Reads `e4/results/raw_{cmaj,cmajc}_*v3LoRA_N{200,1319,...}.jsonl`, joins
question text from GSM8K-test, and writes a single `examples.json`
consumed by the static frontend in `phase2/showcase/static/`.

Tags every example with one or more category labels so the browse page
can filter for "interesting" subsets — unanimous, near-tie correct,
near-tie wrong, redundancy save, commit-LoRA repair, esc-early-trigger.

No GPU. No HF push. Pure read/transform/write.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "e4" / "results"
FRONTIER_DIR = REPO_ROOT / "phase2" / "frontier_compare" / "results_50"

# Frontier_compare model order for the showcase Detail page. Tier label
# governs how the frontend groups models. Skip files matching skip_substrs.
FRONTIER_MODELS = [
    # (file_stem, display_name, tier)
    ("claude45",       "Claude Sonnet 4.5",   "frontier"),
    ("gpt4o",          "GPT-4o",              "frontier"),
    ("gemini25",       "Gemini 2.5 Pro",      "frontier"),
    ("deepseek_chat",  "DeepSeek Chat",       "oss-large"),
    ("qwen25_72b",     "Qwen2.5 72B",         "oss-large"),
    ("llama33_70b",    "Llama 3.3 70B",       "oss-large"),
    ("qwen3_30b_a3b",  "Qwen3 30B-A3B",       "oss-mid"),
    ("qwen25_7b",      "Qwen2.5 7B",          "oss-peer"),
    ("llama31_8b",     "Llama 3.1 8B",        "oss-peer"),
    ("mistral_7b_v01", "Mistral 7B v0.1",     "oss-peer"),
]
# 7-8B "peer class" — the fairest comparison to sfumato's ~7B active params.
PEER_STEMS = {"qwen25_7b", "llama31_8b", "mistral_7b_v01"}
# Write into static/ so the deploy directory (Pages root) is self-contained.
OUT = REPO_ROOT / "phase2" / "showcase" / "static" / "examples.json"

# Source JSONLs to merge. Each row already has condition/seed/k_steps/
# pred/gold/correct/flops/trace. We add question + tags.
SOURCES = {
    # Big legacy runs (Phase-1 paper data)
    "cmaj_n200_v3LoRA_seed0": RESULTS / "raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl",
    "cmajc_n100_v3LoRA_seed1": RESULTS / "raw_cmajc_k64_seed1_b5_v3LoRA_N100.jsonl",
    "cmajc_n100_v3LoRA_seed2": RESULTS / "raw_cmajc_k64_seed2_b5_v3LoRA_N100.jsonl",
    # All-conditions sweep N=20 (post-spike-chain showcase data)
    "c1_n20_seed0":      RESULTS / "raw_c1_k0_seed0.jsonl",
    "c2_n20_seed0":      RESULTS / "raw_c2_k64_seed0.jsonl",
    "c2_n20_seed1":      RESULTS / "raw_c2_k64_seed1.jsonl",
    "c2c_n20_seed0":     RESULTS / "raw_c2c_k64_seed0.jsonl",
    "c2c_n20_seed1":     RESULTS / "raw_c2c_k64_seed1.jsonl",
    "c2c_n20_seed2":     RESULTS / "raw_c2c_k64_seed2.jsonl",
    "c2hint_n20_seed0":  RESULTS / "raw_c2hint_k64_seed0.jsonl",
    "c2hint_n20_seed1":  RESULTS / "raw_c2hint_k64_seed1.jsonl",
    "c2empty_n20_seed0": RESULTS / "raw_c2empty_k64_seed0.jsonl",
    "c3_n20_seed0":      RESULTS / "raw_c3_k64_seed0.jsonl",
    "c3p_n20_seed0":     RESULTS / "raw_c3p_k64_seed0.jsonl",
    "c4_n20_seed0":      RESULTS / "raw_c4_k64_seed0.jsonl",
    "cmaj_n20_seed0":    RESULTS / "raw_cmaj_k64_seed0.jsonl",
    "cmajc_n20_seed0":   RESULTS / "raw_cmajc_k64_seed0.jsonl",
    "cmerge_n20_seed0":  RESULTS / "raw_cmerge_k64_seed0.jsonl",
    "crev_n20_seed0":    RESULTS / "raw_crev_k64_seed0.jsonl",
}

# Speed-comparison "post-spike" sources keyed by SAME stem as a baseline
# above. When an entry exists, build_examples joins by `idx` to attach
# `wallclock_ms_post` + `pred_post` + `correct_post`. Activates the
# showcase v1 speed panel when data lands.
#
# Baseline above is treated as "pre-spike" for the matching POST entry.
SPEED_PAIRS: dict[str, Path] = {
    # Example (flip on once pod runs return):
    # "cmajc_n100_v3LoRA_seed1": RESULTS / "raw_cmajc_k64_seed1_b5_v3LoRA_N100_S0S1.jsonl",
    # "cmaj_n200_v3LoRA_seed0":  RESULTS / "raw_cmaj_k64_seed0_b5_v3LoRA_N200_S0S1.jsonl",
}


def _load_gsm8k_test() -> list[dict]:
    """Return list of GSM8K-test problems indexed by position."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("datasets not installed; install via: pip install datasets",
              file=sys.stderr)
        sys.exit(1)
    ds = load_dataset("gsm8k", "main", split="test")
    out = []
    for r in ds:
        # gold answer pattern: "rationale...\n#### N"
        m = re.search(r"####\s*(-?[\d,\.]+)", r["answer"])
        gold = m.group(1).replace(",", "") if m else ""
        out.append(
            {
                "question": r["question"],
                "gold_rationale": r["answer"],
                "gold": gold,
            }
        )
    return out


def _detect_n_branches(trace: dict) -> int:
    """Count branch_* keys in a trace dict; 0 if single-branch / no branching."""
    return sum(1 for k in trace if k.startswith("branch_"))


def _tags_for(rec: dict) -> list[str]:
    """Compute category tags. Auto-detects n_branches from trace shape."""
    tags: list[str] = []
    correct = bool(rec.get("correct"))
    cond = rec.get("condition", "")
    trace = rec["trace"]
    n_branches = _detect_n_branches(trace)

    # Always tag the condition family.
    tags.append(f"cond_{cond}")
    if cond == "cmajc":
        tags.append("commit_lora_active")
    if "esc_trigger_block" in trace:
        tags.append("esc_early_trigger")

    if n_branches == 0:
        # Single-branch path. Just tag correctness.
        tags.append("single_branch_correct" if correct else "single_branch_wrong")
        return tags

    # Branched path: tally votes.
    votes_str = trace.get("votes", "")
    votes = [v.strip() for v in votes_str.split("|")] if votes_str else []
    nonempty = [v for v in votes if v]
    counts = Counter(nonempty)
    if not counts:
        tags.append("no_extractable_answer")
        return tags

    top_a, top_c = counts.most_common(1)[0]
    quorum = (n_branches // 2) + 1
    clear = quorum + 1
    if top_c == n_branches:
        tags.append("unanimous_correct" if correct else "unanimous_wrong")
    elif top_c == quorum:
        tags.append("near_tie_correct" if correct else "near_tie_wrong")
    elif top_c >= clear:
        tags.append("clear_majority_correct" if correct else "clear_majority_wrong")

    if correct and len(set(nonempty)) > 1:
        tags.append("redundancy_save")
    return tags


def _merge_sources_with_questions(
    sources: dict[str, Path], questions: list[dict]
) -> tuple[list[dict], dict]:
    """Walk each source JSONL, attach question + tags, return flat list."""
    out: list[dict] = []
    stats: dict = {"by_source": {}, "by_tag": Counter(), "n_total": 0}

    for src_name, src_path in sources.items():
        if not src_path.exists():
            print(f"[skip] {src_path} not found", file=sys.stderr)
            continue
        n_in_src = 0
        n_correct = 0
        with src_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = int(rec["id"]) if str(rec["id"]).isdigit() else int(rec["idx"])
                if idx < 0 or idx >= len(questions):
                    continue
                q = questions[idx]
                tags = _tags_for(rec)

                trace = rec["trace"]
                n_branches = _detect_n_branches(trace)
                # Collect branch traces (variable count). Empty list when
                # single-branch (cot/diffusion_cot/finalize fields used instead).
                branches = [trace.get(f"branch_{i}", "") for i in range(n_branches)]

                merged = {
                    "source": src_name,
                    "id": str(rec["id"]),
                    "idx": int(rec["idx"]),
                    "condition": rec["condition"],
                    "k_steps": rec["k_steps"],
                    "seed": rec["seed"],
                    "lora_path": rec.get("lora_path", ""),
                    "commit_lora_path": rec.get("commit_lora_path", ""),
                    "n_branches": n_branches,
                    "question": q["question"],
                    "gold": q["gold"],
                    "gold_rationale": q["gold_rationale"],
                    "pred": rec["pred"],
                    "correct": bool(rec["correct"]),
                    "flops": rec.get("flops", 0),
                    "wallclock_ms": rec.get("wallclock_ms"),
                    "branches": branches,
                    # Single-branch trace fields (kept verbatim for the frontend).
                    "cot": trace.get("cot", ""),
                    "plan": trace.get("plan", ""),
                    "diffusion_cot": trace.get("diffusion_cot", ""),
                    "diffusion_scaffold": trace.get("diffusion_scaffold", ""),
                    "finalize": trace.get("finalize", ""),
                    "votes_str": trace.get("votes", ""),
                    "winner": trace.get("winner", ""),
                    "esc_trigger_block": trace.get("esc_trigger_block"),
                    "esc_branches_pruned": trace.get("esc_branches_pruned"),
                    "tags": tags,
                }
                out.append(merged)
                n_in_src += 1
                n_correct += int(merged["correct"])
                for t in tags:
                    stats["by_tag"][t] += 1
        stats["by_source"][src_name] = {
            "n": n_in_src,
            "n_correct": n_correct,
            "accuracy": (n_correct / max(n_in_src, 1)),
        }
    stats["n_total"] = len(out)
    return out, stats


def _attach_speed_pairs(
    records: list[dict],
    speed_pairs: dict[str, Path],
) -> int:
    """For each (source, post_path) pair, attach post-spike fields by idx.

    Adds to matching baseline records:
      - wallclock_ms_post        (wallclock from the post-spike run)
      - pred_post                (post-spike prediction, may differ from pre)
      - correct_post             (bool)
      - speedup                  (float, pre / post; null if either is 0)
      - esc_trigger_block_post   (carried over if present)
      - esc_branches_pruned_post (carried over)

    Returns count of records updated.
    """
    if not speed_pairs:
        return 0

    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], {})[r["idx"]] = r

    updated = 0
    for src, path in speed_pairs.items():
        if not path.exists():
            print(f"[skip-speed] {path} not found", file=sys.stderr)
            continue
        baseline_lookup = by_source.get(src, {})
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = int(row["idx"])
                if idx not in baseline_lookup:
                    continue
                base_rec = baseline_lookup[idx]
                w_pre = base_rec.get("wallclock_ms")
                w_post = row.get("wallclock_ms")
                base_rec["wallclock_ms_post"] = w_post
                base_rec["pred_post"] = row.get("pred", "")
                base_rec["correct_post"] = bool(row.get("correct"))
                if w_pre and w_post and w_post > 0:
                    base_rec["speedup"] = float(w_pre) / float(w_post)
                else:
                    base_rec["speedup"] = None
                tr = row.get("trace", {})
                if "esc_trigger_block" in tr:
                    base_rec["esc_trigger_block_post"] = tr["esc_trigger_block"]
                    base_rec["esc_branches_pruned_post"] = tr.get(
                        "esc_branches_pruned"
                    )
                    if "esc_post_triggered" not in base_rec["tags"]:
                        base_rec["tags"].append("esc_post_triggered")
                updated += 1
    return updated


def _load_frontier_compare() -> dict[int, dict[str, dict]]:
    """Load 10 frontier_compare/results_50/<model>.jsonl files.

    Returns: {idx: {stem: {pred, correct, latency_ms, cost, display, tier}}}
    Skips truncated/error subdirs and any model file missing from disk.
    """
    out: dict[int, dict[str, dict]] = {}
    for stem, display, tier in FRONTIER_MODELS:
        path = FRONTIER_DIR / f"{stem}.jsonl"
        if not path.exists():
            print(f"[frontier] skip missing: {path}", file=sys.stderr)
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = int(r["idx"])
            cell = {
                "pred": r.get("pred"),
                "correct": bool(r.get("correct")),
                "latency_ms": r.get("latency_ms"),
                "cost": (r.get("usage") or {}).get("cost"),
                "display": display,
                "tier": tier,
            }
            out.setdefault(idx, {})[stem] = cell
    return out


def _attach_frontier_compare(
    records: list[dict],
    frontier: dict[int, dict[str, dict]],
) -> int:
    """For records with idx in frontier (idx 0..49), attach a
    `frontier_compare` dict and tag `frontier_unique_win_peer` when sfumato
    is correct AND no peer-class 7-8B model is correct on that idx.

    Only tags on cmajc records (the primary sfumato condition this pilot
    compared against). Returns count of records updated.
    """
    if not frontier:
        return 0
    updated = 0
    tagged_unique = 0
    for r in records:
        cells = frontier.get(r["idx"])
        if not cells:
            continue
        # Attach the per-model comparison block.
        r["frontier_compare"] = cells
        updated += 1
        # Tag unique-WIN over peer class only on cmajc records (primary
        # sfumato condition the pilot compared against).
        if r["condition"] == "cmajc" and r["correct"]:
            peer_correct = any(
                cells.get(p, {}).get("correct", False) for p in PEER_STEMS
            )
            if not peer_correct:
                if "frontier_unique_win_peer" not in r["tags"]:
                    r["tags"].append("frontier_unique_win_peer")
                    tagged_unique += 1
    print(f"[frontier] attached frontier_compare to {updated} records "
          f"(unique_win_peer tagged: {tagged_unique})")
    return updated


def _join_cmaj_vs_cmajc_repair(records: list[dict]) -> int:
    """Tag commit_lora_repair: same idx where cmaj loses but cmajc wins.

    Joins by `idx` across cmaj and cmajc sources. Returns count tagged.
    """
    cmaj_by_idx = {r["idx"]: r for r in records if r["condition"] == "cmaj"}
    cmajc_by_idx: dict[int, list[dict]] = {}
    for r in records:
        if r["condition"] == "cmajc":
            cmajc_by_idx.setdefault(r["idx"], []).append(r)

    tagged = 0
    for idx, cmaj_rec in cmaj_by_idx.items():
        cmajc_recs = cmajc_by_idx.get(idx, [])
        if not cmajc_recs:
            continue
        cmajc_correct_any = any(c["correct"] for c in cmajc_recs)
        if not cmaj_rec["correct"] and cmajc_correct_any:
            for c in cmajc_recs:
                if c["correct"] and "commit_lora_repair" not in c["tags"]:
                    c["tags"].append("commit_lora_repair")
                    tagged += 1
    return tagged


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("[showcase] loading GSM8K-test ...", flush=True)
    questions = _load_gsm8k_test()
    print(f"[showcase] {len(questions)} questions loaded", flush=True)

    print("[showcase] merging source JSONLs ...", flush=True)
    records, stats = _merge_sources_with_questions(SOURCES, questions)
    print(f"[showcase] merged: {stats['n_total']} records", flush=True)
    for src, s in stats["by_source"].items():
        print(f"           {src}: n={s['n']} acc={s['accuracy']:.3f}", flush=True)

    repair_n = _join_cmaj_vs_cmajc_repair(records)
    print(f"[showcase] tagged commit_lora_repair: {repair_n}", flush=True)

    speed_n = _attach_speed_pairs(records, SPEED_PAIRS)
    print(f"[showcase] attached speed-pair records: {speed_n}", flush=True)

    frontier = _load_frontier_compare()
    front_n = _attach_frontier_compare(records, frontier)
    print(f"[showcase] frontier_compare records: {front_n}", flush=True)

    # Re-aggregate tag counts after repair pass.
    tag_counts: Counter = Counter()
    for r in records:
        for t in r["tags"]:
            tag_counts[t] += 1

    payload = {
        "version": 1,
        "generated_at": "2026-05-04",
        "n_records": len(records),
        "stats": {
            "by_source": stats["by_source"],
            "by_tag": dict(tag_counts.most_common()),
        },
        "records": records,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=None))
    print(f"[showcase] wrote {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size} bytes)")
    print("[showcase] tag distribution:")
    for t, c in tag_counts.most_common():
        print(f"           {t}: {c}")


if __name__ == "__main__":
    main()
