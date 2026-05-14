"""D step 2 — LLaDA-scaffolded inference with Sonnet problem setups.

For each of N=200 GSM8K-dev problems, run LLaDA-8B-Instruct + Track-1-v3
LoRA + commit-v3 LoRA with the Sonnet-extracted structured setup as the
prompt. Compare accuracy to a same-config baseline that uses the raw
problem.

Variants (controlled by VARIANT env, default "both"):
  - scaffold: prompt = sonnet_setup only
  - baseline: prompt = raw question (apples-to-apples with no-scaffold)
  - both: run both back-to-back (default)

Reads:
  - e4/results/comprehension_scaffold/sonnet_setups.jsonl (from step 1)
  - HF Hub adapters: eren23/sfumato-llada-prefix-robust-v3,
                     eren23/sfumato-llada-commit-v3

Writes:
  - e4/results/comprehension_scaffold/llada_outputs.jsonl
  - e4/results/comprehension_scaffold/summary.json
  - e4/results/comprehension_scaffold/summary.md

Hardware: single RTX 4090 (24 GB) or comparable. Loads LLaDA-8B in bf16.

Env vars:
  N_PROBLEMS=200       cap on problems (smoke test with 5 or 10)
  VARIANT=both         scaffold | baseline | both
  K_STEPS=64           diffusion steps
  TEMPERATURE=0.0      single-branch greedy
  COMMIT_N_BLOCKS=3    cmajc-v3 default
  BASE_MODEL_NAME=...  override (default GSAI-ML/LLaDA-8B-Instruct)
  LORA_PATH=...        override (default eren23/sfumato-llada-prefix-robust-v3)
  COMMIT_LORA_PATH=... override (default eren23/sfumato-llada-commit-v3)

Usage (on GPU pod after sync_code):
  python e4/scripts/d_llada_scaffold.py 2>&1 | tee /tmp/d_llada_scaffold.log
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e4 import diff_llada, grade  # noqa: E402

SETUPS_JSONL = REPO_ROOT / "e4" / "results" / "comprehension_scaffold" / "sonnet_setups.jsonl"
OUT_DIR = REPO_ROOT / "e4" / "results" / "comprehension_scaffold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "llada_outputs.jsonl"
OUT_SUMMARY_JSON = OUT_DIR / "summary.json"
OUT_SUMMARY_MD = OUT_DIR / "summary.md"

DEFAULT_BASE = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_LORA = "eren23/sfumato-llada-prefix-robust-v3"
DEFAULT_COMMIT_LORA = "eren23/sfumato-llada-commit-v3"


def env_int(k, d):
    return int(os.environ.get(k, str(d)))


def env_float(k, d):
    return float(os.environ.get(k, str(d)))


def main() -> None:
    n_problems = env_int("N_PROBLEMS", 200)
    variant = os.environ.get("VARIANT", "both")
    k_steps = env_int("K_STEPS", 64)
    temperature = env_float("TEMPERATURE", 0.0)
    commit_n_blocks = env_int("COMMIT_N_BLOCKS", 3)
    base = os.environ.get("BASE_MODEL_NAME", DEFAULT_BASE)
    lora = os.environ.get("LORA_PATH", DEFAULT_LORA)
    commit_lora = os.environ.get("COMMIT_LORA_PATH", DEFAULT_COMMIT_LORA)

    assert SETUPS_JSONL.exists(), f"missing {SETUPS_JSONL} — run d_sonnet_setups.py first"

    rows = [json.loads(l) for l in SETUPS_JSONL.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("sonnet_setup")]
    rows.sort(key=lambda r: r["idx"])
    rows = rows[:n_problems]
    print(f"loaded {len(rows)} problems with non-null sonnet_setup")

    print(f"loading LLaDA: base={base} lora={lora} commit_lora={commit_lora}")
    t0 = time.time()
    model = diff_llada.load(
        name=base,
        lora_path=lora,
        commit_lora_path=commit_lora,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    variants_to_run = ["scaffold", "baseline"] if variant == "both" else [variant]
    print(f"variants: {variants_to_run}, k_steps={k_steps}, temperature={temperature}, commit_n_blocks={commit_n_blocks}")

    # Open output JSONL for streaming writes (resume-friendly)
    fh = open(OUT_JSONL, "w")

    results_by_variant = {v: [] for v in variants_to_run}
    n_done = 0
    t_run = time.time()

    for r in rows:
        idx = r["idx"]
        gold = r["gold"]
        question = r["question"]
        setup = r["sonnet_setup"]

        for v in variants_to_run:
            if v == "scaffold":
                prompt = setup
            elif v == "baseline":
                prompt = question
            else:
                raise ValueError(f"unknown variant {v}")

            text, flops_used = model.denoise_block(
                prompt=prompt,
                k_steps=k_steps,
                seed=0,
                temperature=temperature,
                apply_commit=True,
                commit_n_blocks=commit_n_blocks,
            )
            pred = grade.extract_final_answer(text) or grade.extract_answer(text)
            correct = (pred is not None and str(pred).strip() == str(gold).strip())

            out_rec = {
                "idx": idx,
                "variant": v,
                "gold": gold,
                "pred": pred,
                "correct": int(correct),
                "text": text,
                "flops": flops_used,
                "k_steps": k_steps,
                "temperature": temperature,
                "commit_n_blocks": commit_n_blocks,
                "lora_path": lora,
                "commit_lora_path": commit_lora,
            }
            fh.write(json.dumps(out_rec) + "\n")
            fh.flush()
            results_by_variant[v].append(out_rec)

        n_done += 1
        if n_done % 10 == 0 or n_done == len(rows):
            el = time.time() - t_run
            rate = n_done / el if el > 0 else 0
            eta = (len(rows) - n_done) / rate if rate > 0 else 0
            accs = {v: sum(rr["correct"] for rr in results_by_variant[v]) / len(results_by_variant[v])
                    for v in variants_to_run if results_by_variant[v]}
            print(f"  [{n_done}/{len(rows)}] el={el:.0f}s rate={rate:.2f}/s eta={eta:.0f}s {accs}", flush=True)

    fh.close()

    # ---- aggregate ----
    summary = {
        "n_problems": len(rows),
        "k_steps": k_steps,
        "temperature": temperature,
        "commit_n_blocks": commit_n_blocks,
        "base_model": base,
        "lora_path": lora,
        "commit_lora_path": commit_lora,
        "wall_s": round(time.time() - t_run, 1),
        "variants": {},
    }

    for v in variants_to_run:
        recs = results_by_variant[v]
        n_corr = sum(r["correct"] for r in recs)
        summary["variants"][v] = {
            "n": len(recs),
            "n_correct": n_corr,
            "accuracy": round(n_corr / max(1, len(recs)), 4),
        }

    if "scaffold" in summary["variants"] and "baseline" in summary["variants"]:
        delta = summary["variants"]["scaffold"]["accuracy"] - summary["variants"]["baseline"]["accuracy"]
        summary["delta_scaffold_vs_baseline"] = round(delta, 4)
        summary["delta_pp"] = round(delta * 100, 2)

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    md = "# D — comprehension scaffolding test\n\n"
    md += f"## Setup\n\n- Base: `{base}`\n- LoRA: `{lora}`\n- Commit-LoRA: `{commit_lora}`\n"
    md += f"- N={summary['n_problems']}, k_steps={k_steps}, T={temperature}, commit_n_blocks={commit_n_blocks}\n"
    md += f"- Wall: {summary['wall_s']:.0f}s\n\n"
    md += "## Headline\n\n| variant | n | correct | accuracy |\n|---|---|---|---|\n"
    for v in variants_to_run:
        s = summary["variants"][v]
        md += f"| `{v}` | {s['n']} | {s['n_correct']} | {s['accuracy']*100:.1f}% |\n"
    if "delta_pp" in summary:
        md += f"\n**Δ (scaffold − baseline): {summary['delta_pp']:+.2f}pp**\n"
    md += "\n## Reference numbers (from paper 1)\n\n- LLaDA c2 (single branch, no LoRAs, raw Q): 74%\n- cmajc-v3 (b=5, both LoRAs, raw Q): 82.5%\n- Qwen-2.5-7B AR baseline: 86.5%\n"
    md += "\n## Interpretation\n\n"
    if "delta_pp" in summary:
        d = summary["delta_pp"]
        sc = summary["variants"]["scaffold"]["accuracy"] * 100
        bl = summary["variants"]["baseline"]["accuracy"] * 100
        if sc >= 90:
            md += f"- Scaffold lifts to {sc:.1f}% (vs baseline {bl:.1f}%). Δ={d:+.1f}pp. **Paper-1 mechanism claim hardened.** Inference recipe `LLaDA + Sonnet setup` is a deployable variant worth its own paper section.\n"
        elif d >= 4:
            md += f"- Scaffold lifts by {d:+.1f}pp ({bl:.1f}% → {sc:.1f}%). Claim partially supported — meaningful comprehension component but not the whole story.\n"
        elif d >= 1:
            md += f"- Scaffold lifts by {d:+.1f}pp ({bl:.1f}% → {sc:.1f}%). Small positive effect within noise. Worth multi-seed before drawing strong conclusions.\n"
        elif abs(d) < 1:
            md += f"- Scaffold and baseline indistinguishable ({bl:.1f}% vs {sc:.1f}%, Δ={d:+.1f}pp). Comprehension is NOT the dominant axis — claim weakened. Reread paper-1 §discussion.\n"
        else:
            md += f"- Scaffold REGRESSES vs baseline ({bl:.1f}% → {sc:.1f}%, Δ={d:+.1f}pp). LLaDA may struggle with Sonnet's structured format. Try variant B (prepend setup to original Q) as a follow-up.\n"
    md += "\n## Files\n\n- `sonnet_setups.jsonl`, `sonnet_meta.json` — step 1 output\n- `llada_outputs.jsonl` — per-problem raw\n- `summary.json` — machine-readable\n- `summary.md` — this document\n"

    OUT_SUMMARY_MD.write_text(md)

    print()
    print("=== SUMMARY ===")
    for v in variants_to_run:
        s = summary["variants"][v]
        print(f"  {v:<10}: {s['n_correct']}/{s['n']} = {s['accuracy']*100:.1f}%")
    if "delta_pp" in summary:
        print(f"  delta: {summary['delta_pp']:+.2f}pp")
    print(f"  wall: {summary['wall_s']:.0f}s")
    print(f"  wrote {OUT_JSONL}")
    print(f"  wrote {OUT_SUMMARY_JSON}")
    print(f"  wrote {OUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
