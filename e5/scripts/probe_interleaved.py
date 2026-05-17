"""Phase I.0 — scripted multi-round AR ↔ diff interleaving.

Hypothesis: chaining gen_ar() and diff_revise() in a loop (AR for K
tokens, then diff-revise the last J tokens, repeat) either lifts
GSM8K-dev accuracy over the single-round probe5 baseline OR converges
to a fixed point OR diverges into loops. Each is informative.

Configs (5 of them) match the plan file's "Phase I.0" table:

  single_ar_128         pure AR baseline, 128 tokens
  single_switch_64_32   probe5's AR(64) + diff-revise last 32, n_steps=16
  interleaved_16_8_x3   AR(16) → diff(8) × 3 rounds  (72 tokens total)
  interleaved_8_4_x6    AR(8) → diff(4) × 6 rounds   (72 tokens total)
  interleaved_32_16_x2  AR(32) → diff(16) × 2 rounds (96 tokens total)

Each diff round uses count-based rep_pen on the diff head per Phase H+.

Env:
  CKPT=path/to/model.pt
  N_EVAL=50
  OUT=path/to/probe_interleaved.json
  PROMPT_FORMAT=qa|prose|fewshot   (default qa)
  DECODE_GREEDY=1                  (default 0, sampling enabled)
  TEMP=0.8 TOP_P=0.9 REP_PEN=1.15 NO_REPEAT_NGRAM=3
  DIFF_TEMP=0.8 DIFF_TOP_P=0.9 DIFF_REP_PEN=1.15
  DEVICE=cuda|mps|cpu              (auto-detect by default)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402
from e5.data import load_gsm8k_dev_questions  # noqa: E402
from e5.scripts.probe5_mode_switch import (  # noqa: E402
    gen_ar, diff_revise, extract_answer, eval_mode,
)
from e5.scripts.loop_rate import is_loopy, word_runs, phrase_runs  # noqa: E402


@torch.no_grad()
def gen_interleaved(model, prompt, schedule, ar_kw, diff_kw, n_diff_steps: int = 8,
                    track_intermediate: bool = True):
    """Chain gen_ar + diff_revise per the schedule.

    schedule: list of (mode, chunk_size) pairs.
      ("ar",   K) → append K tokens via gen_ar
      ("diff", J) → re-mask and diff-revise the last J tokens

    Returns dict {gen, intermediate_states, intermediate_loopy}.
    intermediate_states: list of strings (one per phase boundary).
    intermediate_loopy: list of bools (one per intermediate state).
    """
    block_size = model.cfg.block_size
    full = list(prompt)
    gen_start = len(prompt)
    intermediate_states: list[list[int]] = []

    for mode, chunk in schedule:
        if mode == "ar":
            # gen_ar takes the FULL prompt context and returns only the new tokens
            new = gen_ar(model, full, max_new=chunk, **ar_kw)
            full = full + new
            # respect block_size: slide window if we exceed
            if len(full) > block_size:
                # keep last block_size tokens; gen_start shifts accordingly
                drop = len(full) - block_size
                full = full[drop:]
                gen_start = max(0, gen_start - drop)
        elif mode == "diff":
            # re-mask the last `chunk` tokens of full and diff-revise them
            if len(full) < chunk + 1:
                continue
            revise_end = len(full)
            revise_start = revise_end - chunk
            full = diff_revise(model, full, revise_start, revise_end,
                               n_steps=n_diff_steps, **diff_kw)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        if track_intermediate:
            intermediate_states.append(list(full[gen_start:]))

    gen = full[gen_start:]
    return {"gen": gen, "intermediate": intermediate_states}


def gen_text_factory(schedule, ar_kw, diff_kw, n_diff_steps=8):
    def _fn(model, prompt):
        out = gen_interleaved(model, prompt, schedule, ar_kw, diff_kw, n_diff_steps=n_diff_steps)
        return out["gen"]
    return _fn


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "50"))
    out_path = Path(os.environ.get("OUT", "probe_interleaved.json"))
    prompt_format = os.environ.get("PROMPT_FORMAT", "qa")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = os.environ.get("DEVICE", device)
    print(f"device={device} ckpt={ckpt} prompt_format={prompt_format}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M param composite")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    problems = load_gsm8k_dev_questions(n=n_eval, prompt_format=prompt_format)

    # Decode configs (anti-rep)
    if os.environ.get("DECODE_GREEDY", "0") == "1":
        ar_kw: dict = {}
        diff_kw: dict = {}
    else:
        ar_kw = dict(
            temperature=float(os.environ.get("TEMP", "0.8")),
            top_p=float(os.environ.get("TOP_P", "0.9")),
            repetition_penalty=float(os.environ.get("REP_PEN", "1.15")),
            no_repeat_ngram_size=int(os.environ.get("NO_REPEAT_NGRAM", "3")),
        )
        diff_kw = dict(
            diff_temperature=float(os.environ.get("DIFF_TEMP", "0.8")),
            diff_top_p=float(os.environ.get("DIFF_TOP_P", "0.9")),
            diff_repetition_penalty=float(os.environ.get("DIFF_REP_PEN", "1.15")),
        )

    # The 5 configs from the plan
    configs = {
        "single_ar_128":         [("ar", 128)],
        "single_switch_64_32":   [("ar", 64), ("diff", 32)],
        "interleaved_16_8_x3":   [("ar", 16), ("diff", 8)] * 3,
        "interleaved_8_4_x6":    [("ar", 8),  ("diff", 4)] * 6,
        "interleaved_32_16_x2":  [("ar", 32), ("diff", 16)] * 2,
    }

    out = {
        "ckpt": str(ckpt),
        "n_eval": n_eval,
        "prompt_format": prompt_format,
        "decode": {"ar": ar_kw, "diff": diff_kw},
        "configs": {},
    }

    for name, schedule in configs.items():
        print(f"\n[{name}] schedule={schedule}")
        t0 = time.time()
        per_problem = []
        for i, p in enumerate(problems):
            res = gen_interleaved(model, p["prompt_tokens"], schedule, ar_kw, diff_kw)
            text = tok.decode([t for t in res["gen"] if t < 50257], skip_special_tokens=True)
            pred = extract_answer(text)
            correct = int(pred is not None and pred == p["gold"])

            inter_loopy = []
            inter_word_runs = []
            for s in res["intermediate"]:
                s_text = tok.decode([t for t in s if t < 50257], skip_special_tokens=True)
                inter_loopy.append(int(is_loopy(s_text)))
                inter_word_runs.append(word_runs(s_text))

            per_problem.append({
                "idx": p["idx"],
                "gold": p["gold"],
                "pred": pred,
                "correct": correct,
                "text": text,
                "final_loopy": int(is_loopy(text)),
                "final_word_run": word_runs(text),
                "final_phrase_run": phrase_runs(text, 3),
                "intermediate_loopy": inter_loopy,
                "intermediate_word_runs": inter_word_runs,
            })

        n_corr = sum(r["correct"] for r in per_problem)
        n_loopy_final = sum(r["final_loopy"] for r in per_problem)
        mean_inter_loopy = (
            sum(sum(r["intermediate_loopy"]) for r in per_problem) /
            max(1, sum(len(r["intermediate_loopy"]) for r in per_problem))
        )
        max_word_run = max((r["final_word_run"] for r in per_problem), default=0)

        cfg_summary = {
            "schedule": schedule,
            "n": len(per_problem),
            "n_correct": n_corr,
            "accuracy": round(n_corr / max(1, len(per_problem)), 4),
            "loop_rate_final": round(n_loopy_final / max(1, len(per_problem)), 4),
            "mean_intermediate_loop_rate": round(mean_inter_loopy, 4),
            "max_word_run": max_word_run,
            "wall_s": round(time.time() - t0, 1),
            "results_sample": per_problem[:10],
        }
        out["configs"][name] = cfg_summary
        print(f"  acc={cfg_summary['accuracy']*100:.1f}%  loop={cfg_summary['loop_rate_final']*100:.0f}%  "
              f"int_loop={cfg_summary['mean_intermediate_loop_rate']*100:.0f}%  "
              f"max_run={cfg_summary['max_word_run']}  wall={cfg_summary['wall_s']:.0f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print("Summary:")
    for name, c in out["configs"].items():
        print(f"  {name:24s} acc={c['accuracy']*100:5.1f}%  loop_final={c['loop_rate_final']*100:4.0f}%  "
              f"int_loop={c['mean_intermediate_loop_rate']*100:4.0f}%  max_run={c['max_word_run']:3d}")


if __name__ == "__main__":
    main()
