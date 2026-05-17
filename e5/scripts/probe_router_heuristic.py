"""Phase I.1 — heuristic router on top of interleaved AR↔diff.

Phase I.0 confirmed that fine-grained alternation (AR(8) → diff(4) × 6)
beats both pure AR and single-switch. The plan's I.1 question: can a
non-learned router that picks chunk-mode based on the model's own
signals beat the best FIXED schedule?

Three heuristics, all use the existing CompositeLM.forward(idx, mode=...)
and produce a `("ar"|"diff", chunk_size)` schedule on the fly.

  H1 entropy-gate:
      after each AR chunk, compute the entropy of the last-position
      AR logits. If entropy > T_HIGH and we are not already in diff
      → next chunk is diff (refine). Else continue AR.

  H2 diversity-gate:
      after each AR chunk, count unique tokens in the last K=8 generated
      tokens. If unique < M=4 → switch to diff (signs of looping).
      Else continue AR.

  H3 confidence-gate (on diff head):
      during a diff round, after each unmasking sub-step, compute mean
      top-1 prob across the just-unmasked positions. If conf > P_HIGH
      → accept and switch back to AR. Else continue diffing.

Each heuristic produces a schedule capped at ~80 generated tokens and
~16 chunks total to bound compute. Compare against:
- pure AR (128 tokens)
- best fixed schedule from I.0 (interleaved_8_4_x6)

Env follows probe_interleaved.py conventions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402
from e5.data import load_gsm8k_dev_questions  # noqa: E402
from e5.scripts.probe5_mode_switch import (  # noqa: E402
    gen_ar, diff_revise, extract_answer,
)
from e5.scripts.probe_interleaved import gen_interleaved  # noqa: E402
from e5.scripts.loop_rate import is_loopy, word_runs, phrase_runs  # noqa: E402


@torch.no_grad()
def _last_pos_entropy(model, ids):
    """Entropy of the last-position AR logits."""
    device = next(model.parameters()).device
    block = model.cfg.block_size
    idx = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
    logits = model(idx, mode="ar")[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * (probs.clamp_min(1e-12)).log()).sum().item())


def _unique_in_tail(ids, k=8):
    return len(set(ids[-k:])) if len(ids) >= k else len(set(ids))


@torch.no_grad()
def gen_h1_entropy(model, prompt, ar_kw, diff_kw, t_high=4.5,
                   ar_chunk=8, diff_chunk=4, n_diff_steps=8,
                   max_tokens=80, max_chunks=16):
    """H1 entropy-gate router."""
    full = list(prompt)
    gen_start = len(prompt)
    chunks_used = 0
    trace = []  # list of (mode, chunk, entropy_seen)
    last_mode = None
    while len(full) - gen_start < max_tokens and chunks_used < max_chunks:
        ent = _last_pos_entropy(model, full)
        # Switch to diff when uncertain AND we have AR content to refine AND we didn't JUST diff
        if ent > t_high and (len(full) - gen_start) >= diff_chunk and last_mode != "diff":
            mode = "diff"
            revise_end = len(full)
            revise_start = revise_end - diff_chunk
            full = diff_revise(model, full, revise_start, revise_end,
                               n_steps=n_diff_steps, **diff_kw)
            trace.append(("diff", diff_chunk, ent))
        else:
            new = gen_ar(model, full, max_new=ar_chunk, **ar_kw)
            full = full + new
            trace.append(("ar", len(new), ent))
            mode = "ar"
        last_mode = mode
        chunks_used += 1
        if model.cfg.block_size and len(full) > model.cfg.block_size:
            drop = len(full) - model.cfg.block_size
            full = full[drop:]
            gen_start = max(0, gen_start - drop)
    return {"gen": full[gen_start:], "trace": trace}


@torch.no_grad()
def gen_h2_diversity(model, prompt, ar_kw, diff_kw, k=8, m=4,
                     ar_chunk=8, diff_chunk=4, n_diff_steps=8,
                     max_tokens=80, max_chunks=16):
    """H2 diversity-gate router."""
    full = list(prompt)
    gen_start = len(prompt)
    chunks_used = 0
    trace = []
    last_mode = None
    while len(full) - gen_start < max_tokens and chunks_used < max_chunks:
        u = _unique_in_tail(full[gen_start:], k=k)
        if (len(full) - gen_start) >= k and u < m and last_mode != "diff":
            mode = "diff"
            revise_end = len(full)
            revise_start = revise_end - diff_chunk
            full = diff_revise(model, full, revise_start, revise_end,
                               n_steps=n_diff_steps, **diff_kw)
            trace.append(("diff", diff_chunk, u))
        else:
            new = gen_ar(model, full, max_new=ar_chunk, **ar_kw)
            full = full + new
            trace.append(("ar", len(new), u))
            mode = "ar"
        last_mode = mode
        chunks_used += 1
        if model.cfg.block_size and len(full) > model.cfg.block_size:
            drop = len(full) - model.cfg.block_size
            full = full[drop:]
            gen_start = max(0, gen_start - drop)
    return {"gen": full[gen_start:], "trace": trace}


@torch.no_grad()
def gen_h3_diff_confidence(model, prompt, ar_kw, diff_kw, p_high=0.5,
                            ar_chunk=8, diff_chunk_initial=8, max_diff_rounds=4,
                            n_diff_steps=8, max_tokens=80, max_chunks=16):
    """H3 diff-confidence gate. During a diff round we KEEP diffing
    only while mean conf across unmasked positions stays below p_high."""
    # Simpler form: enter diff after every AR chunk; keep diffing
    # adaptively until mean conf > p_high or max_diff_rounds reached.
    device = next(model.parameters()).device
    full = list(prompt)
    gen_start = len(prompt)
    chunks_used = 0
    trace = []
    while len(full) - gen_start < max_tokens and chunks_used < max_chunks:
        # AR chunk
        new = gen_ar(model, full, max_new=ar_chunk, **ar_kw)
        full = full + new
        trace.append(("ar", len(new), None))
        chunks_used += 1
        if chunks_used >= max_chunks: break

        # Adaptive diff phase
        diff_chunk = diff_chunk_initial
        for r in range(max_diff_rounds):
            if (len(full) - gen_start) < diff_chunk: break
            # diff_revise then measure mean conf of last region under diff head
            revise_end = len(full)
            revise_start = revise_end - diff_chunk
            before = list(full)
            full = diff_revise(model, full, revise_start, revise_end,
                               n_steps=n_diff_steps, **diff_kw)
            idx = torch.tensor([full[-model.cfg.block_size:]], dtype=torch.long, device=device)
            logits = model(idx, mode="diff")[0, -diff_chunk:, :].float()
            probs = torch.softmax(logits, dim=-1)
            mean_conf = float(probs.max(dim=-1).values.mean().item())
            trace.append(("diff", diff_chunk, mean_conf))
            chunks_used += 1
            if mean_conf > p_high: break
        if model.cfg.block_size and len(full) > model.cfg.block_size:
            drop = len(full) - model.cfg.block_size
            full = full[drop:]
            gen_start = max(0, gen_start - drop)
    return {"gen": full[gen_start:], "trace": trace}


def eval_router(model, problems, router_fn, tok, ar_kw, diff_kw, name: str):
    print(f"\n[{name}]")
    t0 = time.time()
    per = []
    trigger_hist = {"ar": 0, "diff": 0}
    for p in problems:
        res = router_fn(model, p["prompt_tokens"], ar_kw, diff_kw)
        text = tok.decode([t for t in res["gen"] if t < 50257], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred is not None and pred == p["gold"])
        for tr in res["trace"]:
            trigger_hist[tr[0]] = trigger_hist.get(tr[0], 0) + 1
        per.append({
            "idx": p["idx"], "gold": p["gold"], "pred": pred, "correct": correct,
            "text": text,
            "final_loopy": int(is_loopy(text)),
            "final_word_run": word_runs(text),
            "final_phrase_run": phrase_runs(text, 3),
            "trace_len": len(res["trace"]),
            "n_diff_in_trace": sum(1 for tr in res["trace"] if tr[0] == "diff"),
        })
    n_corr = sum(r["correct"] for r in per)
    n_loopy = sum(r["final_loopy"] for r in per)
    n_total_chunks = sum(r["trace_len"] for r in per)
    summary = {
        "name": name,
        "n": len(per),
        "n_correct": n_corr,
        "accuracy": round(n_corr / max(1, len(per)), 4),
        "loop_rate_final": round(n_loopy / max(1, len(per)), 4),
        "max_word_run": max((r["final_word_run"] for r in per), default=0),
        "wall_s": round(time.time() - t0, 1),
        "trigger_ar_frac": round(trigger_hist["ar"] / max(1, n_total_chunks), 3),
        "trigger_diff_frac": round(trigger_hist["diff"] / max(1, n_total_chunks), 3),
        "results_sample": per[:10],
    }
    print(f"  acc={summary['accuracy']*100:.1f}%  loop={summary['loop_rate_final']*100:.0f}%  "
          f"max_run={summary['max_word_run']}  "
          f"trig_ar={summary['trigger_ar_frac']:.2f}  trig_diff={summary['trigger_diff_frac']:.2f}  "
          f"wall={summary['wall_s']:.0f}s")
    return summary


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "50"))
    out_path = Path(os.environ.get("OUT", "probe_router_heuristic.json"))
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

    if os.environ.get("DECODE_GREEDY", "0") == "1":
        ar_kw, diff_kw = {}, {}
    else:
        ar_kw = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3)
        diff_kw = dict(diff_temperature=0.8, diff_top_p=0.9, diff_repetition_penalty=1.15)

    # H1 entropy thresholds to sweep
    t_high = float(os.environ.get("T_HIGH", "4.5"))
    p_high = float(os.environ.get("P_HIGH", "0.5"))

    out = {"ckpt": str(ckpt), "n_eval": n_eval, "decode": {"ar": ar_kw, "diff": diff_kw},
           "heuristics": {}}

    out["heuristics"]["h1_entropy_t4.5"] = eval_router(
        model, problems,
        lambda m, pr, a, d: gen_h1_entropy(m, pr, a, d, t_high=t_high),
        tok, ar_kw, diff_kw, f"H1 entropy>={t_high}",
    )
    out["heuristics"]["h2_diversity_u4"] = eval_router(
        model, problems,
        lambda m, pr, a, d: gen_h2_diversity(m, pr, a, d, k=8, m=4),
        tok, ar_kw, diff_kw, "H2 diversity<4",
    )
    out["heuristics"]["h3_diff_confidence_p0.5"] = eval_router(
        model, problems,
        lambda m, pr, a, d: gen_h3_diff_confidence(m, pr, a, d, p_high=p_high),
        tok, ar_kw, diff_kw, f"H3 diff_conf>={p_high}",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print("Summary:")
    for name, c in out["heuristics"].items():
        print(f"  {name:30s} acc={c['accuracy']*100:5.1f}%  loop={c['loop_rate_final']*100:4.0f}%  "
              f"diff_frac={c['trigger_diff_frac']:.2f}  max_run={c['max_word_run']:3d}")


if __name__ == "__main__":
    main()
