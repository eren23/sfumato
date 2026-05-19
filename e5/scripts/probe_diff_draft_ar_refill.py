"""Phase K.2 — per-token diff-draft + AR-refill.

User's idea: composite diff fills K tokens in parallel (one forward
pass amortised across N=16 denoising steps). For positions where the
diff head is uncertain (top-1 probability < threshold), AR re-fills
only those positions with its causal attention. Diff covers the
"easy" mass, AR covers the "structure" / suspect tokens.

Novel vs published lit:
  - DEER (2512.15176) drafts with diffusion, verifies entire BLOCKS
    with AR. We verify per-position.
  - I-DLM / Corrective DLMs / Deferred Commitment do
    within-diffusion correction without ever escaping to AR.
  - We are the first per-token diff→AR refill at <1B text scale.

Pipeline:
  1. Optionally AR-generate a prefix of length k_ar (default 64).
  2. Diff-fill the next k_diff tokens in n_steps (default 16).
  3. Read per-position top-1 probability from the diff head's final
     output.
  4. Flag positions where confidence < T (default 0.6).
  5. For each flagged position, AR-refill via a single forward pass
     conditioned on [prefix + remaining diff-filled tokens up to that
     position]. (Causal attention naturally; future flagged tokens
     don't influence this position's AR prediction.)
  6. Optionally re-run diff-fill on the refined sequence to check
     for new low-confidence regions.

Benchmarks at N=50 (held-out GSM8K problems offset 7000+):
  - refill_rate: fraction of diff-filled tokens that get AR-refilled
  - per-token NLL on gold continuation (Phase J.2 style)
  - wall-clock per generation

Env:
  CKPT=path/to/model.pt   N_EVAL=50   OUT=path
  K_AR=64                 K_DIFF=64
  N_DIFF_STEPS=16         CONF_THRESHOLD=0.6
  AR_REFILL_MODE=argmax|sample
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402
from e5.scripts.probe5_mode_switch import (  # noqa: E402
    gen_ar, diff_revise, _diff_sample_pred,
)
from e5.scripts.loop_rate import is_loopy, word_runs, phrase_runs  # noqa: E402


def load_gsm8k_probe_problems(n: int = 50, offset: int = 7000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(offset, min(offset + n, len(ds))):
        row = ds[idx]
        p = tok.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
        a = tok.encode(" " + row["answer"], add_special_tokens=False)
        out.append({"idx": idx, "prompt_tokens": p, "gold_answer_tokens": a})
    return out, tok


@torch.no_grad()
def _diff_revise_with_conf(model, full_ids: list[int], revise_start: int,
                            revise_end: int, n_steps: int = 16,
                            diff_temperature: float = 0.0, diff_top_p=None,
                            diff_repetition_penalty: float = 1.0):
    """Same as diff_revise but ALSO records per-position commit-time
    confidence (the top-1 prob at the step the position got unmasked).
    Returns (refined_seq, commit_conf) where commit_conf has length
    (revise_end - revise_start)."""
    device = next(model.parameters()).device
    idx = torch.tensor([full_ids], dtype=torch.long, device=device)
    idx[0, revise_start:revise_end] = MASK_TOKEN_ID
    region_len = revise_end - revise_start
    commit_conf = [None] * region_len  # filled when each position gets unmasked

    for step in range(n_steps):
        logits = model(idx, mode="diff")
        region = idx[0, revise_start:revise_end]
        masked = (region == MASK_TOKEN_ID)
        if not masked.any():
            break
        conf, pred = _diff_sample_pred(
            logits[0, revise_start:revise_end], idx[0],
            temperature=diff_temperature, top_p=diff_top_p,
            repetition_penalty=diff_repetition_penalty,
        )
        n_masked = int(masked.sum().item())
        n_to_unmask = max(1, int(n_masked * (step + 1) / n_steps) - ((revise_end - revise_start) - n_masked))
        n_to_unmask = min(n_to_unmask, n_masked)
        conf_masked = torch.where(masked, conf, torch.full_like(conf, -1.0))
        _, top_idx = torch.topk(conf_masked, k=n_to_unmask)
        new_region = region.clone()
        for j in top_idx.tolist():
            commit_conf[j] = float(conf[j].item())
            new_region[j] = pred[j]
        idx[0, revise_start:revise_end] = new_region

    # Any remaining masks get force-decoded to eos with conf=0
    final_region = idx[0, revise_start:revise_end].tolist()
    for j in range(region_len):
        if commit_conf[j] is None:
            commit_conf[j] = 0.0
            if final_region[j] == MASK_TOKEN_ID:
                final_region[j] = 50256  # eos
    return full_ids[:revise_start] + final_region + full_ids[revise_end:], commit_conf


@torch.no_grad()
def extract_diff_confidence(model, seq: list[int], region_start: int, region_end: int):
    """For each position in [region_start, region_end), return the AR
    head's teacher-forced probability for the placed token (i.e., the
    AR-verifier endorsement).

    This is the classical speculative-decoding accept/reject signal,
    adapted to per-position rather than per-block: the diff head DRAFTED
    a token at this position; the AR head's probability for that token
    (given left context) measures whether AR would have chosen the same
    or something close.

    Low AR-prob = AR strongly disagrees with the draft → flag for refill.
    High AR-prob = AR endorses → keep.

    Implementation: one batched AR forward pass over the entire sequence,
    read logits at each position-1 (which predicts the token at position).
    This is O(1) forward passes for the whole region, much cheaper than
    re-masking per position."""
    device = next(model.parameters()).device
    block = model.cfg.block_size
    if len(seq) > block:
        # truncate from the LEFT so the region is intact at the tail
        start = max(0, len(seq) - block)
        window = seq[start:]
        offset = start
    else:
        window = seq
        offset = 0
    idx = torch.tensor([window], dtype=torch.long, device=device)
    logits = model(idx, mode="ar")[0].float()  # (T, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    confs: list[float] = []
    for pos in range(region_start, region_end):
        # AR's prediction for position `pos` is at logits row pos-1
        pred_row = pos - 1 - offset
        if pred_row < 0 or pred_row >= log_probs.size(0):
            confs.append(0.0)
            continue
        placed_token = seq[pos]
        # convert log prob to prob
        p = float(log_probs[pred_row, placed_token].exp().item())
        confs.append(p)
    return confs


def flag_suspect_positions(conf_per_pos: list[float], threshold: float) -> list[int]:
    """Flag positions for AR refill.

    If `threshold` >= 1.0 it is interpreted as a PERCENTILE rank: e.g.
    threshold=25 flags the bottom-25% of positions by confidence. This is
    robust to the absolute scale of conf, which at 305M is universally
    low (mean ~0.03 for our model).

    If `threshold` < 1.0 it is interpreted as an ABSOLUTE confidence
    threshold (legacy behaviour): flag positions with conf < threshold."""
    if threshold >= 1.0:
        # percentile mode
        pct = threshold
        n_flag = int(round(len(conf_per_pos) * pct / 100.0))
        if n_flag == 0:
            return []
        # Sort ascending, take indices of the n_flag lowest
        order = sorted(range(len(conf_per_pos)), key=lambda i: conf_per_pos[i])
        return sorted(order[:n_flag])
    return [i for i, c in enumerate(conf_per_pos) if c < threshold]


@torch.no_grad()
def ar_refill(model, seq: list[int], region_start: int,
              suspect_positions: list[int], mode: str = "argmax"):
    """For each flagged position, replace seq[region_start + i] with the
    AR head's prediction conditioned on [seq[:region_start + i]] (the
    left context up to but not including the flagged position). Returns
    the refilled sequence."""
    device = next(model.parameters()).device
    block = model.cfg.block_size
    refined = list(seq)
    for i in suspect_positions:
        pos = region_start + i
        # AR sees everything up to position pos-1; we ask "what is the
        # token at pos given that context?"
        ctx = refined[:pos]
        if len(ctx) == 0:
            continue
        idx = torch.tensor([ctx[-block:]], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")[0, -1, :].float()
        if mode == "argmax":
            tok = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / 0.8, dim=-1)
            tok = int(torch.multinomial(probs, num_samples=1).item())
        refined[pos] = tok
    return refined


@torch.no_grad()
def gen_diff_draft_ar_refill(model, prompt: list[int], k_ar: int = 64,
                              k_diff: int = 64, n_diff_steps: int = 16,
                              conf_threshold: float = 0.6,
                              n_outer_iter: int = 1,
                              ar_kw: dict | None = None,
                              diff_kw: dict | None = None,
                              ar_refill_mode: str = "argmax"):
    """Top-level recipe.

    Returns dict with:
      'gen'        — the generated tokens (k_ar + k_diff)
      'refill_idx' — which positions in the diff region were AR-refilled
      'conf_pre'   — per-position confidence after diff fill, before refill
    """
    ar_kw = ar_kw or {}
    diff_kw = diff_kw or {}
    # Stage A: AR prefix
    if k_ar > 0:
        ar_part = gen_ar(model, prompt, max_new=k_ar, **ar_kw)
    else:
        ar_part = []
    full = list(prompt) + ar_part
    diff_region_start = len(full)

    # Stage B: diff fill k_diff tokens, capturing per-position commit-
    # time confidence (the prob the diff head assigned to the token it
    # chose, at the step the position got unmasked).
    full_pre_diff_len = len(full)
    full = full + [MASK_TOKEN_ID] * k_diff
    full, commit_conf = _diff_revise_with_conf(model, full, full_pre_diff_len,
                                                full_pre_diff_len + k_diff,
                                                n_steps=n_diff_steps, **diff_kw)
    # We'll use commit_conf as the routing signal in Stage C.

    refill_history = []
    conf_history = []
    # First outer iter: use commit-time confidence from the diff fill
    conf_history.append(commit_conf)
    suspects = flag_suspect_positions(commit_conf, conf_threshold)
    refill_history.append(suspects)
    if suspects:
        full = ar_refill(model, full, diff_region_start, suspects, mode=ar_refill_mode)

    # Subsequent outer iters: re-run diff on the refilled region, capture
    # new commit-time conf, flag again
    for outer in range(1, n_outer_iter):
        # Re-mask only the previously-refilled positions and re-run diff
        if not refill_history[-1]:
            break
        scratch = list(full)
        for i in refill_history[-1]:
            scratch[diff_region_start + i] = MASK_TOKEN_ID
        scratch, conf = _diff_revise_with_conf(model, scratch, diff_region_start,
                                                diff_region_start + k_diff,
                                                n_steps=n_diff_steps, **diff_kw)
        conf_history.append(conf)
        suspects = flag_suspect_positions(conf, conf_threshold)
        refill_history.append(suspects)
        if not suspects:
            full = scratch
            break
        full = ar_refill(model, scratch, diff_region_start, suspects, mode=ar_refill_mode)

    gen = full[len(prompt):]
    return {
        "gen": gen,
        "refill_history": refill_history,
        "conf_history": conf_history,
        "ar_part_len": len(ar_part),
        "diff_region_start_in_gen": len(ar_part),
        "k_diff": k_diff,
    }


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "50"))
    out_path = Path(os.environ.get("OUT", "probe_diff_draft_ar_refill.json"))
    k_ar = int(os.environ.get("K_AR", "64"))
    k_diff = int(os.environ.get("K_DIFF", "64"))
    n_diff_steps = int(os.environ.get("N_DIFF_STEPS", "16"))
    conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.6"))
    n_outer = int(os.environ.get("N_OUTER", "1"))

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = os.environ.get("DEVICE", device)
    print(f"device={device} ckpt={ckpt}")
    print(f"k_ar={k_ar} k_diff={k_diff} n_diff_steps={n_diff_steps} T={conf_threshold} n_outer={n_outer}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {n_params/1e6:.1f}M param composite")

    problems, tok = load_gsm8k_probe_problems(n=n_eval)
    print(f"loaded {len(problems)} held-out problems")

    ar_kw = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3)
    diff_kw = dict(diff_temperature=0.8, diff_top_p=0.9, diff_repetition_penalty=1.15)

    # Sweep refill percentages (interpreted as 'flag bottom-K% by conf')
    thresholds = [10.0, 25.0, 50.0, 75.0]
    out = {
        "ckpt": str(ckpt), "n_eval": n_eval,
        "k_ar": k_ar, "k_diff": k_diff, "n_diff_steps": n_diff_steps,
        "n_outer": n_outer,
        "configs": {},
    }

    for T in thresholds:
        # Naming: T>=1 means percentile-K%; T<1 means absolute conf threshold
        name = f"diff_draft_ar_refill_pct{int(T)}" if T >= 1.0 else f"diff_draft_ar_refill_T{T:.2f}"
        print(f"\n[{name}]")
        per_problem = []
        refill_rates = []
        walls = []
        for p in problems:
            prompt = p["prompt_tokens"]
            gold = p["gold_answer_tokens"]
            t0 = time.perf_counter()
            res = gen_diff_draft_ar_refill(model, prompt,
                                           k_ar=k_ar, k_diff=k_diff,
                                           n_diff_steps=n_diff_steps,
                                           conf_threshold=T,
                                           n_outer_iter=n_outer,
                                           ar_kw=ar_kw, diff_kw=diff_kw)
            wall = time.perf_counter() - t0
            walls.append(wall)
            n_filled = res["k_diff"]
            n_refilled = sum(len(s) for s in res["refill_history"])
            refill_rates.append(n_refilled / max(1, n_filled))

            # Score per-token gold NLL on the entire generation (Phase J.2 style)
            gen = res["gen"]
            n_compare = min(len(gen), len(gold), 128)
            nlls = []
            for t_idx in range(n_compare):
                ctx = prompt + gen[:t_idx]
                idx = torch.tensor([ctx[-cfg.block_size:]], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(idx, mode="ar")[0, -1, :].float()
                    log_probs = torch.log_softmax(logits, dim=-1)
                    nll = float(-log_probs[gold[t_idx]].item())
                nlls.append(nll)
            text = tok.decode([t for t in gen if t < 50257], skip_special_tokens=True)
            per_problem.append({
                "idx": p["idx"],
                "n_compare": n_compare,
                "mean_nll": round(sum(nlls) / max(1, len(nlls)), 4),
                "refill_count": n_refilled,
                "refill_rate": round(n_refilled / max(1, n_filled), 3),
                "wall_s": round(wall, 3),
                "final_loopy": int(is_loopy(text)),
                "max_word_run": word_runs(text),
                "text_head": text[:200],
            })

        all_nlls = [r["mean_nll"] for r in per_problem]
        mean_nll = statistics.mean(all_nlls)
        sd_nll = statistics.stdev(all_nlls) if len(all_nlls) > 1 else 0.0
        mean_refill = statistics.mean(refill_rates)
        mean_wall = statistics.mean(walls)
        n_loopy = sum(r["final_loopy"] for r in per_problem)

        out["configs"][name] = {
            "threshold": T,
            "n": len(per_problem),
            "mean_nll_on_gold": round(mean_nll, 4),
            "sd_nll": round(sd_nll, 4),
            "mean_refill_rate": round(mean_refill, 3),
            "mean_wall_s": round(mean_wall, 3),
            "loop_rate": round(n_loopy / max(1, len(per_problem)), 3),
            "results_sample": per_problem[:10],
        }
        print(f"  T={T}  mean_nll={mean_nll:.4f}±{sd_nll:.4f}  refill_rate={mean_refill:.2%}  "
              f"wall={mean_wall:.3f}s  loop={n_loopy}/{len(per_problem)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print("\nSummary table:")
    print(f"  {'threshold':>10s} {'mean_nll':>10s} {'refill%':>10s} {'wall_s':>10s} {'loop':>8s}")
    for name, c in out["configs"].items():
        print(f"  T={c['threshold']:.1f}      {c['mean_nll_on_gold']:>10.4f} "
              f"{c['mean_refill_rate']*100:>9.1f}% {c['mean_wall_s']:>10.3f} {c['loop_rate']*100:>7.1f}%")


if __name__ == "__main__":
    main()
