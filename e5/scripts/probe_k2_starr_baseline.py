"""Phase O — STaRR-style intra-diff baseline for K.2 head-to-head.

K.2 (our paper): diff fills K positions in parallel, then AR refills the
percentile-bottom-K%. Cross-mode.

STaRR-style baseline (this script): diff fills K positions, then DIFF
RE-FILLS the percentile-bottom-K% via re-masking and another diff round.
Intra-diff only. Tests "is cross-mode the actual lever, or does the
percentile-rank-and-refill mechanism work intra-diff too?"

This is a stripped-down approximation of STaRR (arxiv:2601.04205) — they
use spatial-temporal variance tracking; we use the same percentile-rank
commit-time confidence signal as K.2 for fair comparison.

ENV: CKPT=...  N_EVAL=100  OUT=...  K_AR=64  K_DIFF=64  N_DIFF_STEPS=16
     CONF_THRESHOLD=50.0  DEVICE=cuda|mps|cpu
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from e5.model_composite import CompositeLM, CompositeConfig, MASK_TOKEN_ID
from e5.scripts.probe_diff_draft_ar_refill import (
    _diff_revise_with_conf, flag_suspect_positions, load_gsm8k_probe_problems,
)
from e5.scripts.probe5_mode_switch import gen_ar


def _diff_refill_in_place(model, full: list[int], region_start: int,
                          region_end: int, suspect_positions: list[int],
                          n_diff_steps: int, diff_kw: dict):
    """Re-mask suspect positions and run another diff pass to fill them.
    suspect_positions are indices INTO the diff region (0..region_len-1).
    """
    if not suspect_positions:
        return full, []
    scratch = list(full)
    for i in suspect_positions:
        scratch[region_start + i] = MASK_TOKEN_ID
    scratch, new_conf = _diff_revise_with_conf(model, scratch, region_start,
                                                region_end, n_steps=n_diff_steps,
                                                **diff_kw)
    return scratch, new_conf


@torch.no_grad()
def gen_starr_intra_diff(model, prompt: list[int], k_ar: int, k_diff: int,
                         n_diff_steps: int, conf_threshold: float,
                         ar_kw: dict, diff_kw: dict):
    """Same as K.2 but Stage C is intra-diff re-fill instead of AR refill."""
    # Stage A: AR prefix
    if k_ar > 0:
        ar_part = gen_ar(model, prompt, max_new=k_ar, **ar_kw)
    else:
        ar_part = []
    full = list(prompt) + ar_part
    region_start = len(full)
    full = full + [MASK_TOKEN_ID] * k_diff
    region_end = region_start + k_diff

    # Stage B: diff fill
    full, commit_conf = _diff_revise_with_conf(
        model, full, region_start, region_end, n_steps=n_diff_steps, **diff_kw
    )

    # Stage C: STaRR-style intra-diff re-fill of bottom-K% positions
    suspects = flag_suspect_positions(commit_conf, conf_threshold)
    full, new_conf = _diff_refill_in_place(model, full, region_start, region_end,
                                            suspects, n_diff_steps, diff_kw)

    return {
        "gen": full[len(prompt):],
        "ar_part_len": len(ar_part),
        "diff_region_start_in_gen": len(ar_part),
        "k_diff": k_diff,
        "n_refilled": len(suspects),
    }


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "100"))
    out_path = Path(os.environ.get("OUT", "probe_starr_baseline.json"))
    k_ar = int(os.environ.get("K_AR", "64"))
    k_diff = int(os.environ.get("K_DIFF", "64"))
    n_diff_steps = int(os.environ.get("N_DIFF_STEPS", "16"))
    conf_thresholds = [float(t) for t in os.environ.get("CONF_THRESHOLDS",
                                                          "10,25,50,75").split(",")]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = os.environ.get("DEVICE", device)
    print(f"device={device} ckpt={ckpt}", flush=True)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CompositeConfig(**{k: ck["config"][k] for k in CompositeConfig.__dataclass_fields__ if k in ck["config"]})
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M param composite", flush=True)

    problems, tok = load_gsm8k_probe_problems(n=n_eval)
    print(f"loaded {len(problems)} held-out problems", flush=True)

    # Anti-rep defaults match K.2 for fair comparison
    ar_kw = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3)
    diff_kw = dict(diff_temperature=0.8, diff_top_p=0.9, diff_repetition_penalty=1.15)

    out = {"ckpt": str(ckpt), "n_eval": n_eval, "k_ar": k_ar, "k_diff": k_diff,
           "n_diff_steps": n_diff_steps, "configs": {}}

    for T in conf_thresholds:
        nlls = []
        wall = []
        loops = 0
        for p in problems:
            t0 = time.time()
            r = gen_starr_intra_diff(model, p["prompt_tokens"], k_ar, k_diff,
                                      n_diff_steps, T, ar_kw, diff_kw)
            wall.append(time.time() - t0)
            gen = r["gen"]
            gold = p["gold_answer_tokens"]
            # Per-token NLL on gold (matches probe_diff_draft_ar_refill behavior)
            n_compare = min(len(gen), len(gold), 128)
            seq_nll = 0.0
            cnt = 0
            for t_idx in range(n_compare):
                ctx = p["prompt_tokens"] + gen[:t_idx]
                if len(ctx) == 0:
                    continue
                idx = torch.tensor([ctx[-cfg.block_size:]], dtype=torch.long, device=device)
                logits = model(idx, mode="ar")[0, -1, :].float()
                log_probs = F.log_softmax(logits, dim=-1)
                nll = float(-log_probs[gold[t_idx]].item())
                seq_nll += nll
                cnt += 1
            if cnt > 0:
                nlls.append(seq_nll / cnt)
            # Loop check (1/2/3-gram repeated 3+ times in tail)
            tail = gen[-20:]
            is_loopy = False
            for n in (1, 2, 3):
                for i in range(max(0, len(tail) - n * 3)):
                    gram = tail[i:i+n]
                    if all(tail[i+k*n:i+(k+1)*n] == gram for k in range(3)):
                        is_loopy = True
                        break
                if is_loopy:
                    break
            if is_loopy:
                loops += 1

        mean_nll = float(np.mean(nlls)) if nlls else float("nan")
        cfg_label = f"starr_intra_diff_pct{int(T)}"
        out["configs"][cfg_label] = {
            "threshold": T,
            "n": len(nlls),
            "mean_nll_on_gold": round(mean_nll, 4),
            "mean_wall_s": round(float(np.mean(wall)), 3) if wall else 0.0,
            "loop_rate": round(loops / max(1, len(problems)), 3),
        }
        print(f"[{cfg_label}] NLL={mean_nll:.3f} wall={np.mean(wall):.2f}s loop={loops}/{len(problems)}",
              flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
