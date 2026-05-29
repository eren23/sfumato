"""Phase P.4b — steering CONTROLS (the causal-validation the review flagged).

P.4 showed that steering an AR-specific SAE feature changes generation.
But divergence-from-baseline is not, by itself, evidence the *feature
direction* matters: any large enough perturbation moves generation. The
control holds magnitude fixed (unit direction x k) and asks whether the
AR-feature direction causes MORE divergence than:
  (a) random unit directions (the null — generic perturbation), and
  (b) a diff-mode feature direction.

If AR-feature divergence >> random, the effect is direction-specific
(supports the causal reading). If AR-feature ~= random, the steering is
generic and P.4 stays merely suggestive (we report that honestly).

Metric: mean over prompts of (1 - token-equality vs unsteered baseline),
greedy AR generation, matched k.

ENV: FEATURE_ID=8435  STEER_K=4.0  MAX_NEW=48  N_RANDOM=4
"""
from __future__ import annotations

import os
import sys
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from e5.interp.load_model import load_composite_for_interp
from e5.interp.p3_per_prompt_analysis import load_sae
from e5.interp.p4_steering import generate_ar


def divergence(base, steered):
    n = min(len(base), len(steered))
    if n == 0:
        return 0.0
    same = sum(1 for a, b in zip(base[:n], steered[:n]) if a == b) / n
    return 1.0 - same


def main():
    feat_id = int(os.environ.get("FEATURE_ID", "8435"))
    steer_k = float(os.environ.get("STEER_K", "4.0"))
    max_new = int(os.environ.get("MAX_NEW", "48"))
    n_random = int(os.environ.get("N_RANDOM", "4"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[ctrl] device={device} feat={feat_id} k={steer_k} n_random={n_random}", flush=True)

    _, raw_model, _ = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt", device=device)
    sae_ar, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt", device)
    sae_di, _ = load_sae(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt", device)

    d_model = sae_ar.W_dec.shape[1]

    def unit(v):
        return (v / v.norm().clamp_min(1e-6)).to(device)

    dirs = {}
    dirs["AR-feature #%d" % feat_id] = unit(sae_ar.W_dec[feat_id].detach())
    # diff-mode feature direction (same index, diff SAE)
    dirs["diff-feature #%d" % feat_id] = unit(sae_di.W_dec[feat_id].detach())
    # random unit directions (matched magnitude); fixed seeds for reproducibility
    for r in range(n_random):
        g = torch.Generator(device="cpu").manual_seed(1000 + r)
        rv = torch.randn(d_model, generator=g)
        dirs["random #%d" % r] = unit(rv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompts = [
        "Question: Janet's ducks lay 16 eggs per day. How much does she make daily?\nAnswer:",
        "The Roman Empire under Trajan extended from",
        "Let x = 3 + 4 * 2. Then x = ",
        "The recipe calls for three cups of flour and",
        "In 1969 the first humans landed on the",
        "She counted the marbles: 12 red, 8 blue, and",
    ]

    # Baselines once per prompt
    base = {}
    for i, p in enumerate(prompts):
        ids = tok.encode(p, add_special_tokens=False)
        base[i] = (ids, generate_ar(raw_model, ids, max_new, None, 0.0, mode="ar", device=device))

    results = {}
    for name, vec in dirs.items():
        divs = []
        for i, p in enumerate(prompts):
            ids, b = base[i]
            s = generate_ar(raw_model, ids, max_new, vec, steer_k, mode="ar", device=device)
            divs.append(divergence(b, s))
        results[name] = divs
        print(f"[ctrl] {name:22s} mean div = {st.mean(divs):.3f}", flush=True)

    # Aggregate random
    rand_means = [st.mean(results[k]) for k in results if k.startswith("random")]
    ar_mean = st.mean(results["AR-feature #%d" % feat_id])
    di_mean = st.mean(results["diff-feature #%d" % feat_id])
    rand_mean = st.mean(rand_means)
    rand_sd = st.stdev(rand_means) if len(rand_means) > 1 else 0.0

    print("\n================= P.4b STEERING CONTROL =================")
    print(f"  AR-feature #{feat_id}   mean divergence = {ar_mean:.3f}")
    print(f"  diff-feature #{feat_id} mean divergence = {di_mean:.3f}")
    print(f"  random dirs (n={n_random}) mean divergence = {rand_mean:.3f} ± {rand_sd:.3f}")
    verdict = ("AR-feature > random => direction-specific (supports causal reading)"
               if ar_mean > rand_mean + rand_sd else
               "AR-feature ~= random => generic perturbation (P.4 stays suggestive)")
    print(f"  VERDICT: {verdict}")
    print("=========================================================")

    import json
    out = REPO_ROOT / "e5/interp/results/p4_steering_controls.json"
    out.write_text(json.dumps({
        "feature_id": feat_id, "steer_k": steer_k, "max_new": max_new,
        "n_prompts": len(prompts),
        "ar_feature_mean_div": ar_mean, "diff_feature_mean_div": di_mean,
        "random_mean_div": rand_mean, "random_sd": rand_sd,
        "per_direction": {k: round(st.mean(v), 4) for k, v in results.items()},
        "verdict": verdict,
    }, indent=2))
    print(f"[ctrl] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
