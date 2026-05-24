"""Phase P.3 — per-prompt cross-head feature analysis (lighter alternative
to circuit-tracer attribution graphs).

For a battery of prompts spanning GSM8K math, prose, and explicit math
mode-switch positions, we:
  1. Run F10 in BOTH modes, capture ln_f activations
  2. Encode each through the AR-head SAE and the diff-head SAE
  3. Identify the top-firing features in each mode at each token
  4. Highlight "shared" features (high cross-head cosine, found in
     Phase P.2 top-20) and "specialised" features
  5. Report which features fire on which tokens in each mode

Output:
  e5/interp/results/p3_per_prompt/
    summary.md
    per_prompt/{prompt_label}.md

Runs locally on Mac MPS. No pod needed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from e5.interp.load_model import load_composite_for_interp
from e5.interp.topk_sae import TopKSAE
from e5.interp.train_saes import attach_hook


def load_sae(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sae = TopKSAE(d_in=cfg["d_in"], d_features=cfg["d_features"], k=cfg["k"])
    sae.load_state_dict(ck["state_dict"])
    sae.to(device).train(False)
    return sae, cfg


def collect_ln_f(raw_model, idx, mode: str):
    store = {"act": None}
    h = raw_model.ln_f.register_forward_hook(
        lambda m, i, o: store.__setitem__("act", o if not isinstance(o, tuple) else o[0])
    )
    try:
        with torch.no_grad():
            raw_model(idx, mode=mode)
    finally:
        h.remove()
    return store["act"][0].detach().float()  # (T, d)


def main():
    out_dir = Path(REPO_ROOT / "e5/interp/results/p3_per_prompt")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_prompt").mkdir(parents=True, exist_ok=True)
    n_top = int(os.environ.get("N_TOP", "8"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p3] device={device}", flush=True)

    # Load F10 + both SAEs
    sae_ar_path = Path(REPO_ROOT / "e5/interp/saes/ln_f_ar/sae.pt")
    sae_diff_path = Path(REPO_ROOT / "e5/interp/saes/ln_f_diff/sae.pt")
    nn_model, raw_model, ccfg = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt",
        device=device)
    sae_ar, _ = load_sae(sae_ar_path, device)
    sae_diff, _ = load_sae(sae_diff_path, device)
    print(f"[p3] F10 + 2 SAEs loaded", flush=True)

    # Load matched-pair table (P.2 top-20) for "bridge feature" highlighting
    overlap_json = REPO_ROOT / "e5/interp/results/cross_head_overlap.json"
    bridge_pairs = []
    if overlap_json.exists():
        d_ov = json.loads(overlap_json.read_text())
        bridge_pairs = d_ov["top20_matched_pairs"]
    bridge_ar_features = {p["ar_feature"] for p in bridge_pairs}
    bridge_diff_features = {p["diff_feature"] for p in bridge_pairs}
    print(f"[p3] {len(bridge_pairs)} bridge pairs from P.2", flush=True)

    prompts = {
        # GSM8K-style — math reasoning
        "G1_janet_ducks": "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 each. How much does she make daily?\nAnswer:",
        "G2_baker_cookies": "Question: A baker makes 240 cookies. He packs them in boxes of 12 each. How many boxes does he need?\nAnswer:",
        "G3_apples_division": "Question: There are 96 apples to share among 8 children equally. How many apples does each child get?\nAnswer:",
        "G4_age_problem": "Question: Alice is 12 years old. Her brother is 4 years younger. In 5 years, how old will Alice's brother be?\nAnswer:",
        # FineWeb-Edu prose
        "F1_history": "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 AD, stretching from Britain in the north to Mesopotamia in the east.",
        "F2_science": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
        "F3_geography": "The Amazon rainforest, often referred to as the lungs of the planet, produces more than 20 percent of the world's oxygen supply through its dense vegetation.",
        "F4_narrative": "She closed the book slowly, as if the story were a guest she did not want to send away, and looked out at the rain blurring the garden lights.",
        # Explicit math mode (formulas mid-text)
        "M1_simple_arith": "The total cost is 50 - 12 + 7 = ",
        "M2_two_step": "Let x = 3 + 4 * 2. Then x = ",
        "M3_word_to_math": "John has 15 marbles. He gives Mary 4 and Tom 3. He has 15 - 4 - 3 = ",
        "M4_equation_chain": "If a = 5 and b = a + 2 then b = 7. So a + b = 5 + 7 = ",
    }

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    summary_rows = []

    for label, text in prompts.items():
        ids = tok.encode(text, add_special_tokens=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        T = len(ids)

        # Collect ln_f activations in both modes
        act_ar = collect_ln_f(raw_model, idx, "ar")     # (T, 1024)
        act_diff = collect_ln_f(raw_model, idx, "diff")

        # Encode through each head's SAE
        z_ar = sae_ar.encode(act_ar).cpu()              # (T, 16384) sparse
        z_diff = sae_diff.encode(act_diff).cpu()

        # Per-prompt summary stats
        ar_total = z_ar.sum(dim=0)
        diff_total = z_diff.sum(dim=0)
        top_ar = ar_total.topk(n_top).indices.tolist()
        top_diff = diff_total.topk(n_top).indices.tolist()

        # Bridge features that fire in this prompt
        bridge_ar_active = [f for f in top_ar if f in bridge_ar_features]
        bridge_diff_active = [f for f in top_diff if f in bridge_diff_features]

        # Cosine of the residual stream at each token (AR vs diff)
        per_token_cos = F.cosine_similarity(act_ar, act_diff, dim=-1).tolist()
        cos_min = min(per_token_cos)
        cos_min_idx = per_token_cos.index(cos_min)
        cos_min_token = tok.decode([ids[cos_min_idx]])

        # Per-prompt markdown
        out_md = []
        out_md.append(f"# {label}\n")
        out_md.append(f"**Prompt:** `{text[:200]}{'...' if len(text)>200 else ''}`\n")
        out_md.append(f"\n## Mode-divergence (cosine of residual at each token)\n")
        out_md.append(f"- Mean cos(AR, diff) = {sum(per_token_cos)/len(per_token_cos):.3f}")
        out_md.append(f"- Min cos at token #{cos_min_idx} = {cos_min:.3f} (token: `{cos_min_token!r}`)")

        out_md.append(f"\n## Top-{n_top} AR-mode features by total activation\n")
        for f in top_ar:
            tag = " ⛓️ BRIDGE" if f in bridge_ar_features else ""
            top_token_idx = int(z_ar[:, f].argmax())
            top_token = tok.decode([ids[top_token_idx]])
            out_md.append(f"- AR feature **#{f}**  total_act={float(ar_total[f]):.2f}  "
                          f"peaks at pos {top_token_idx} (`{top_token!r}`){tag}")

        out_md.append(f"\n## Top-{n_top} diff-mode features by total activation\n")
        for f in top_diff:
            tag = " ⛓️ BRIDGE" if f in bridge_diff_features else ""
            top_token_idx = int(z_diff[:, f].argmax())
            top_token = tok.decode([ids[top_token_idx]])
            out_md.append(f"- diff feature **#{f}**  total_act={float(diff_total[f]):.2f}  "
                          f"peaks at pos {top_token_idx} (`{top_token!r}`){tag}")

        # Features in AR top-N that are NOT in diff top-N (and vice versa)
        # — these are the mode-specific high-firing features
        ar_set = set(top_ar)
        diff_set = set(top_diff)
        ar_only = ar_set - diff_set
        diff_only = diff_set - ar_set
        out_md.append(f"\n## Mode-specific features (top-{n_top} membership)\n")
        out_md.append(f"- AR-only top features: {sorted(ar_only)}")
        out_md.append(f"- diff-only top features: {sorted(diff_only)}")
        out_md.append(f"- shared top features (same index in both top-{n_top}): {sorted(ar_set & diff_set)}")

        (out_dir / "per_prompt" / f"{label}.md").write_text("\n".join(out_md))

        summary_rows.append({
            "label": label,
            "T": T,
            "mean_cos_ar_diff": float(sum(per_token_cos)/len(per_token_cos)),
            "min_cos_ar_diff": cos_min,
            "n_bridge_ar_in_topN": len(bridge_ar_active),
            "n_bridge_diff_in_topN": len(bridge_diff_active),
            "n_top_shared_indices": len(ar_set & diff_set),
        })

    # Overall summary
    sm = ["# P.3 — per-prompt cross-head feature analysis\n",
          f"12 prompts × 2 modes × ln_f SAEs (k=64, d_features=16384). "
          f"Output: e5/interp/results/p3_per_prompt/per_prompt/*\n",
          "## Summary table\n",
          "| Prompt | T | mean cos(AR,diff) | min cos | bridge AR top-N | bridge diff top-N | top-N shared indices |",
          "|---|---|---|---|---|---|---|",
          ]
    for r in summary_rows:
        sm.append(
            f"| {r['label']} | {r['T']} | {r['mean_cos_ar_diff']:.3f} | "
            f"{r['min_cos_ar_diff']:.3f} | {r['n_bridge_ar_in_topN']} | "
            f"{r['n_bridge_diff_in_topN']} | {r['n_top_shared_indices']} |"
        )
    sm.append("")
    sm.append("## Reading")
    sm.append("- **bridge AR/diff top-N**: how many of P.2's 20 high-cosine matched pairs appear in the top-N firing features for this prompt. Bridge features are candidate \"mode-switch routing tokens\".")
    sm.append("- **top-N shared indices**: how many feature INDICES coincide in AR's top-N and diff's top-N. Low values reinforce P.2's \"modes specialise\" finding; high values would say \"same features fire just at different magnitudes\".")
    sm.append("- **min cos**: token position where AR and diff residual diverge the most. Often a math operator or quantity token.")

    (out_dir / "summary.md").write_text("\n".join(sm))
    print(f"[p3] wrote {out_dir / 'summary.md'}", flush=True)
    print(f"[p3] {len(prompts)} per-prompt reports in {out_dir / 'per_prompt'}", flush=True)

    # Print compact table to stdout
    print("\n=== Per-prompt summary ===", flush=True)
    print(f"{'label':22s} {'T':4s} {'mean_cos':10s} {'min_cos':10s} {'br_ar':6s} {'br_di':6s} {'shared':6s}")
    for r in summary_rows:
        print(f"{r['label']:22s} {r['T']:4d} {r['mean_cos_ar_diff']:10.3f} "
              f"{r['min_cos_ar_diff']:10.3f} {r['n_bridge_ar_in_topN']:6d} "
              f"{r['n_bridge_diff_in_topN']:6d} {r['n_top_shared_indices']:6d}")


if __name__ == "__main__":
    main()
