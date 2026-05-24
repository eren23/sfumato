"""Phase P.4 — causal validation via steering.

Pick a top mode-specific feature from P.3, add a scalar multiple of its
SAE decoder column to F10's ln_f activations during generation, and
compare the generated text to the unsteered baseline.

If the feature really drives mode-specific behaviour:
  - Adding it in AR mode should shift AR outputs noticeably
  - Adding it in diff mode should change diff outputs less

ENV:
  FEATURE_ID=8435     (default — a top AR feature from P.3)
  FEATURE_SAE=ar      (which head's SAE column to use as direction)
  STEER_K=4.0         (scalar magnitude, decoder cols are unit-norm)
  MAX_NEW=64
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from e5.interp.load_model import load_composite_for_interp
from e5.interp.p3_per_prompt_analysis import load_sae


@torch.no_grad()
def generate_ar(raw_model, prompt_ids, max_new, ln_f_steer_vec,
                steer_k, mode="ar", temperature=0.0,
                device="mps"):
    """Greedy AR generation; if ln_f_steer_vec given, add k * vec to ln_f
    output on every forward pass."""
    out = list(prompt_ids)

    hook_handle = None
    if ln_f_steer_vec is not None:
        steer = steer_k * ln_f_steer_vec.to(device)
        def _hook(module, inputs, output):
            return output + steer
        hook_handle = raw_model.ln_f.register_forward_hook(_hook)
    try:
        for _ in range(max_new):
            idx = torch.tensor([out[-raw_model.cfg.block_size :]],
                                dtype=torch.long, device=device)
            logits = raw_model(idx, mode=mode)[0, -1, :].float()
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1).item())
            else:
                nxt = int(logits.argmax(dim=-1).item())
            out.append(nxt)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return out


def main():
    feat_id = int(os.environ.get("FEATURE_ID", "8435"))
    which_sae = os.environ.get("FEATURE_SAE", "ar")
    steer_k = float(os.environ.get("STEER_K", "4.0"))
    max_new = int(os.environ.get("MAX_NEW", "48"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[steer] device={device}  feature={which_sae}/{feat_id}  k={steer_k}",
          flush=True)

    nn_model, raw_model, _ = load_composite_for_interp(
        REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt",
        device=device)
    sae_path = REPO_ROOT / f"e5/interp/saes/ln_f_{which_sae}/sae.pt"
    sae, _ = load_sae(sae_path, device)

    # Steering direction = the SAE's decoder column for that feature
    direction = sae.W_dec[feat_id].detach().to(device)  # (d_in,)
    direction_normed = direction / direction.norm().clamp_min(1e-6)
    print(f"[steer] direction norm = {float(direction.norm()):.3f}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    test_prompts = [
        "Question: Janet's ducks lay 16 eggs per day. How much does she make daily?\nAnswer:",
        "The Roman Empire under Trajan extended from",
        "Let x = 3 + 4 * 2. Then x = ",
    ]

    out_dir = Path(REPO_ROOT / "e5/interp/results/p4_steering")
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# P.4 steering — feature {which_sae}/#{feat_id}  k={steer_k}\n"]

    for i, prompt in enumerate(test_prompts):
        ids = tok.encode(prompt, add_special_tokens=False)
        md.append(f"\n## Prompt {i+1}\n```\n{prompt}\n```\n")

        for mode in ["ar", "diff"]:
            # Baseline (no steering)
            base = generate_ar(raw_model, ids, max_new, None, 0.0,
                                mode=mode, device=device)
            base_text = tok.decode(base[len(ids):])
            md.append(f"\n### {mode} baseline (no steer)\n```\n{base_text}\n```\n")

            # Steered (+k × direction)
            steered = generate_ar(raw_model, ids, max_new, direction_normed,
                                   steer_k, mode=mode, device=device)
            steered_text = tok.decode(steered[len(ids):])
            md.append(f"\n### {mode} STEERED (+{steer_k:.1f} × dir)\n```\n{steered_text}\n```\n")

            # Quick diff metric
            same = sum(1 for a, b in zip(base, steered) if a == b) / max(1, len(base))
            md.append(f"_token-equality with baseline: {same*100:.0f}%_\n")
            print(f"[steer] {mode} {('steered','base')[0 if direction is None else 0]} prompt#{i+1} done", flush=True)

    (out_dir / f"steer_{which_sae}_{feat_id}_k{steer_k}.md").write_text("\n".join(md))
    print(f"\n[steer] wrote {out_dir / f'steer_{which_sae}_{feat_id}_k{steer_k}.md'}", flush=True)


if __name__ == "__main__":
    main()
