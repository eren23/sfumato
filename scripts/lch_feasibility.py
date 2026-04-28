"""LCH (Linear Centroids Hypothesis) feasibility spike for LLaDA + PEFT.

Single-problem test: can we compute centroids — Jacobian-vector-product
sums of the input-output map of a sub-component — through the PEFT-wrapped
LLaDA-8B model?

If this returns a finite, non-zero vector with shape matching the target
module's output dim, LCH centroids on this stack are feasible and we can
build the Track 1 mechanism figure (PCA of centroids on ~20 problems × 8
prefix conditions, with vs without Track 1 LoRA).

If this fails (NaN, shape mismatch, autograd error through trust_remote_code,
PEFT adapter-switch corrupting the graph), the LCH figure is not buildable
in <1 day of work and we abandon for the current paper.

Spec from the LCH paper (Walker et al, arxiv 2604.11962):
  centroid(f, x) = sum_i J_i  (Jacobian of f at x, summed across the input dim)

We use a JVP shortcut: J_sum = J @ ones_like(x). torch.autograd.functional.jvp
gives us exactly this without materializing the full Jacobian.

Usage on pod:
  python scripts/lch_feasibility.py --lora_path eren23/sfumato-llada-prefix-robust-v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="eren23/sfumato-llada-prefix-robust-v3")
    parser.add_argument("--diff_model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    args = parser.parse_args()

    import torch  # type: ignore
    from datasets import load_dataset  # type: ignore

    from e4 import diff_llada  # type: ignore

    print(f"[lch] loading {args.diff_model} + lora {args.lora_path}", flush=True)
    model = diff_llada.load(
        args.diff_model, mock=False, lora_path=args.lora_path, commit_lora_path=None
    )
    model._ensure_loaded()
    inner = model._model
    tokenizer = model._tokenizer
    device = inner.device

    # Pick a single problem
    ds = load_dataset("gsm8k", "main", split="test")
    question = ds[0]["question"]
    print(f"[lch] question (idx=0, first 80 chars): {question[:80]}...", flush=True)

    messages = [
        {"role": "system", "content": "You are a careful math tutor."},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    print(f"[lch] input_ids shape: {tuple(input_ids.shape)}", flush=True)

    # We want the Jacobian of the LAST FFN's output wrt its input.
    # Inspect the model module tree to find the last FFN.
    last_ffn = None
    for name, module in inner.named_modules():
        if "blocks.31" in name and ("ff_out" in name or "down_proj" in name):
            last_ffn = (name, module)
    if last_ffn is None:
        # Fall back: find any block.31 module
        for name, module in inner.named_modules():
            if "blocks.31" in name and isinstance(module, torch.nn.Linear):
                last_ffn = (name, module)
                break
    if last_ffn is None:
        print("[lch] FAIL: could not locate last FFN module", file=sys.stderr)
        return 1
    print(f"[lch] target module: {last_ffn[0]}  type={type(last_ffn[1]).__name__}", flush=True)

    # Hook into the module — capture input + use JVP from there.
    captured = {}

    def fwd_hook(mod, inp, out):
        captured["input"] = inp[0].detach()
        captured["output"] = out.detach()
        return None

    h = last_ffn[1].register_forward_hook(fwd_hook)

    # Run a forward pass to populate captured.
    with torch.no_grad():
        _ = inner(input_ids)
    h.remove()

    print(f"[lch] captured input shape: {tuple(captured['input'].shape)}  output: {tuple(captured['output'].shape)}", flush=True)

    # Attempt JVP: compute (∂f/∂x) @ ones_like(x) for the captured slice.
    # We treat the linear module as f(x) = Wx + b.
    x = captured["input"][0:1]  # (1, T, in_dim)
    x = x.to(torch.float32).requires_grad_(True)
    try:
        # Direct JVP through the module
        v = torch.ones_like(x)
        out, jvp_out = torch.autograd.functional.jvp(
            lambda inp: last_ffn[1](inp), (x,), (v,), create_graph=False
        )
        # Sum over the token dim → centroid (averaged across positions)
        centroid = jvp_out.sum(dim=1).squeeze(0)  # (out_dim,)
        norm = centroid.norm().item()
        finite = bool(torch.isfinite(centroid).all().item())
        print(f"[lch] PASS:  centroid shape={tuple(centroid.shape)}  ||centroid||={norm:.4e}  finite={finite}", flush=True)
        if not finite:
            print("[lch] but centroid contains NaN/Inf — feasibility marginal", file=sys.stderr)
            return 1
        if norm < 1e-10:
            print("[lch] but centroid is ~zero — feasibility marginal (might be PEFT/dtype issue)", file=sys.stderr)
            return 1
        print("[lch] LCH centroid extraction is feasible on this stack", flush=True)
        print("[lch] full Track 1 figure (~20 problems × 8 prefixes × 2 conditions) is buildable.", flush=True)
        return 0
    except Exception as exc:
        print(f"[lch] FAIL: JVP raised {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
