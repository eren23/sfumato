"""Phase P.0 — sanity dump of residual-stream activations.

Given a CompositeLM checkpoint + a prompt, dump:
- residual stream after each block (n_layers tensors, shape (T, d_model))
- post-ln_f hidden state (the input to both heads)
- pre-head_ar logits projection input  (= post-ln_f x.float(), used by AR)
- pre-head_diff projection input       (= post-ln_f, before head_diff_proj)
- pre-head_diff post-projection        (= head_diff_proj(post-ln_f))

Both modes (mode="ar", mode="diff") run separately so we get the
activations the AR head sees vs the diff head sees.

Saves to a .pt file under e5/interp/cache/.

ENV:
  CKPT=path/to/model_slim.pt
  PROMPT="some text..."   (or PROMPT_FILE=path/to/prompt.txt)
  OUT=path/to/dump.pt
  DEVICE=cpu|mps|cuda
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from e5.interp.load_model import load_composite_for_interp, architecture_summary


def dump_activations(ckpt_path, prompt_text, out_path, device="cpu"):
    nn_model, raw_model, cfg = load_composite_for_interp(ckpt_path, device=device)
    print(f"[interp] {architecture_summary(cfg)}")
    print(f"[interp] {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M params")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode(prompt_text, add_special_tokens=False)
    print(f"[interp] prompt {len(ids)} tokens")
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # We use a manual hook approach for robustness across nnsight versions.
    # Hook each block's output (post-residual) and the post-ln_f activation.
    captured = {"blocks_ar": [], "blocks_diff": [], "ln_f_ar": None,
                "ln_f_diff": None, "head_diff_proj_out": None}

    hooks = []

    def make_block_hook(i, store):
        def _hook(module, inputs, output):
            store.append(output.detach().cpu())
        return _hook

    def ln_f_hook_factory(key):
        def _hook(module, inputs, output):
            captured[key] = output.detach().cpu()
        return _hook

    def head_diff_proj_hook(module, inputs, output):
        captured["head_diff_proj_out"] = output.detach().cpu()

    # ---- AR mode ----
    captured["blocks_ar"] = []
    hooks_ar = []
    for i, block in enumerate(raw_model.blocks):
        hooks_ar.append(block.register_forward_hook(make_block_hook(i, captured["blocks_ar"])))
    hooks_ar.append(raw_model.ln_f.register_forward_hook(ln_f_hook_factory("ln_f_ar")))
    with torch.no_grad():
        _ = raw_model(idx, mode="ar")
    for h in hooks_ar:
        h.remove()

    # ---- Diff mode ----
    captured["blocks_diff"] = []
    hooks_diff = []
    for i, block in enumerate(raw_model.blocks):
        hooks_diff.append(block.register_forward_hook(make_block_hook(i, captured["blocks_diff"])))
    hooks_diff.append(raw_model.ln_f.register_forward_hook(ln_f_hook_factory("ln_f_diff")))
    if hasattr(raw_model, "head_diff_proj") and not isinstance(
            raw_model.head_diff_proj, torch.nn.Identity):
        hooks_diff.append(raw_model.head_diff_proj.register_forward_hook(head_diff_proj_hook))
    with torch.no_grad():
        _ = raw_model(idx, mode="diff")
    for h in hooks_diff:
        h.remove()

    out = {
        "prompt_text": prompt_text,
        "prompt_ids": ids,
        "config": cfg.__dict__,
        "ckpt_path": str(ckpt_path),
        "blocks_ar": captured["blocks_ar"],          # list[Tensor(B, T, d)]
        "blocks_diff": captured["blocks_diff"],
        "ln_f_ar": captured["ln_f_ar"],              # Tensor(B, T, d)
        "ln_f_diff": captured["ln_f_diff"],
        "head_diff_proj_out": captured["head_diff_proj_out"],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[interp] saved {out_path}  ({sum(t.numel() for t in captured['blocks_ar']) * 2 / 1e6:.1f}MB AR + diff)")

    # Sanity report
    print(f"[interp] AR  blocks: {len(captured['blocks_ar'])} tensors, "
          f"each shape={captured['blocks_ar'][0].shape}")
    print(f"[interp] diff blocks: {len(captured['blocks_diff'])} tensors, "
          f"each shape={captured['blocks_diff'][0].shape}")
    print(f"[interp] ln_f_ar  shape={captured['ln_f_ar'].shape}, "
          f"max={captured['ln_f_ar'].abs().max():.3f}")
    print(f"[interp] ln_f_diff shape={captured['ln_f_diff'].shape}, "
          f"max={captured['ln_f_diff'].abs().max():.3f}")
    if captured["head_diff_proj_out"] is not None:
        print(f"[interp] head_diff_proj_out shape={captured['head_diff_proj_out'].shape}, "
              f"max={captured['head_diff_proj_out'].abs().max():.3f}")


def main():
    ckpt = os.environ.get("CKPT",
        str(REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt"))
    prompt_file = os.environ.get("PROMPT_FILE")
    if prompt_file:
        prompt = Path(prompt_file).read_text()
    else:
        prompt = os.environ.get("PROMPT",
            "Question: Janet's ducks lay 16 eggs per day. She eats three for "
            "breakfast every morning and bakes muffins for her friends every day "
            "with four. She sells the remainder at the farmers' market daily for "
            "$2 per fresh duck egg. How much in dollars does she make every day "
            "at the farmers' market?\nAnswer:")
    out = os.environ.get("OUT", str(REPO_ROOT / "e5/interp/cache/p0_dump_f10.pt"))
    device = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dump_activations(ckpt, prompt, out, device=device)


if __name__ == "__main__":
    main()
