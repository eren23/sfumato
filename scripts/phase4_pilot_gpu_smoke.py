"""Phase-4 Direction A mini-pilot GPU smoke (plumbing-only, no optimizer.step).

Verifies the production GPU path the same way the CPU smokes verified
the math:
  1. Load LLaDA + commit-LoRA via the trainer's main_train wiring
     (peft model trainable, _Real wrapper, phase_emb attached).
  2. Set SCHEDULE_RL=1 + EMIT_LOGPROBS=1.
  3. Run M=2 rollouts on 1 prompt via _rollout (no_grad).
  4. Verify rollout texts non-degenerate (>=20 chars, no all-mask).
  5. Run _rescore on each rollout's committed tokens through the live
     model with grad enabled.
  6. Verify gradients flow: sum of log-pi has non-zero grad on
     phase_emb.param AND on at least one LoRA-A weight.
  7. Print a one-line VERDICT: PASS or which sanity rule failed.

Cost: ~$0.10 on a 24GB pod (one model load + 2 rollouts + 6 rescore
forwards). Fits comfortably in 24GB because no optimizer state is
ever materialized.

If this passes, the actual mini-pilot training (optimizer.step over
N=20 prompts × M=4 rollouts × 1 epoch) needs 48GB+ and a separate
dispatch.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    os.environ["SCHEDULE_RL"] = "1"
    os.environ["EMIT_LOGPROBS"] = "1"
    # MERGE_ADAPTER=0 keeps LoRA un-fused during rollouts so the LoRA-A/B
    # weights stay in the autograd graph for the post-rollout re-score
    # forward. (`_enable_commit` calls `model.merge_adapter()` by default,
    # which folds LoRA into the base linear's weight matrix; the merged
    # path is grad-disconnected from the base because base is frozen.)
    os.environ["MERGE_ADAPTER"] = "0"
    # Help with VRAM fragmentation on the 24GB 4090.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    import e4.diff_llada as _diff
    from e4 import phase_emb_state as _pes
    from scripts.train_track2_commit_rl import (
        PhaseEmbedding, _rollout, _rescore_committed_logprobs,
        NUM_SUB_BLOCKS, COMMIT_N_BLOCKS, LORA_R, LORA_LAYERS_TO_TRANSFORM, LORA_TARGETS,
    )

    fail = []
    t_total = time.time()

    print("=" * 64, flush=True)
    print("Phase-4 Direction A — mini-pilot GPU plumbing smoke", flush=True)
    print("=" * 64, flush=True)

    MODEL_NAME = os.environ.get("MODEL_NAME", "GSAI-ML/LLaDA-8B-Instruct")
    COMMIT_REPO = os.environ.get("RESUME_FROM", "eren23/sfumato-llada-commit-v3")
    HF_TOKEN = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    # 1. Load
    t0 = time.time()
    print(f"[1/6] Loading {MODEL_NAME} + commit-LoRA from {COMMIT_REPO}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    base.requires_grad_(False)
    peft_model = PeftModel.from_pretrained(
        base, COMMIT_REPO,
        is_trainable=True, adapter_name="commit", token=HF_TOKEN,
    )
    n_grad_after_peft = sum(1 for p in peft_model.parameters() if p.requires_grad)
    print(f"      DBG after PeftModel.from_pretrained: trainable param tensors = {n_grad_after_peft}", flush=True)
    diff_model = _diff._Real(name=MODEL_NAME, commit_lora_path=COMMIT_REPO)
    diff_model._model = peft_model
    diff_model._tokenizer = tokenizer
    print(f"      load took {time.time() - t0:.1f}s", flush=True)
    print(f"      VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    # 2. Phase emb attach
    print("[2/6] Attaching phase_emb monkey-patch", flush=True)
    phase_emb = PhaseEmbedding(num_phases=NUM_SUB_BLOCKS, lora_rank=LORA_R)
    n_hooked = phase_emb.attach(peft_model)
    expected = len(LORA_LAYERS_TO_TRANSFORM) * len(LORA_TARGETS)
    if n_hooked != expected:
        fail.append(f"phase_emb hooks {n_hooked} != expected {expected}")
    phase_emb.param.data = phase_emb.param.data.to("cuda")
    # Tiny non-zero init so the additive bias actually does something
    # observable when phase_emb gradients are checked below.
    with torch.no_grad():
        phase_emb.param.data.normal_(0.0, 0.01)
    phase_emb.param.requires_grad_(True)
    print(f"      hooks installed: {n_hooked}/{expected}", flush=True)
    n_grad_after_attach = sum(1 for p in peft_model.parameters() if p.requires_grad)
    print(f"      DBG after phase_emb.attach: trainable param tensors = {n_grad_after_attach}", flush=True)

    n_grad_pre_rollout = sum(1 for p in peft_model.parameters() if p.requires_grad)
    print(f"      DBG pre-rollout: trainable param tensors = {n_grad_pre_rollout}", flush=True)

    # 3. Rollout
    # M=1 only on a 24GB GPU — re-score with grad enabled is VRAM-tight,
    # and we backward+free per (rollout, sub-block) below to bound peak.
    M_SMOKE = int(os.environ.get("M_SMOKE", "1"))
    print(f"[3/6] Running M={M_SMOKE} rollout(s) under torch.no_grad()", flush=True)
    prompt = (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?"
    )
    gold = "72"
    t0 = time.time()
    with torch.no_grad():
        rollouts = _rollout(
            diff_model, tokenizer, prompt, gold, m=M_SMOKE, seed=0,
        )
    print(f"      {len(rollouts)} rollouts in {time.time() - t0:.1f}s", flush=True)
    print(f"      VRAM after rollout: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    # 3.5 Enable gradient checkpointing on the wrapped HF base — re-score
    # forward keeps activations across the full sequence, and on a 24GB
    # 4090 the third forward OOMs without checkpointing. Trades ~2x
    # forward time for ~5-10x activation memory savings.
    try:
        peft_model.base_model.model.gradient_checkpointing_enable()
        # Make sure inputs require grad so checkpointing's autograd hook fires.
        peft_model.base_model.model.enable_input_require_grads()
        print("      DBG enabled gradient_checkpointing on base model", flush=True)
    except (AttributeError, ValueError) as e:
        print(f"      DBG gradient_checkpointing not available: {e}", flush=True)

    # 3.5b inspect LoRA layer state
    for name, mod in peft_model.named_modules():
        if "blocks.24.ff_proj" in name and "lora_A" in name and not name.endswith("commit"):
            print(f"      DBG sample lora_A: name={name} type={type(mod).__name__}", flush=True)
            for sub_name, sub in mod.named_children():
                rg = getattr(sub, "weight", None)
                rg_val = rg.requires_grad if rg is not None else None
                print(f"        child {sub_name}: type={type(sub).__name__} weight.requires_grad={rg_val}", flush=True)
            break
    for name, mod in peft_model.named_modules():
        if name.endswith("blocks.24.ff_proj") and hasattr(mod, "merged"):
            print(f"      DBG ff_proj layer: merged={getattr(mod, 'merged', '?')} active_adapters={getattr(mod, 'active_adapters', '?')} disable_adapters={getattr(mod, '_disable_adapters', getattr(mod, 'disable_adapters', '?'))}", flush=True)
            break

    # 3.6 Re-enable LoRA forward + restore grads.
    # `_rollout` -> `denoise_block` -> `_disable_commit` -> `model.disable_adapter_layers()`
    # leaves us in a state where (a) LoRA is bypassed in forward, and (b) LoRA
    # params have `requires_grad=False`. We re-score WITH adapter active and
    # WITH grad, so flip both back on.
    peft_model.enable_adapter_layers()
    peft_model.set_adapter("commit")
    n_restored = 0
    for name, p in peft_model.named_parameters():
        if ".lora_" in name and ".commit" in name:
            p.requires_grad_(True)
            n_restored += 1
    print(f"      DBG post-rollout restore: re-enabled grads on {n_restored} LoRA params", flush=True)
    # Re-inspect AFTER restore.
    for name, mod in peft_model.named_modules():
        if "blocks.24.ff_proj" in name and "lora_A" in name and not name.endswith("commit"):
            for sub_name, sub in mod.named_children():
                rg = getattr(sub, "weight", None)
                rg_val = rg.requires_grad if rg is not None else None
                print(f"      DBG POST-RESTORE child {sub_name}: weight.requires_grad={rg_val}", flush=True)
            break
    for name, mod in peft_model.named_modules():
        if name.endswith("blocks.24.ff_proj") and hasattr(mod, "merged"):
            print(f"      DBG POST-RESTORE ff_proj: merged={getattr(mod, 'merged', '?')} active_adapters={getattr(mod, 'active_adapters', '?')} disable_adapters={getattr(mod, '_disable_adapters', getattr(mod, 'disable_adapters', '?'))}", flush=True)
            break

    # 4. Sanity-check texts
    for i, res in enumerate(rollouts):
        committed_per_sb, lp_per_sb, text, reward = res
        n_text = len(text)
        n_committed_total = sum(len(c) for c in committed_per_sb)
        first_sub_blocks_observed = [_pes.get_sub_block()]
        print(
            f"      rollout {i}: text_len={n_text} "
            f"committed_total={n_committed_total} reward={reward:.0f} "
            f"text_head={text[:80]!r}",
            flush=True,
        )
        if n_text < 20:
            fail.append(f"rollout {i} text too short: {n_text} chars")
        if n_committed_total < 32:
            fail.append(f"rollout {i} committed_total too low: {n_committed_total}")

    # 5+6. Re-score with grad + per-(rollout, sub-block) backward.
    # On 24GB, accumulating activations across 3 rescore forwards OOMs at
    # rollout 1's first forward. Strategy: rescore one sub-block at a time,
    # backward immediately, free the activation graph, accumulate grads.
    print("[5+6/6] Per-sub-block rescore + backward + grad accumulation", flush=True)
    advantages = [1.0, -1.0][:M_SMOKE]  # +1 for rollout 0; if M=2, -1 for rollout 1
    n_grad_params = sum(1 for p in peft_model.parameters() if p.requires_grad)
    print(f"      DBG peft_model trainable param tensors: {n_grad_params}", flush=True)

    # Inline single-sub-block rescore so we can run forward+backward+free
    # for ONE sub-block at a time; never holds more than one activation
    # graph at peak.
    import torch.nn.functional as F
    from e4 import phase_emb_state as _pes
    SUB_BLOCK_LEN = 32
    _MASK_ID = _diff._LLADA_MASK_ID

    def _pad(ids, n):
        ids = list(ids)
        return ids[:n] if len(ids) >= n else ids + [_MASK_ID] * (n - len(ids))

    n_backward = 0
    total_loss = 0.0
    sample_lp_grad_fn = None
    for i, res in enumerate(rollouts):
        committed_per_sb = res.committed_tokens
        adv = advantages[i] if i < len(advantages) else 1.0
        # Pre-build chat-templated prompt once per rollout.
        messages = [
            {"role": "system", "content": _diff._DENOISE_SYS},
            {"role": "user", "content": prompt},
        ]
        prompt_token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        prompt_len = len(prompt_token_ids)
        padded_per_sb = [
            _pad(committed_per_sb[sb] if sb < len(committed_per_sb) else [], SUB_BLOCK_LEN)
            for sb in range(NUM_SUB_BLOCKS)
        ]
        for t in range(1, COMMIT_N_BLOCKS + 1):
            committed_t = list(committed_per_sb[t]) if t < len(committed_per_sb) else []
            if not committed_t:
                continue
            seq = list(prompt_token_ids)
            for sb in range(NUM_SUB_BLOCKS):
                seq.extend(padded_per_sb[sb] if sb < t else [_MASK_ID] * SUB_BLOCK_LEN)
            full_ids = torch.tensor([seq], dtype=torch.long, device="cuda")
            _pes.force_set_sub_block(t)
            try:
                outputs = peft_model(full_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                sb_start = prompt_len + t * SUB_BLOCK_LEN
                n_committed = len(committed_t)
                sb_logits = logits[0, sb_start : sb_start + n_committed]
                log_p = F.log_softmax(sb_logits, dim=-1)
                committed_ids = torch.tensor(committed_t, dtype=torch.long, device="cuda")
                token_lp = log_p.gather(1, committed_ids.unsqueeze(1)).squeeze(1)
                if sample_lp_grad_fn is None:
                    sample_lp_grad_fn = (
                        token_lp.requires_grad,
                        str(type(token_lp.grad_fn).__name__),
                    )
                sb_loss = -(token_lp.sum()) * adv
                total_loss += float(sb_loss)
                sb_loss.backward()
                n_backward += 1
            finally:
                _pes.clear()
            del outputs, logits, sb_logits, log_p, token_lp, sb_loss, full_ids, committed_ids
            torch.cuda.empty_cache()
    print(
        f"      total synthetic policy loss = {total_loss:.4f}, n_backward = {n_backward}",
        flush=True,
    )
    print(f"      sample_lp grad info: requires_grad/grad_fn = {sample_lp_grad_fn}", flush=True)
    print(f"      VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    if sample_lp_grad_fn is None or not sample_lp_grad_fn[0]:
        fail.append("rescored log-probs have requires_grad=False — autograd path broken")

    # Phase_emb gradient check.
    pe_grad = phase_emb.param.grad
    pe_grad_l1 = (
        pe_grad.abs().sum().item() if pe_grad is not None else 0.0
    )
    print(f"      phase_emb.param.grad L1 = {pe_grad_l1:.4e}", flush=True)
    if pe_grad is None or pe_grad_l1 == 0.0:
        fail.append("phase_emb.param has no gradient after backward")
    else:
        # Per-row L1 — we expect rows for t in {1,2,3} to be non-zero,
        # row 0 to be zero (commit-LoRA OFF).
        rows_l1 = pe_grad.abs().sum(dim=1).tolist()
        print(f"      phase_emb per-t grad L1: {[f'{x:.3e}' for x in rows_l1]}", flush=True)
        if rows_l1[0] != 0.0:
            fail.append(f"phase_emb t=0 grad expected 0, got {rows_l1[0]}")
        for t in range(1, 4):
            if rows_l1[t] == 0.0:
                fail.append(f"phase_emb t={t} grad expected non-zero, got 0")

    # LoRA-A grad check: walk the peft model, find any lora_A.<adapter> param,
    # confirm it has a non-zero grad. Adapter name varies (default / commit).
    lora_a_params_with_grad = 0
    sample_lora_grad_l1 = None
    for name, param in peft_model.named_parameters():
        if (".lora_A." in name and param.requires_grad and
                param.grad is not None and param.grad.abs().sum().item() > 0):
            lora_a_params_with_grad += 1
            if sample_lora_grad_l1 is None:
                sample_lora_grad_l1 = param.grad.abs().sum().item()
    print(f"      lora_A params with grad: {lora_a_params_with_grad}", flush=True)
    print(f"      sample lora_A.grad L1: {sample_lora_grad_l1!r}", flush=True)
    if lora_a_params_with_grad == 0:
        fail.append("no lora_A params received gradients")

    print("=" * 64, flush=True)
    print(f"Total wallclock: {time.time() - t_total:.1f}s", flush=True)
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    if fail:
        print(f"VERDICT: FAIL — {len(fail)} sanity rules failed:", flush=True)
        for f in fail:
            print(f"  - {f}", flush=True)
        return 1
    print("VERDICT: PASS — all GPU-plumbing sanity rules green", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
