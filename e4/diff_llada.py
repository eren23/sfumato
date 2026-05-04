"""LLaDA-Instruct diffusion-LM wrapper.

Real path ports the reference sampler from ML-GSAI/LLaDA (Apache-2.0):
  https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

Mock path returns deterministic strings for local CI.

Workstream C (Phase 2) added an optional `step_callback` hook to `_generate`
that fires once per sub-block boundary so the inference visualizer can
intercept state, surface it to the UI, and inject mechanism-switch
directives (continue / AR-extend / cmaj-branch). Default callback is a
no-op pass-through so all existing runner.py callers stay bit-identical.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from e4 import flops as flops_mod


# ── Workstream C callback contract ─────────────────────────────────────────
# `_generate` fires `step_callback(StepState)` once per sub-block boundary
# (after all `steps_per_block` diffusion rounds inside that block complete —
# never mid-block, since rolling back partial commits is too costly). The
# callback returns a `StepDirective` that tells the sampler what to do for
# the NEXT sub-block. Default callback returns continue_llada() to preserve
# existing runner.py semantics bit-for-bit.

@dataclass
class StepState:
    """Snapshot handed to the callback at each sub-block boundary."""

    step_idx: int                # monotonically increasing across sub-blocks
    sub_block: int               # 0..num_blocks-1
    num_blocks: int
    mechanism: str               # "llada" | "ar_extend" | "cmaj_branch"
    tokens_committed: list[int]
    token_strings: list[str]
    positions: list[int]         # absolute positions in the full sequence
    entropy: list[float]         # per-position Shannon entropy (nats)
    top_k_logits: list[list[tuple[int, float]]]
    commit_lora_active: bool
    logit_shift_norm: Optional[float]
    temperature: float
    steps_per_block: int
    wallclock_ms: int
    # Live tensor handle so the AR-extend / cmaj-branch directives can mutate
    # the in-flight x BEFORE the next sub-block runs. Caller must NOT detach.
    x_handle: Any = None
    prompt_len: int = 0
    block_start: int = 0
    block_end: int = 0


@dataclass
class StepDirective:
    """What the callback wants the sampler to do for the next sub-block.

    Tagged union via `kind`:
      - "continue_llada" — default; just keep going with normal LLaDA.
      - "switch_to_ar"   — run AR extend for n_tokens, graft into x, then
                           hand control back to LLaDA for remaining blocks.
      - "branch_cmaj"    — fork b parallel LLaDA continuations from the
                           current x; caller picks winner. Sampler returns
                           early after appending branch metadata to state.
      - "stop"           — stop generation here (e.g., user aborted).
    """

    kind: str = "continue_llada"
    n_tokens: int = 0                 # for switch_to_ar
    model_name: str = ""              # for switch_to_ar
    b: int = 0                        # for branch_cmaj
    extra: dict | None = None         # arbitrary passthrough (e.g., text injection override)

    @classmethod
    def continue_llada(cls) -> "StepDirective":
        return cls(kind="continue_llada")

    @classmethod
    def switch_to_ar(cls, n_tokens: int, model_name: str = "") -> "StepDirective":
        return cls(kind="switch_to_ar", n_tokens=n_tokens, model_name=model_name)

    @classmethod
    def branch_cmaj(cls, b: int) -> "StepDirective":
        return cls(kind="branch_cmaj", b=b)

    @classmethod
    def stop(cls) -> "StepDirective":
        return cls(kind="stop")


def _default_step_callback(state: StepState) -> StepDirective:
    """No-op default — preserves bit-identical behavior for existing callers."""
    return StepDirective.continue_llada()


StepCallback = Callable[[StepState], StepDirective]


# ── Batched callback contract (S0 branch batching) ─────────────────────────
# When `denoise_block_batched` is used to run B branches in a single forward
# pass, the per-row StepState above is replaced with a `BatchStepState` that
# carries the live `(B, L+gen)` tensor so the caller (cmaj/cmajc/cmerge in
# runner.py) can vote on partial extracted answers across rows. ESC quorum
# (S1) returns a `BatchStepDirective` with per-row `should_stop` flags;
# pruned rows freeze (their transfer mask becomes all-False) but still
# occupy their batch slot until the outermost loop completes.

@dataclass
class BatchStepState:
    """Per-batch snapshot at sub-block boundary for batched cmaj/cmajc/cmerge."""

    step_idx: int
    sub_block: int
    num_blocks: int
    B: int                              # number of branches
    active: list[bool]                  # which rows are still committing
    x_handle: Any = None                # live (B, L+gen) tensor; do NOT mutate
    prompt_len: int = 0
    block_start: int = 0
    block_end: int = 0
    commit_lora_active: bool = False
    temperature: float = 0.0
    steps_per_block: int = 0
    wallclock_ms: int = 0


@dataclass
class BatchStepDirective:
    """Per-row directive for the next sub-block in batched mode."""

    should_stop: list[bool] = field(default_factory=list)

    @classmethod
    def continue_all(cls, B: int) -> "BatchStepDirective":
        return cls(should_stop=[False] * B)


def _default_batch_step_callback(state: "BatchStepState") -> "BatchStepDirective":
    """No-op default — preserves bit-identical (per-row) behavior."""
    return BatchStepDirective.continue_all(state.B)


BatchStepCallback = Callable[["BatchStepState"], "BatchStepDirective"]


_LLADA_MASK_ID = 126336  # LLaDA's [MASK] token id
_LLADA_EOT_ID = 126081   # LLaDA's <|endoftext|> id (for optional masking, see LLaDA App. B.4)

_DENOISE_SYS = (
    "You are a careful math tutor. Think step by step about the problem "
    "below; show numeric work; end with 'Answer: <number>'."
)


@dataclass
class _Mock:
    name: str
    block_len: int = 256
    # Mock semi-AR schedule mirrors the real one so the visualizer renders
    # a 4×32 grid in MOCK_MODELS=1.
    gen_length: int = 128
    sub_block_length: int = 32

    def denoise_block(
        self,
        prompt: str,
        k_steps: int,
        seed: int = 0,
        temperature: float = 0.0,
        apply_commit: bool = False,
        commit_n_blocks: int = 1,
        step_callback: Optional[StepCallback] = None,
    ) -> tuple[str, int]:
        # apply_commit is a no-op in mock mode (kept so the runner can pass it).
        h = hashlib.sha256(
            f"{prompt}|{k_steps}|{seed}|{temperature}|{int(apply_commit)}".encode()
        ).hexdigest()[:8]
        tag = "+commit" if apply_commit else ""

        # If a step_callback was supplied (visualizer mode), synthesize
        # 4 deterministic sub-block StepStates so the UI / trace path is
        # exercised end-to-end without a GPU.
        if step_callback is not None:
            num_blocks = self.gen_length // self.sub_block_length
            commit_n = max(1, min(commit_n_blocks, num_blocks))
            first_commit = num_blocks - commit_n
            # Try to compute the actual answer for the synthetic mock problem
            # so block 3's "#### <num>" tokens line up with the gold the
            # harness reports. Mock problems are "Mock problem N: 2 + N = ?"
            # → answer = N + 2. Fall back to "<num>" if the prompt doesn't
            # match (e.g. real-mode caller passes step_callback to the mock).
            import re as _re
            m = _re.search(r"(\d+)\s*\+\s*(\d+)\s*=\s*\?", prompt)
            answer_str = str(int(m.group(1)) + int(m.group(2))) if m else "<num>"
            answer_digits = list(answer_str) if answer_str != "<num>" else ["<num>"]
            # Generic per-block reasoning vocabulary so cells look like a CoT
            # trace without committing to a specific problem. Block 0 = setup,
            # block 1 = arithmetic, block 2 = consolidation, block 3 = final
            # answer span (with the actual computed sum embedded).
            block_vocab = [
                ["Let", "'s", "think", "step", "by", "step", ".", "We", "are",
                 "given", "a", "math", "problem", "with", "two", "numbers",
                 "to", "combine", ".", "Identify", "the", "operands", ",",
                 "apply", "the", "operator", ",", "and", "report", "the",
                 "value", "."],
                ["Step", "1", ":", "read", "the", "operands", ".", "Step", "2",
                 ":", "apply", "the", "operator", "(", "addition", ")", ".",
                 "Step", "3", ":", "compute", "the", "sum", ".", "Sum", "=",
                 "operand_a", "+", "operand_b", ".", "Carry", "."],
                ["The", "result", "is", "the", "arithmetic", "sum", "of", "the",
                 "two", "values", ".", "Verify", "by", "re", "-", "adding", "in",
                 "the", "opposite", "order", "(", "commutative", ")", ".",
                 "Both", "directions", "agree", ".", "Answer", "is", "stable",
                 "."],
                ["Final", "answer", ":", "the", "sum", "is", *answer_digits,
                 ".", "####", *answer_digits, " ", " ", " ", " ", " ", " ", " ",
                 " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
                 " ", " "][:32],
            ]
            # Per-block entropy band: early blocks uncertain, later blocks more
            # confident; commit-LoRA-active blocks pulled even lower (the whole
            # point of the commit adapter — sharpen final-block answer logits).
            band_no_commit = [(0.9, 1.6), (0.6, 1.2), (0.4, 0.9), (0.3, 0.7)]
            band_commit    = [(0.9, 1.6), (0.4, 0.8), (0.25, 0.55), (0.10, 0.30)]
            rng_seed = seed
            for b_idx in range(num_blocks):
                rng_seed = (rng_seed * 1103515245 + 12345) & 0xFFFFFFFF
                positions = list(
                    range(
                        len(prompt) + b_idx * self.sub_block_length,
                        len(prompt) + (b_idx + 1) * self.sub_block_length,
                    )
                )
                tokens = [(rng_seed + p) & 0xFFFF for p in positions]
                vocab = block_vocab[b_idx % len(block_vocab)]
                strings = [vocab[i % len(vocab)] for i in range(len(positions))]
                commit_active = bool(apply_commit) and (b_idx >= first_commit)
                lo, hi = (band_commit if commit_active else band_no_commit)[b_idx]
                ent = [
                    lo + (hi - lo) * (((p * 2654435761) & 0xFF) / 255.0)
                    for p in positions
                ]
                topk = [
                    [(t, 0.6), (t + 1, 0.2), (t + 2, 0.1), (t + 3, 0.06), (t + 4, 0.04)]
                    for t in tokens
                ]
                state = StepState(
                    step_idx=b_idx,
                    sub_block=b_idx,
                    num_blocks=num_blocks,
                    mechanism="llada",
                    tokens_committed=tokens,
                    token_strings=strings,
                    positions=positions,
                    entropy=ent,
                    top_k_logits=topk,
                    commit_lora_active=commit_active,
                    logit_shift_norm=None,
                    temperature=float(temperature),
                    steps_per_block=max(k_steps // num_blocks, 1),
                    wallclock_ms=10,
                    x_handle=None,
                    prompt_len=len(prompt),
                    block_start=positions[0] if positions else 0,
                    block_end=(positions[-1] + 1) if positions else 0,
                )
                directive = step_callback(state)
                if directive is None:
                    directive = StepDirective.continue_llada()
                if directive.kind in ("stop", "branch_cmaj"):
                    break

        text = f"Mock diffusion CoT (k={k_steps}{tag}) hash={h}"
        used = flops_mod.llada_forward_flops(
            name=self.name, n_tokens=self.block_len, n_steps=k_steps
        )
        return text, used

    def denoise_block_batched(
        self,
        prompt: str,
        k_steps: int,
        seeds: list[int],
        temperature: float = 0.0,
        apply_commit: bool = False,
        commit_n_blocks: int = 1,
        step_callback: Optional["BatchStepCallback"] = None,
    ) -> list[tuple[str, int]]:
        """Mock-mode batched denoise — pass-through to per-seed denoise_block.

        Mock has no batched semantics to optimise; this exists so the
        runner.py cmaj/cmajc/cmerge codepath can drive batched API uniformly
        in MOCK_MODELS=1 smoke tests. ESC/step_callback in mock mode is
        ignored (mock has no live tensor to vote on).
        """
        return [
            self.denoise_block(
                prompt=prompt,
                k_steps=k_steps,
                seed=int(s),
                temperature=temperature,
                apply_commit=apply_commit,
                commit_n_blocks=commit_n_blocks,
                step_callback=None,
            )
            for s in seeds
        ]


def _add_gumbel_noise(logits, temperature: float):
    import torch  # type: ignore

    if temperature <= 0.0:
        return logits.to(torch.float64)
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_(min=1e-20, max=1.0)
    gumbel = -torch.log(-torch.log(noise))
    return logits / temperature + gumbel


def _add_gumbel_noise_batched(logits, temperature: float, generators):
    """Batched gumbel noise with per-row torch.Generator for branch determinism.

    `logits` is (B, L, V). `generators` is a list of B torch.Generator on the
    same device as logits. Each row b gets noise drawn from generators[b] so
    branches stay independently seeded across calls.

    Memory-wise, the naive `torch.stack` of B fp64 noise tensors holds
    `B × L × V × 8` bytes simultaneously which OOMs on a 48GB A6000 at
    B=5, L+gen≈280, V≈126K (peak ≈1.4 GB just for noise on top of the
    8B-param bf16 model and per-step activations). We instead allocate one
    output tensor and write each row's gumbel-corrected logits in place,
    so peak noise allocation is **one row** rather than B.
    """
    import torch  # type: ignore

    if temperature <= 0.0:
        return logits.to(torch.float64)
    logits_f64 = logits.to(torch.float64)
    B = logits_f64.shape[0]
    out = torch.empty_like(logits_f64)
    inv_t = 1.0 / float(temperature)

    if generators is None or len(generators) != B:
        # Fallback: global RNG (loses per-branch determinism but won't crash).
        for b in range(B):
            noise = torch.empty(
                logits_f64.shape[1:],
                dtype=torch.float64,
                device=logits_f64.device,
            ).uniform_(0.0, 1.0).clamp_(min=1e-20, max=1.0)
            torch.log_(noise)
            torch.neg_(noise)
            torch.log_(noise)
            torch.neg_(noise)  # noise now == -log(-log(uniform))
            out[b] = logits_f64[b] * inv_t + noise
            del noise
    else:
        for b in range(B):
            noise = torch.empty(
                logits_f64.shape[1:],
                dtype=torch.float64,
                device=logits_f64.device,
            ).uniform_(0.0, 1.0, generator=generators[b]).clamp_(min=1e-20, max=1.0)
            torch.log_(noise)
            torch.neg_(noise)
            torch.log_(noise)
            torch.neg_(noise)
            out[b] = logits_f64[b] * inv_t + noise
            del noise
    return out


def _os_get_hf_token():
    """Read HF token from any of the standard env names. Returns None when unset."""
    import os as _os
    return (
        _os.environ.get("HF_TOKEN")
        or _os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _num_transfer_tokens(mask_index, steps: int):
    """How many masked positions to commit at each diffusion step.

    Splits the total number of masks across `steps` rounds as evenly as
    possible. Returns a (B, steps) int tensor.
    """
    import torch  # type: ignore

    n_mask = mask_index.sum(dim=1, keepdim=True)
    base = n_mask // steps
    remainder = n_mask % steps
    out = base.repeat(1, steps)
    for b in range(out.shape[0]):
        out[b, : int(remainder[b].item())] += 1
    return out.to(torch.long)


@dataclass
class _Real:
    """LLaDA-Instruct semi-AR sampler.

    Inference flow (matches reference generate.py):
      1. apply chat template -> token ids
      2. append `gen_length` mask tokens
      3. for each block of `block_length`, do `steps_per_block = steps // num_blocks`
         denoising rounds, each committing `num_transfer_tokens[i]` of the
         lowest-confidence masks
    """

    name: str
    block_len: int = 128
    gen_length: int = 128
    sub_block_length: int = 32  # internal block_length within gen_length
    # NOTE: LLaDA's reference (chat.py / generate.py) uses steps=128 for
    # gen_length=128 → 32 steps/block. With our gen_length=128, block=32 →
    # 4 sub-blocks; k=32 gives 8 steps/block (the minimum we observed
    # produces fluent output); k=64 gives 16/block; k=128 = reference.

    # Optional LoRA adapters. `lora_path` is the Track-1 base adapter (loaded
    # active under name "base_lora"). `commit_lora_path` is the Track-2 commit
    # adapter (loaded as a NAMED adapter that is DISABLED by default; the
    # runner enables it for the final block when apply_commit=True).
    lora_path: str | None = None
    commit_lora_path: str | None = None

    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # S4: load via Fast-dLLM v1's LLaDAModelLM (custom HF subclass with
        # block-wise KV-cache support). Otherwise fall through to vanilla
        # AutoModel + LLaDA's trust_remote_code modeling.
        from e4 import fast_dllm_adapter as _fdll
        if _fdll.is_enabled():
            try:
                model = _fdll.load_fast_dllm_model(
                    self.name, dtype=dtype, device=device,
                    hf_token=_os_get_hf_token(),
                )
            except ImportError as _e:
                raise RuntimeError(
                    f"FAST_DLLM=1 set but Fast-dLLM model load failed: {_e}"
                ) from _e
        else:
            # NOTE: LLaDA's custom modeling code (trust_remote_code) targets
            # transformers ≤ 4.x. Pin transformers==4.46.3 on the pod.
            model = AutoModel.from_pretrained(
                self.name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
            model.requires_grad_(False)

        # Apply LoRA if requested. PEFT adapter switching is slow per-call
        # (it walks every Linear layer), so the runner only switches at most
        # once per denoise_block invocation — never per diffusion step.
        # PEFT's PeftModel.from_pretrained doesn't auto-discover HF tokens
        # the way datasets.load_dataset does — pass it explicitly so private
        # adapter repos load. Read HF_TOKEN, HUGGINGFACE_HUB_TOKEN, or
        # HUGGING_FACE_HUB_TOKEN.
        import os as _os

        _hf_token = (
            _os.environ.get("HF_TOKEN")
            or _os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if self.lora_path:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(
                model,
                self.lora_path,
                is_trainable=False,
                adapter_name="base_lora",
                token=_hf_token,
            )
            if self.commit_lora_path:
                # Load commit as a NAMED adapter, then switch back to base_lora
                # so commit is disabled by default.
                model.load_adapter(
                    self.commit_lora_path,
                    adapter_name="commit",
                    is_trainable=False,
                    token=_hf_token,
                )
                model.set_adapter("base_lora")
        elif self.commit_lora_path:
            # Commit-only path: no Track-1 base LoRA. We still want commit
            # disabled by default, so we load it as the only adapter and
            # call disable_adapter_layers(); the runner re-enables it for
            # the final block.
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(
                model,
                self.commit_lora_path,
                is_trainable=False,
                adapter_name="commit",
                token=_hf_token,
            )
            model.disable_adapter_layers()

        # NOTE: S4 model-class swap happens above in the AutoModel-vs-Fast-dLLM
        # branch. PEFT adapters wrap whichever underlying model was chosen.

        self._model = model

    def _generate(
        self,
        prompt_ids,  # torch.LongTensor (1, L)
        steps: int,
        temperature: float,
        commit_last_block: bool = False,
        commit_n_blocks: int = 1,
        step_callback: Optional[StepCallback] = None,
    ):
        """Run the semi-AR denoiser.

        If commit_last_block=True, the LAST `commit_n_blocks` sub-blocks are
        denoised with the commit adapter enabled (PEFT switches exactly twice:
        ON at the boundary of the first commit-enabled block, OFF after the
        last block). commit_n_blocks=1 = original "commit only the final block"
        behavior; commit_n_blocks=3 = blocks 2-4 of 4 (the v3 follow-up).

        ``step_callback`` (Workstream C visualizer hook): if provided, fires
        once per sub-block boundary AFTER the block's diffusion rounds complete
        and returns a ``StepDirective`` controlling the next block. Default is
        ``_default_step_callback`` which always returns continue_llada(), so
        existing callers stay bit-identical.
        """
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        import os as _os

        # S4 fastpath: when FAST_DLLM=1, call upstream `generate()` end-to-end
        # (KV-cache + confidence-aware parallel decoding from arXiv:2505.22618).
        # Skips our per-step loop entirely. step_callback is ignored on this
        # path — Fast-dLLM doesn't expose sub-block boundaries; trace-mode
        # users should set FAST_DLLM=0.
        if _os.environ.get("FAST_DLLM", "0") == "1":
            from e4 import fast_dllm_adapter as _fdll
            threshold = _os.environ.get("FAST_DLLM_TAU")
            tau = float(threshold) if threshold else None
            full = _fdll.fast_dllm_generate(
                self._model,
                prompt_ids,
                steps=max(steps, 4),
                gen_length=self.gen_length,
                block_length=self.sub_block_length,
                temperature=temperature,
                threshold=tau,
            )
            return full[:, prompt_ids.shape[1]:]

        cb = step_callback or _default_step_callback

        device = self._model.device  # type: ignore[attr-defined]
        gen_length = self.gen_length
        block_length = self.sub_block_length
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        # Steps must be divisible by num_blocks; round up to nearest multiple.
        if steps % num_blocks != 0:
            steps = ((steps + num_blocks - 1) // num_blocks) * num_blocks
        steps_per_block = steps // num_blocks
        commit_n_blocks = max(1, min(commit_n_blocks, num_blocks))
        first_commit_block = num_blocks - commit_n_blocks

        x = torch.full(
            (1, prompt_ids.shape[1] + gen_length),
            _LLADA_MASK_ID,
            dtype=torch.long,
            device=device,
        )
        x[:, : prompt_ids.shape[1]] = prompt_ids

        b_idx = 0
        step_idx = 0
        while b_idx < num_blocks:
            # Switch adapters exactly once, at the first commit-enabled block.
            if commit_last_block and b_idx == first_commit_block:
                self._enable_commit()

            commit_active_now = bool(
                commit_last_block and b_idx >= first_commit_block
            )

            blk_start = prompt_ids.shape[1] + b_idx * block_length
            blk_end = prompt_ids.shape[1] + (b_idx + 1) * block_length
            block_mask = (x[:, blk_start:blk_end] == _LLADA_MASK_ID)
            n_transfer = _num_transfer_tokens(block_mask, steps_per_block)

            # Per-sub-block trace accumulators (only populated when callback
            # is non-default — but cheap enough to always collect).
            t0 = time.time()
            committed_positions: list[int] = []
            committed_tokens: list[int] = []
            committed_entropy: list[float] = []
            committed_topk: list[list[tuple[int, float]]] = []
            last_logits = None  # for logit_shift_norm if we ever shadow-run base

            for s in range(steps_per_block):
                mask_index = x == _LLADA_MASK_ID
                logits = self._model(x).logits  # type: ignore[attr-defined]

                logits_n = _add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_n, dim=-1)

                p = F.softmax(logits.to(torch.float64), dim=-1)
                conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                # Don't commit positions past the current block.
                conf[:, blk_end:] = float("-inf")

                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, conf, torch.full_like(conf, float("-inf")))

                transfer = torch.zeros_like(x0, dtype=torch.bool)
                k = int(n_transfer[0, s].item())
                if k > 0:
                    _, sel = torch.topk(conf[0], k=k)
                    transfer[0, sel] = True

                    # Record per-position trace data (only used by callback).
                    if step_callback is not None:
                        sel_list = sel.tolist()
                        for pos in sel_list:
                            committed_positions.append(int(pos))
                            committed_tokens.append(int(x0[0, pos].item()))
                            row = p[0, pos]  # (vocab,)
                            ent = float(-(row * (row.clamp(min=1e-20)).log()).sum().item())
                            committed_entropy.append(ent)
                            top_p, top_i = torch.topk(row, k=5)
                            committed_topk.append(
                                [
                                    (int(top_i[j].item()), float(top_p[j].item()))
                                    for j in range(5)
                                ]
                            )

                x[transfer] = x0[transfer]
                last_logits = logits

            # ── sub-block boundary: fire callback ──
            wallclock_ms = int((time.time() - t0) * 1000)
            token_strings: list[str] = []
            if step_callback is not None and self._tokenizer is not None:
                try:
                    token_strings = [
                        self._tokenizer.decode([tid], skip_special_tokens=False)  # type: ignore[attr-defined]
                        for tid in committed_tokens
                    ]
                except Exception:
                    token_strings = ["?"] * len(committed_tokens)

            # S3: populate logit_shift_norm only when commit-LoRA is active and
            # the LOGIT_SHIFT_NORM env knob is on. Costs 1 extra forward.
            shift_norm: Optional[float] = None
            if commit_active_now and last_logits is not None:
                shift_norm = self._maybe_logit_shift_norm(
                    last_logits, x, committed_positions
                )

            state = StepState(
                step_idx=step_idx,
                sub_block=b_idx,
                num_blocks=num_blocks,
                mechanism="llada",
                tokens_committed=committed_tokens,
                token_strings=token_strings,
                positions=committed_positions,
                entropy=committed_entropy,
                top_k_logits=committed_topk,
                commit_lora_active=commit_active_now,
                logit_shift_norm=shift_norm,
                temperature=float(temperature),
                steps_per_block=steps_per_block,
                wallclock_ms=wallclock_ms,
                x_handle=x,
                prompt_len=int(prompt_ids.shape[1]),
                block_start=blk_start,
                block_end=blk_end,
            )

            directive = cb(state)
            if directive is None:
                directive = StepDirective.continue_llada()

            if directive.kind == "stop":
                break
            elif directive.kind == "continue_llada":
                b_idx += 1
                step_idx += 1
                continue
            elif directive.kind == "switch_to_ar":
                # AR-extend handoff: caller is responsible for grafting AR
                # tokens into `x` BEFORE returning the directive (since the
                # callback holds x_handle). After the graft we just advance
                # to the next block — the AR tokens occupy the masked slots
                # they were written into.
                b_idx += 1
                step_idx += 1
                continue
            elif directive.kind == "branch_cmaj":
                # Caller takes over branching responsibility (server.py forks
                # parallel LLaDA continuations from x_handle). We stop here;
                # the merged result is owned by the caller.
                break
            else:
                # Unknown directive — be conservative and stop.
                break

        # Always reset adapter state at the end so the next call starts clean.
        if commit_last_block:
            self._disable_commit()

        return x[:, prompt_ids.shape[1] :]

    # ── PEFT adapter helpers ──
    # These are no-ops if no commit adapter is configured. Switching is at
    # most once-per-denoise (never per diffusion step) because PEFT walks
    # every Linear layer to flip adapters and is not free.
    #
    # S2 merge-on-toggle: after switching to the commit adapter, call
    # `merge_adapter()` so subsequent forward passes use a pre-fused
    # `W_merged = W_base + α/r · B @ A` instead of doing two matmuls per
    # Linear per step. `unmerge_adapter()` reverses the math exactly
    # (PEFT's merge is `W += α/r · BA`, unmerge is `W -= α/r · BA`),
    # within bf16 rounding. The merge fires at most twice per
    # `denoise_block` call.
    def _enable_commit(self) -> None:
        if not self.commit_lora_path:
            return
        import os as _os
        model = self._model
        if self.lora_path:
            # Both base + commit loaded as named adapters → switch to commit.
            model.set_adapter("commit")  # type: ignore[attr-defined]
        else:
            # Commit-only → just enable adapter layers (commit is the only one).
            model.enable_adapter_layers()  # type: ignore[attr-defined]
        # S2: pre-fuse the active adapter into the base weights so each
        # subsequent forward is one matmul per Linear instead of two.
        # MERGE_ADAPTER=0 disables the S2 path for paired baseline runs.
        if _os.environ.get("MERGE_ADAPTER", "1") != "1":
            self._commit_merged = False
            return
        try:
            model.merge_adapter()  # type: ignore[attr-defined]
            self._commit_merged = True
        except Exception:
            # PEFT versions before merge_adapter or models without LoRA-on-Linear
            # silently keep the un-merged path.
            self._commit_merged = False

    def _disable_commit(self) -> None:
        if not self.commit_lora_path:
            return
        model = self._model
        # S2: unmerge first so the base weights are restored before we
        # switch back to the base_lora adapter (or disable layers).
        if getattr(self, "_commit_merged", False):
            try:
                model.unmerge_adapter()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._commit_merged = False
        if self.lora_path:
            model.set_adapter("base_lora")  # type: ignore[attr-defined]
        else:
            model.disable_adapter_layers()  # type: ignore[attr-defined]

    # ── S3: logit-shift measurement helper ─────────────────────────────────
    # When LOGIT_SHIFT_NORM=1 in the environment, fires at the LAST step of
    # each commit-active sub-block. Returns the L2 norm of
    # (logits_with_commit - logits_base) over committed positions, or None
    # when disabled / inapplicable. Always involves one extra base-only
    # forward pass — gated to avoid the cost on production runs.
    def _maybe_logit_shift_norm(
        self,
        logits_with_commit,
        x,
        committed_positions: list[int],
    ):
        import os as _os
        if _os.environ.get("LOGIT_SHIFT_NORM", "0") != "1":
            return None
        if not self.commit_lora_path or not getattr(self, "_commit_merged", False):
            return None
        if not committed_positions:
            return 0.0
        import torch  # type: ignore
        model = self._model
        # Temporarily un-merge to get a clean base forward.
        try:
            model.unmerge_adapter()  # type: ignore[attr-defined]
            self._commit_merged = False
            with torch.no_grad():
                logits_base = model(x).logits  # type: ignore[attr-defined]
            # Re-merge so the next step keeps the speed win.
            model.merge_adapter()  # type: ignore[attr-defined]
            self._commit_merged = True
        except Exception:
            return None
        # L2 norm over committed positions of (commit - base) on row 0.
        # For batched mode, caller passes a single row's slice.
        idx = torch.tensor(committed_positions, device=x.device, dtype=torch.long)
        diff = (
            logits_with_commit.index_select(-2, idx)
            - logits_base.index_select(-2, idx)
        )
        return float(diff.norm().item())

    def denoise_block(
        self,
        prompt: str,
        k_steps: int,
        seed: int = 0,
        temperature: float = 0.0,
        apply_commit: bool = False,
        commit_n_blocks: int = 1,
        step_callback: Optional[StepCallback] = None,
    ) -> tuple[str, int]:
        """Denoise. temperature>0 enables stochastic sampling (needed for
        self-consistency / branch ensembles to actually diverge).

        If apply_commit=True AND a commit_lora_path was configured, the LAST
        `commit_n_blocks` sub-blocks of the semi-AR schedule are denoised with
        the commit adapter active. commit_n_blocks=1 (default) reproduces the
        original behavior; commit_n_blocks=3 is the v3 follow-up "commit on
        blocks 2-4 of 4". If no commit adapter is configured, apply_commit is
        a no-op.

        ``step_callback`` is the Workstream C visualizer hook; default None
        preserves bit-identical behavior.
        """
        import torch  # type: ignore

        self._ensure_loaded()
        torch.manual_seed(seed)
        messages = [
            {"role": "system", "content": _DENOISE_SYS},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self._tokenizer(
            prompt_text, return_tensors="pt"
        )["input_ids"].to(self._model.device)  # type: ignore[attr-defined]

        commit_last = bool(apply_commit) and bool(self.commit_lora_path)
        gen_ids = self._generate(
            prompt_ids,
            steps=max(k_steps, 4),
            temperature=temperature,
            commit_last_block=commit_last,
            commit_n_blocks=commit_n_blocks,
            step_callback=step_callback,
        )
        text = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]  # type: ignore[attr-defined]
        used = flops_mod.llada_forward_flops(
            name=self.name, n_tokens=self.gen_length, n_steps=max(k_steps, 4)
        )
        return text.strip(), used

    # ── S0: batched denoise ─────────────────────────────────────────────────
    # Same prompt, B independent stochastic branches in one (B, L+gen) forward
    # pass. Used by runner.py:cmaj/cmajc/cmerge. The legacy `denoise_block`
    # path stays bit-identical because none of its callers touch _generate_batched.
    def denoise_block_batched(
        self,
        prompt: str,
        k_steps: int,
        seeds: list[int],
        temperature: float = 0.7,
        apply_commit: bool = False,
        commit_n_blocks: int = 1,
        step_callback: Optional["BatchStepCallback"] = None,
    ) -> list[tuple[str, int]]:
        """Denoise B branches sharing the same prompt in a single batched call.

        seeds[i] becomes the seed for branch i (independent torch.Generator
        per row → branches diverge). Returns one (text, flops) tuple per
        branch in input order. step_callback (BatchStepCallback) fires once
        per sub-block boundary; ESC quorum directives can flip per-row stop
        flags so pruned rows freeze without aborting the batched matmul.
        """
        import torch  # type: ignore

        self._ensure_loaded()
        B = len(seeds)
        if B == 0:
            return []
        if B == 1:
            # Single-branch path: defer to the legacy code so backcompat is preserved.
            return [
                self.denoise_block(
                    prompt=prompt,
                    k_steps=k_steps,
                    seed=int(seeds[0]),
                    temperature=temperature,
                    apply_commit=apply_commit,
                    commit_n_blocks=commit_n_blocks,
                )
            ]

        messages = [
            {"role": "system", "content": _DENOISE_SYS},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids_single = self._tokenizer(
            prompt_text, return_tensors="pt"
        )["input_ids"].to(self._model.device)  # type: ignore[attr-defined]
        prompt_ids = prompt_ids_single.expand(B, -1).contiguous()

        device = self._model.device  # type: ignore[attr-defined]
        # Per-row Generator. Use device='cuda' if model is on GPU; CPU otherwise.
        gen_device = "cuda" if "cuda" in str(device) else "cpu"
        generators = [
            torch.Generator(device=gen_device).manual_seed(int(s)) for s in seeds
        ]

        commit_last = bool(apply_commit) and bool(self.commit_lora_path)
        gen_ids_batched = self._generate_batched(
            prompt_ids,
            steps=max(k_steps, 4),
            temperature=temperature,
            commit_last_block=commit_last,
            commit_n_blocks=commit_n_blocks,
            generators=generators,
            step_callback=step_callback,
        )
        texts = self._tokenizer.batch_decode(  # type: ignore[attr-defined]
            gen_ids_batched, skip_special_tokens=True
        )
        used_per = flops_mod.llada_forward_flops(
            name=self.name, n_tokens=self.gen_length, n_steps=max(k_steps, 4)
        )
        return [(t.strip(), used_per) for t in texts]

    def _generate_batched(
        self,
        prompt_ids,                     # (B, L) on model device
        steps: int,
        temperature: float,
        commit_last_block: bool = False,
        commit_n_blocks: int = 1,
        generators=None,                # list[torch.Generator] of length B
        step_callback: Optional["BatchStepCallback"] = None,
    ):
        """Batched semi-AR denoiser. See `_generate` for single-row reference."""
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        device = self._model.device  # type: ignore[attr-defined]
        B, L = prompt_ids.shape
        gen_length = self.gen_length
        block_length = self.sub_block_length
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            steps = ((steps + num_blocks - 1) // num_blocks) * num_blocks
        steps_per_block = steps // num_blocks
        commit_n_blocks = max(1, min(commit_n_blocks, num_blocks))
        first_commit_block = num_blocks - commit_n_blocks

        x = torch.full(
            (B, L + gen_length),
            _LLADA_MASK_ID,
            dtype=torch.long,
            device=device,
        )
        x[:, :L] = prompt_ids

        # Per-row "still committing" flag — flipped to False by ESC quorum (S1).
        active = torch.ones(B, dtype=torch.bool, device=device)

        b_idx = 0
        step_idx = 0
        while b_idx < num_blocks:
            if commit_last_block and b_idx == first_commit_block:
                self._enable_commit()
            commit_active_now = bool(
                commit_last_block and b_idx >= first_commit_block
            )

            blk_start = L + b_idx * block_length
            blk_end = L + (b_idx + 1) * block_length
            block_mask = (x[:, blk_start:blk_end] == _LLADA_MASK_ID)
            n_transfer = _num_transfer_tokens(block_mask, steps_per_block)  # (B, S)

            t0 = time.time()
            for s in range(steps_per_block):
                mask_index = x == _LLADA_MASK_ID
                logits = self._model(x).logits  # type: ignore[attr-defined]  # (B, L+gen, V)

                logits_n = _add_gumbel_noise_batched(
                    logits, temperature=temperature, generators=generators
                )
                x0 = torch.argmax(logits_n, dim=-1)
                # Free the fp64 gumbel-corrected logits before softmax allocates
                # another full (B, L, V) tensor.
                del logits_n

                # fp32 softmax instead of fp64 — halves the (B, L, V) allocation
                # and the gather-confidence is a single value per position so
                # fp32 precision is more than sufficient.
                p = F.softmax(logits.float(), dim=-1)
                conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                del p
                conf[:, blk_end:] = float("-inf")

                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(
                    mask_index, conf, torch.full_like(conf, float("-inf"))
                )

                # Per-row topk + commit. Pruned rows (active[bi]=False) skip.
                transfer = torch.zeros_like(x0, dtype=torch.bool)
                for bi in range(B):
                    if not bool(active[bi].item()):
                        continue
                    k = int(n_transfer[bi, s].item())
                    if k > 0:
                        _, sel = torch.topk(conf[bi], k=k)
                        transfer[bi, sel] = True

                x[transfer] = x0[transfer]

            wallclock_ms = int((time.time() - t0) * 1000)

            # Sub-block boundary: fire batched callback if provided.
            if step_callback is not None:
                state = BatchStepState(
                    step_idx=step_idx,
                    sub_block=b_idx,
                    num_blocks=num_blocks,
                    B=B,
                    active=active.tolist(),
                    x_handle=x,
                    prompt_len=int(L),
                    block_start=blk_start,
                    block_end=blk_end,
                    commit_lora_active=commit_active_now,
                    temperature=float(temperature),
                    steps_per_block=steps_per_block,
                    wallclock_ms=wallclock_ms,
                )
                directive = step_callback(state)
                if directive is not None and directive.should_stop:
                    for bi in range(B):
                        if bi < len(directive.should_stop) and directive.should_stop[bi]:
                            active[bi] = False
                    if not bool(active.any().item()):
                        break

            b_idx += 1
            step_idx += 1

        if commit_last_block:
            self._disable_commit()

        return x[:, L:]


def load(
    name: str,
    mock: bool = False,
    block_len: int = 256,
    lora_path: str | None = None,
    commit_lora_path: str | None = None,
) -> _Mock | _Real:
    return _Mock(name=name, block_len=block_len) if mock else _Real(
        name=name,
        block_len=block_len,
        lora_path=lora_path,
        commit_lora_path=commit_lora_path,
    )
