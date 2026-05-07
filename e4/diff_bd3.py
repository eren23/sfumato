"""BD3-LMs cross-substrate adapter — SCAFFOLD ONLY.

Phase-4 T2.C scaffold for testing whether sfumato's K2 inverted-U
commit-LoRA finding (cmajc-k3=0.822 GSM8K, +12pp MATH-500) replicates
on a *different* mask-diffusion language model family: BD3-LMs (Block
Diffusion language models, Arrelou et al. ICLR-25, arXiv 2503.09573).

Repo:        https://github.com/kuleshov-group/bd3lms (Apache-2.0)
Checkpoints: kuleshov-group/bd3lm-owt-block_size{4,8,16}
             kuleshov-group/bd3lm-owt-block_size1024-pretrain
PRE_REG:     phase2/spikes/bd3lms-cross-substrate/PRE_REG.md  (commit 998b0b0)

Why this is a scaffold (not a working adapter): BD3-LMs ships ONLY
OpenWebText perplexity pretraining — no math SFT, no PEFT/LoRA on the
custom DiT class (it is NOT a `transformers.AutoModel` subclass), and
the sampler is Hydra-driven (`main.py mode=sample_eval`) with no
programmatic `_generate(prompt_ids, ...)` entrypoint. WebFetch on
2026-05-07 revealed the cost is 3-4× the original $12 + 10-14 day
estimate. See PRE_REG.md "Cost re-estimate (2026-05-07)" section.

What this file does:
  1. Reuses StepState / StepDirective from e4.diff_llada VERBATIM (the
     visualizer + branch-trace tooling key off these exact identities).
  2. Provides a `load(...)` signature mirroring `e4.diff_llada.load(...)`
     so the runner can swap substrates by changing one import line.
  3. Raises NotImplementedError from every method that touches a real
     BD3 forward pass or PEFT toggle, pointing at missing eng prereqs.

TODO(T2.C-eng-prereq): six engineering phases, ~$40-60 + 14-21 days
  Phase 1: fork BD3 Hydra repo, set up local + pod env       3-4 days, $0
  Phase 2: SFT BD3-base on GSM8K-train (~7.5k examples)      2-3 days, $15-25
  Phase 3: add PEFT/LoRA infra to BD3 model class            3-4 days, $0
  Phase 4: train commit-LoRA on BD3-base+SFT                 2-3 days, $5-10
  Phase 5: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0)       1 day,    $2-3
  Phase 6: buffer (debugging + paper write-up)               2-3 days, $10-20
  -------------------------------------------------------------------------
  Total                                                      14-21 days, $40-60

Direction A (schedule-RLHF) is the prioritized Phase-4 bet; T2.C
stays scaffolded but unfunded until budget approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Reuse the Workstream-C callback contract VERBATIM. No re-defining —
# the inference visualizer + branch trace tooling key off these exact
# class identities.
from e4.diff_llada import StepState, StepDirective, _default_step_callback

__all__ = [
    "StepState",
    "StepDirective",
    "BD3_MASK_ID",
    "load",
]


# ── BD3 mask token id ───────────────────────────────────────────────────
# TODO(T2.C-eng-prereq): BD3-LMs uses the GPT-2 tokenizer (vocab 50257)
# with a CUSTOM mask token appended at training time. The exact id is
# not in the HF model card — must be discovered by either:
#   (a) inspecting `kuleshov-group/bd3lm-owt-block_size16` config, or
#   (b) calling tokenizer("<mask>") on the trained tokenizer artifact.
# Set to None as a sentinel until lookup completes.
BD3_MASK_ID: int | None = None


# ── Mock substrate ──────────────────────────────────────────────────────
@dataclass
class _Mock:
    """Deterministic mock for CI / smoke tests. Mirrors `e4.diff_llada._Mock`."""

    name: str
    block_len: int = 1024  # BD3-LMs context length is fixed at 1024
    sub_block_length: int = 16  # BD3 native block sizes: {4, 8, 16}; default 16
    gen_length: int = 256

    @property
    def tokenizer(self) -> Any:
        return None

    @property
    def model(self) -> Any:
        return None

    def _enable_commit(self) -> None:  # pragma: no cover
        return None

    def _disable_commit(self) -> None:  # pragma: no cover
        return None

    def _generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        # Deterministic mock string so existing CI smoke runners can
        # import this module without crashing.
        return f"[bd3-mock] {prompt[:32]}..."


# ── Real substrate (NOT IMPLEMENTED) ────────────────────────────────────
@dataclass
class _Real:
    """BD3-LMs real adapter — STUB. Every method raises NotImplementedError.

    Once the six eng-prereq phases (see top-of-file TODO block) land,
    this class should:
      1. Load the BD3 DiT from `kuleshov-group/bd3lm-owt-block_size{N}`.
         BD3-LMs is NOT a `transformers.AutoModel` subclass — must use
         the custom loader from the BD3 Hydra fork (see PRE_REG).
      2. Apply optional `lora_path` (Track-1 base) and `commit_lora_path`
         (Track-2 commit) PEFT adapters — requires Phase 3 PEFT wiring.
      3. Implement `_generate(...)` matching the LLaDA contract: prompt
         tokens + `gen_length` mask tokens, denoise per-block, commit
         lowest-confidence positions per BD3's native block_size.
      4. Toggle `_enable_commit` / `_disable_commit` at the sub-block
         boundary identified by the K2 ablation (sub-blocks 2-4 of 4).
    """

    name: str
    block_len: int = 1024              # BD3-LMs fixed context length
    gen_length: int = 256
    sub_block_length: int = 16          # BD3 native block size; pinned per checkpoint
    lora_path: str | None = None
    commit_lora_path: str | None = None

    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    @property
    def tokenizer(self) -> Any:
        """HuggingFace AutoTokenizer for BD3 (GPT-2 base + custom mask).

        TODO(T2.C-eng-prereq Phase 1): wire up
        `AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)`
        once the Hydra fork exposes a stable HF-compatible tokenizer
        artifact path.
        """
        return self._tokenizer

    @property
    def model(self) -> Any:
        """BD3 DiT model handle.

        TODO(T2.C-eng-prereq Phase 1): BD3-LMs uses a custom DiT class
        not registered with HF transformers AutoModel. Loading requires
        the BD3 Hydra repo on PYTHONPATH + their custom `load_model()`
        entrypoint. See `kuleshov-group/bd3lms/main.py:load_model` (TBC).
        """
        return self._model

    def _enable_commit(self) -> None:
        raise NotImplementedError(
            "BD3-LMs PEFT integration not yet wired — "
            "see phase2/spikes/bd3lms-cross-substrate/PRE_REG.md "
            "eng-prereq Phase 3 (add PEFT/LoRA infra to BD3 model class)."
        )

    def _disable_commit(self) -> None:
        raise NotImplementedError(
            "BD3-LMs PEFT integration not yet wired — "
            "see phase2/spikes/bd3lms-cross-substrate/PRE_REG.md "
            "eng-prereq Phase 3 (add PEFT/LoRA infra to BD3 model class)."
        )

    def _generate(
        self,
        prompt_ids,                      # torch.LongTensor (1, L)
        steps: int,
        temperature: float,
        commit_last_block: bool = False,
        commit_n_blocks: int = 1,
        step_callback: Optional[Callable[[StepState], StepDirective]] = None,
    ):
        """Run BD3-LMs semi-AR denoiser.

        Signature matches `e4.diff_llada._Real._generate` so the runner
        can swap substrates without caller changes. Every parameter has
        the same semantics as the LLaDA path:
          - `commit_last_block` + `commit_n_blocks` drive the K2 toggle
          - `step_callback` fires once per sub-block boundary

        TODO(T2.C-eng-prereq Phase 1 + 5): port the BD3 sampler loop
        from `kuleshov-group/bd3lms/main.py mode=sample_eval` (Hydra
        config-driven; nucleus_p=0.9, kv_cache=true). Lift into a
        programmatic Python entrypoint that:
          1. Tokenizes `prompt_ids` and pads to BD3's fixed 1024 context
             with `BD3_MASK_ID` mask tokens.
          2. Runs `gen_length // sub_block_length` denoising blocks,
             each with `steps // num_blocks` rounds.
          3. At each sub-block boundary fires `step_callback(StepState)`
             and honors the returned `StepDirective`.
          4. Toggles commit-LoRA via `_enable_commit` / `_disable_commit`
             at the K2 boundary identified in Phase-2 §3.
        """
        raise NotImplementedError(
            "BD3-LMs sampler not yet ported — requires Hydra fork of "
            "kuleshov-group/bd3lms main.py + LoRA hook integration. "
            "See phase2/spikes/bd3lms-cross-substrate/PRE_REG.md "
            "eng-prereq Phases 1, 3, and 5."
        )


def load(
    model_id: str = "kuleshov-group/bd3lm-owt-block_size16",
    mock: bool = False,
    block_len: int = 1024,
    lora_path: str | None = None,
    commit_lora_path: str | None = None,
) -> _Mock | _Real:
    """Load a BD3-LMs adapter (mock or real).

    Mirrors `e4.diff_llada.load(...)` so callers can swap substrates by
    changing one import line. The real path raises NotImplementedError
    on every method that touches a forward pass or PEFT toggle, until
    the T2.C eng prereqs land.

    Args:
      model_id: HF repo id of a BD3-LMs checkpoint. Defaults to the
        block_size=16 OWT pretrain (the closest-to-LLaDA-block sub-block
        granularity in the released family).
      mock: if True, returns a deterministic _Mock that prints a stub
        string. Used for CI smoke without GPU.
      block_len: BD3 fixed context length (1024 for all released ckpts).
      lora_path: optional Track-1 base LoRA path (HF repo or local dir).
        STUB — Phase 3 prereq required before this works.
      commit_lora_path: optional Track-2 commit LoRA path.
        STUB — Phase 3 prereq required before this works.

    Returns:
      _Mock or _Real, both implementing the LLaDA-compatible
      `(tokenizer, model, _enable_commit, _disable_commit, _generate)`
      surface area.

    Raises:
      NotImplementedError: every real-path method, until eng prereqs land.
    """
    if mock:
        return _Mock(name=model_id, block_len=block_len)
    return _Real(
        name=model_id,
        block_len=block_len,
        lora_path=lora_path,
        commit_lora_path=commit_lora_path,
    )
