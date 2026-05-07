"""DiffuLLaMA cross-substrate adapter — SCAFFOLD.

Phase-4 T2.C-light scaffold for testing whether sfumato's K2 inverted-U
commit-LoRA finding (cmajc-k3=0.822 GSM8K, +12pp MATH-500) replicates
on a *different* mask-diffusion language model family: DiffuLLaMA
(AR→DLM-converted LLaMA-2 7B; HKU-NLP / Shansan Gong et al.,
arXiv 2410.17891, ICLR-25).

This is the **cheaper Option B alternative** to BD3-LMs T2.C, per
R3's research validation note (`phase2/research/r3_cross_substrate_priorart.md`,
2026-05-07). DiffuLLaMA's cost envelope is ~$15-25 + 3-7 days vs BD3's
$40-60 + 14-21 days, because:

  - The GSM8K-symbolic LoRA `diffusionfamily/diffullama-gsm` already
    exists — no SFT-from-scratch step (BD3 ships OWT-only pretrain).
  - PEFT/LoRA training configs ship in repo (LLaMA-2 standard module
    names, lora_rank: 16) — no PEFT-from-scratch on a custom DiT.
  - LLaMA-2 backbone is *likely* `transformers.AutoModelForCausalLM`-
    compatible at the architecture level; only the discrete-diffusion
    sampling loop is custom (`inf_diffullama.py`). TBC in Phase 1.

Repo:        https://github.com/HKUNLP/DiffuLLaMA  (license TBC)
Checkpoints: diffusionfamily/diffullama          (LLaMA-2 7B → DLM, 2024-10-25)
             diffusionfamily/diffullama-gsm      (GSM8K-symbolic LoRA, 2025-02-19)
             diffusionfamily/diffugpt-{s,m}      (GPT-2 0.1B/0.4B sister models)
PRE_REG:     phase2/spikes/diffullama-cross-substrate/PRE_REG.md (this commit)

Scaffold pattern mirrors `e4/diff_bd3.py`:
  1. Reuses StepState / StepDirective from e4.diff_llada VERBATIM (the
     visualizer + branch-trace tooling key off these exact identities).
  2. Provides `load(...)` mirroring `e4.diff_llada.load(...)` so the
     runner can swap substrates by changing one import line.
  3. Mock path returns deterministic stub strings (CI-safe, no GPU).
  4. Real path raises NotImplementedError on each method that touches
     a forward pass or PEFT toggle, with TODO pointers to the relevant
     PRE_REG eng-prereq phase. Far fewer phases than BD3 (3 vs 6).

What's still TODO (vs `e4/diff_llada.py` which is full-fat):
  - Phase 1: confirm `AutoModelForCausalLM.from_pretrained(model_id,
    trust_remote_code=True)` returns a callable. WebFetch on 2026-05-07
    was silent on AutoModel registration — repo README points users at
    `inf_diffullama.py` rather than `transformers` directly, which is
    weak evidence against direct compat. May need a custom subclass
    that wraps the AR LLaMA-2 backbone with the discrete-diffusion
    masking schedule on top.
  - Phase 2: lift the sampler loop out of `inf_diffullama.py` into a
    reusable `_generate(prompt_ids, k_steps, ...)` matching the LLaDA
    contract. Map DiffuLLaMA's `diffusion_steps` knob onto sfumato's
    K_STEPS, and identify the sub-block boundary semantics for the K2
    toggle.
  - Phase 3: train commit-LoRA via `scripts/train_diffullama_commit_lora.py`
    (sister scaffold). Standard LLaMA-2 module names — no LLaDA-style
    `ff_proj/up_proj/ff_out` fork-quirk to handle.

If Phase 1 passes (AutoModel compat) the rest is mostly trivial PEFT
plumbing. If Phase 1 fails, costs converge to BD3's ladder (+$5-10).
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
    "DIFFULLAMA_MASK_ID",
    "load",
]


# ── DiffuLLaMA mask token id ────────────────────────────────────────────
# TODO(T2.C-light Phase 1): DiffuLLaMA reuses the LLaMA-2 tokenizer
# (vocab 32000) with a CUSTOM mask token added at AR→DLM conversion
# time. The exact id is not in the HF model card — must be discovered
# either by:
#   (a) inspecting `diffusionfamily/diffullama` config.json /
#       tokenizer_config.json, or
#   (b) calling `tokenizer.convert_tokens_to_ids("<mask>")` (or whatever
#       mask token name DiffuLLaMA uses; check `inf_diffullama.py`).
# Set to None as a sentinel until lookup completes. The arXiv 2410.17891
# paper §3.2 describes the AR→DLM conversion using a "[MASK]" token
# appended to the LLaMA-2 vocabulary — likely id 32000 or 32001
# depending on whether the conversion appended other special tokens
# first. CONFIRM before running real inference.
DIFFULLAMA_MASK_ID: int | None = None


# ── Mock substrate ──────────────────────────────────────────────────────
@dataclass
class _Mock:
    """Deterministic mock for CI / smoke tests. Mirrors `e4.diff_bd3._Mock`.

    Used when MOCK_MODELS=1 is set in the runner env, or when callers
    pass `mock=True` explicitly. Returns deterministic stub strings so
    existing CI smoke runners can import this module without crashing.
    """

    name: str
    block_len: int = 4096          # LLaMA-2 7B native context length
    sub_block_length: int = 32     # match LLaDA default; TBC vs DiffuLLaMA paper
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
        # import this module without crashing. Same shape as BD3 mock.
        return f"[diffullama-mock] {prompt[:32]}..."


# ── Real substrate (NOT IMPLEMENTED) ────────────────────────────────────
@dataclass
class _Real:
    """DiffuLLaMA real adapter — STUB. Every method raises NotImplementedError.

    Once the three eng-prereq phases (PRE_REG §"Eng prerequisites")
    land, this class should:
      1. Load DiffuLLaMA via `AutoModelForCausalLM.from_pretrained(
         self.name, trust_remote_code=True)` (Phase 1 confirmation
         pending). If AutoModel-incompatible, fall back to importing
         the model class from a pinned fork of HKUNLP/DiffuLLaMA.
      2. Apply optional `lora_path` (Track-1 base, e.g.
         `diffusionfamily/diffullama-gsm`) and `commit_lora_path`
         (Track-2 commit, e.g. `eren23/sfumato-diffullama-commit-v1`)
         PEFT adapters via `PeftModel.from_pretrained` — should work
         out of the box on LLaMA-2 standard modules.
      3. Implement `_generate(...)` matching the LLaDA contract: prompt
         tokens + `gen_length` mask tokens, denoise per-block, commit
         lowest-confidence positions per DiffuLLaMA's native sub-block
         schedule.
      4. Toggle `_enable_commit` / `_disable_commit` at the sub-block
         boundary identified by the K2 ablation (sub-blocks 2-4 of 4
         on LLaDA — TBC how this maps onto DiffuLLaMA's diffusion_steps
         knob).
    """

    name: str
    block_len: int = 4096               # LLaMA-2 7B native context
    gen_length: int = 256
    sub_block_length: int = 32          # default from LLaDA; TBC for DiffuLLaMA
    lora_path: str | None = None
    commit_lora_path: str | None = None

    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    @property
    def tokenizer(self) -> Any:
        """HuggingFace AutoTokenizer for DiffuLLaMA (LLaMA-2 base + custom mask).

        TODO(T2.C-light Phase 1): wire up
        `AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)`.
        LLaMA-2 tokenizer is well-supported; only the custom mask
        token id needs verification (see DIFFULLAMA_MASK_ID note).
        """
        return self._tokenizer

    @property
    def model(self) -> Any:
        """DiffuLLaMA model handle (LLaMA-2 7B with discrete-diffusion head).

        TODO(T2.C-light Phase 1): try
        `AutoModelForCausalLM.from_pretrained(self.name, trust_remote_code=True)`
        first. WebFetch 2026-05-07 was silent on AutoModel registration
        — the repo README only documents `inf_diffullama.py`-driven
        inference. If AutoModel raises, fall back to a pinned fork of
        HKUNLP/DiffuLLaMA on PYTHONPATH and use their custom loader.
        """
        return self._model

    def _enable_commit(self) -> None:
        raise NotImplementedError(
            "DiffuLLaMA PEFT integration not yet wired — "
            "see phase2/spikes/diffullama-cross-substrate/PRE_REG.md "
            "Phase 1 (AutoModel compat audit) + Phase 3 (commit-LoRA "
            "training). Should be straightforward once Phase 1 passes "
            "since DiffuLLaMA uses standard LLaMA-2 module names "
            "(gate_proj/up_proj/down_proj) — unlike LLaDA's custom "
            "ff_proj/up_proj/ff_out fork-quirk."
        )

    def _disable_commit(self) -> None:
        raise NotImplementedError(
            "DiffuLLaMA PEFT integration not yet wired — "
            "see phase2/spikes/diffullama-cross-substrate/PRE_REG.md "
            "Phase 1 + Phase 3."
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
        """Run DiffuLLaMA semi-AR denoiser.

        Signature matches `e4.diff_llada._Real._generate` so the runner
        can swap substrates without caller changes. Every parameter has
        the same semantics as the LLaDA path:
          - `commit_last_block` + `commit_n_blocks` drive the K2 toggle
          - `step_callback` fires once per sub-block boundary

        TODO(T2.C-light Phase 2): port the DiffuLLaMA sampler loop
        from `inf_diffullama.py`. The repo's sampler is a vanilla Python
        script (NOT Hydra-driven, unlike BD3-LMs) so this should be a
        ~100 LOC port: tokenize prompt, append `gen_length` mask
        tokens, run `diffusion_steps` denoising rounds with top-k
        nucleus sampling at `logits_temp`, fire `step_callback` at
        each sub-block boundary, honor returned StepDirective.

        Sub-block boundary semantics (TBC Phase 1 audit):
          - DiffuLLaMA paper §4 doesn't explicitly partition the
            denoising into "sub-blocks" the way LLaDA does. The
            `diffusion_steps` count is the closest analog to LLaDA's
            num_sub_blocks. May need to infer the K2 boundary from
            denoising-step k=2 of K=4 (matching LLaDA's K2 finding
            of sub-blocks 2-4 of 4).
          - If DiffuLLaMA uses cosine masking schedule (per arXiv
            2410.17891 §3.3) the sub-block boundary may not align
            with LLaDA's linear schedule. Document the mapping
            choice in RESULT.md when the spike fires.
        """
        raise NotImplementedError(
            "DiffuLLaMA sampler not yet ported — see "
            "phase2/spikes/diffullama-cross-substrate/PRE_REG.md "
            "Phase 2 (port `inf_diffullama.py` sampler into a "
            "programmatic `_generate(prompt_ids, ...)` matching the "
            "LLaDA contract)."
        )


def load(
    model_id: str = "diffusionfamily/diffullama",
    mock: bool = False,
    block_len: int = 4096,
    lora_path: str | None = None,
    commit_lora_path: str | None = None,
) -> _Mock | _Real:
    """Load a DiffuLLaMA adapter (mock or real).

    Mirrors `e4.diff_llada.load(...)` and `e4.diff_bd3.load(...)` so
    callers can swap substrates by changing one import line. The real
    path raises NotImplementedError on every method that touches a
    forward pass or PEFT toggle, until the T2.C-light eng prereqs
    land (3 phases, 3-7 days, $15-25 — much smaller than BD3 T2.C's
    6-phase 14-21 day, $40-60 stack).

    Args:
      model_id: HF repo id of a DiffuLLaMA checkpoint. Defaults to
        `diffusionfamily/diffullama` (LLaMA-2 7B base, AR→DLM-converted
        per arXiv 2410.17891). Sister checkpoints:
          - `diffusionfamily/diffullama-gsm`: GSM8K-symbolic LoRA on
            DiffuLLaMA-base (use as `lora_path=`).
          - `diffusionfamily/diffugpt-{s,m}`: GPT-2 0.1B/0.4B variants
            with the same conversion recipe (cheaper but no GSM tuning).
      mock: if True, returns a deterministic _Mock that prints a stub
        string. Used for CI smoke without GPU. Honored by the runner
        when MOCK_MODELS=1.
      block_len: LLaMA-2 native context length (4096). DiffuLLaMA
        inherits this from the base AR model.
      lora_path: optional Track-1 base LoRA path (HF repo or local dir).
        Recommended: `diffusionfamily/diffullama-gsm` for math substrates.
        STUB — Phase 1 (AutoModel compat) required before this works.
      commit_lora_path: optional Track-2 commit LoRA path. Future
        artifact: `eren23/sfumato-diffullama-commit-v1`. STUB — Phase 3
        (commit-LoRA training) required before this works.

    Returns:
      _Mock or _Real, both implementing the LLaDA-compatible
      `(tokenizer, model, _enable_commit, _disable_commit, _generate)`
      surface area.

    Raises:
      NotImplementedError: every real-path method, until eng prereqs land.

    See also:
      - phase2/spikes/diffullama-cross-substrate/PRE_REG.md (decision
        rules locked 2026-05-07)
      - e4/diff_bd3.py (sister scaffold, more expensive substrate)
      - e4/diff_llada.py (primary contract; ~1185 LOC, full-fat)
    """
    if mock:
        return _Mock(name=model_id, block_len=block_len)
    return _Real(
        name=model_id,
        block_len=block_len,
        lora_path=lora_path,
        commit_lora_path=commit_lora_path,
    )
