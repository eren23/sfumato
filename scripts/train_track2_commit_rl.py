"""Phase-4 Direction A: schedule-aware commit-LoRA via REINFORCE/GRPO.

Forks `scripts/train_track2_commit.py` (CE-loss) into a trajectory-level
policy-gradient trainer. The current commit-LoRA peaks at cmajc-k=3 = 0.822
on GSM8K but was trained with vanilla CE on (q, cot) pairs - the K=3
schedule it gets toggled at is correlational, never causal. This trainer
makes the schedule signal causal:

  - Sample M=8 cmajc-k3 rollouts per prompt.
  - Reward r in {0,1} from `e4.grade.is_correct(final_text, gold)`.
  - GRPO group-relative advantage A_i = (r_i - mean) / (std + eps).
  - Policy gradient on log pi over the COMMIT-active sub-blocks (t in 1,2,3).
  - KL anchor beta=0.05 vs frozen base via `peft.disable_adapter_layers()`.
  - 4-vector phase_emb[t] additive bias on LoRA-A for layers 24-31 (~1k params).

PRE_REG (locked in plan file `bro-bro-bro-bro-prancy-volcano.md`, "Direction A"):
  Substrate     : GSM8K-train 1000 + MATH-train 500 + AIME-train 50, M=8, 2 epochs.
  LoRA arch     : r=8, alpha=16, FFN-only (ff_proj, up_proj, ff_out), layers 24-31.
  Schedule      : fixed k=3 during training (no learned schedule).
  Tripwires     : Jaccard(commit-tokens) > 0.6 across distinct prompts (sliding K=20)
                  + MATH-500 dev N=50 cmajc-k3 regression > 5pp from 0.41 baseline.

  | Outcome  | MATH-500 cmajc-k3 N=200 | GSM8K-1319 cmajc-k3 |
  |----------|------------------------:|--------------------:|
  | WIN      | >= 0.51 (+10pp)         | >= 0.80 (no >2pp regression) |
  | PARTIAL  | 0.45-0.50               | >= 0.78 |
  | LOSS     | < 0.45 OR Jaccard trip  | < 0.78 |

Crucible BYO-trainer contract (env-var driven):
    step:{step}/{total} train_loss:{loss}
    step:{step}/{total} val_loss:{loss} val_bpb:{tokens_seen}
    Serialized model {path} {bytes}

Pinned: transformers==4.46.3 (LLaDA breaks on transformers 5.x).

CPU smoke (no model load, <5s):
    python3 scripts/train_track2_commit_rl.py --smoke

RESOLVED (2026-05 — Phase-4 Direction A blockers landed):
  - per-sub-block logits capture: `e4/diff_llada.StepState.committed_logits`
    populated by `_generate` when env `EMIT_LOGPROBS=1`. Default off.
  - phase_emb forward integration: MethodType monkey-patch on each
    `lora_A.default` linear, gated on a thread-local `t` set by the
    `phase_emb_subblock(t)` context manager. PEFT-version-agnostic.
  - KL anchor: `kl_k3_estimator()` (Schulman k3, always >=0) wired into
    `main_train`; second forward pass uses `model.disable_adapter_layers()`
    + `phase_emb.set_adapter_disabled(True)` for the frozen-base log-probs.

REMAINING TODO:
  (none — phase_emb sub-block writer landed in `e4/diff_llada.py:
  _Real.denoise_block` inner loop. Reader is `_pes.get_sub_block()`
  inside `_wrapped_forward`. Writer side is gated on env `SCHEDULE_RL=1`
  so production paths bypass entirely.)

LANDED (2026-05 — phase4 Direction A real-path rollout):
  - `_rollout` real path: drives `model.denoise_block(...)` with a
    step_callback that captures per-sub-block `tokens_committed` +
    `committed_logits` from StepState (with EMIT_LOGPROBS=1 set in env),
    then computes per-token log-pi via `_logits_to_token_logprobs` for
    each commit-active sub-block (1..commit_n_blocks). Sub-block 0 stays
    empty (commit-LoRA OFF). Reward = `e4.grade.is_correct(text, gold)`.
    Returns `List[RolloutResult]` (NamedTuple, tuple-iterable for legacy
    consumers in grpo_loss / main_train).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from types import MethodType
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ----------------------------------------------------------------------------
# Config (env-driven; mirrors train_track2_commit.py patterns)
# ----------------------------------------------------------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


MODEL_NAME = _env("MODEL_NAME", "GSAI-ML/LLaDA-8B-Instruct")
SUBSTRATE_JSONL = _env(
    "SUBSTRATE_JSONL",
    str(REPO_ROOT / "e4" / "data" / "rl_substrate_phase4_v1.jsonl"),
)
RESUME_FROM = _env("RESUME_FROM", "eren23/sfumato-llada-commit-v3")

LORA_R = _env_int("LORA_R", 8)
LORA_ALPHA = _env_int("LORA_ALPHA", 16)
LORA_DROPOUT = _env_float("LORA_DROPOUT", 0.05)
LR = _env_float("LR", 1e-5)  # RL is more delicate than CE; lower LR
EPOCHS = _env_int("EPOCHS", 2)
WARMUP_STEPS = _env_int("WARMUP_STEPS", 50)
SEED = _env_int("SEED", 42)
LOG_INTERVAL = _env_int("LOG_INTERVAL", 5)

# GRPO hyperparams
M_ROLLOUTS = _env_int("M_ROLLOUTS", 8)
KL_BETA = _env_float("KL_BETA", 0.05)
ADV_EPS = _env_float("ADV_EPS", 1e-8)
COMMIT_N_BLOCKS = _env_int("COMMIT_N_BLOCKS", 3)  # k=3 schedule, sub-blocks 1,2,3
NUM_SUB_BLOCKS = _env_int("NUM_SUB_BLOCKS", 4)
ROLLOUT_K_STEPS = _env_int("ROLLOUT_K_STEPS", 64)
ROLLOUT_TEMPERATURE = _env_float("ROLLOUT_TEMPERATURE", 1.0)

# Tripwires
JACCARD_THRESHOLD = _env_float("JACCARD_THRESHOLD", 0.6)
JACCARD_WINDOW = _env_int("JACCARD_WINDOW", 20)
EVAL_INTERVAL = _env_int("EVAL_INTERVAL", 200)
MATH500_BASELINE = _env_float("MATH500_BASELINE", 0.41)
MATH500_REGRESSION_PP = _env_float("MATH500_REGRESSION_PP", 0.05)
MATH500_DEV_N = _env_int("MATH500_DEV_N", 50)

# LoRA target spec - MUST match the v3 commit-LoRA exactly.
# CRITICAL: LLaDA's LLaDALlamaBlock uses ff_proj/up_proj/ff_out, NOT the
# llama-standard gate_proj/down_proj names. See train_track2_commit.py:90-94.
LORA_TARGETS = ["ff_proj", "up_proj", "ff_out"]
LORA_LAYERS_TO_TRANSFORM = list(range(24, 32))

SAVE_DIR = Path(_env("SAVE_DIR", "/tmp/track2_commit_rl"))
OUTPUT_REPO = _env("HF_OUTPUT_REPO", "eren23/sfumato-llada-commit-rl-v1")
PUSH = _env_bool("PUSH_TO_HUB", False)
WANDB_RUN_NAME = _env("WANDB_RUN_NAME", "track2-commit-rl")


# ----------------------------------------------------------------------------
# GRPO advantage math (pure python - works in --smoke without torch)
# ----------------------------------------------------------------------------
def grpo_advantages(rewards: Sequence[float], eps: float = ADV_EPS) -> List[float]:
    """Group-relative advantage A_i = (r_i - mean) / (std + eps).

    rewards: list of M scalar rewards from M rollouts of a single prompt.
    Returns a list of M advantages (same length).
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    return [(r - mean) / (std + eps) for r in rewards]


# ----------------------------------------------------------------------------
# Reward-hack tripwire: pairwise Jaccard on commit-token id-sets
# ----------------------------------------------------------------------------
def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


class JaccardTripwire:
    """Sliding-window pairwise-Jaccard guard on committed token id-sets.

    Halts training if the MEAN pairwise Jaccard across the last K=window
    DISTINCT prompts exceeds threshold - a reliable proxy for the adapter
    collapsing onto a single rote response that games the reward.
    """

    def __init__(self, window: int = JACCARD_WINDOW, threshold: float = JACCARD_THRESHOLD):
        self.window = window
        self.threshold = threshold
        self.buffer: deque = deque(maxlen=window)

    def add(self, prompt_id: str, commit_token_ids: Sequence[int]) -> None:
        self.buffer.append((prompt_id, list(commit_token_ids)))

    def tripped(self) -> Tuple[bool, float]:
        """Return (tripped, mean_pairwise_jaccard)."""
        items = list(self.buffer)
        if len(items) < self.window:
            return False, 0.0
        sims: List[float] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] == items[j][0]:
                    continue  # skip same-prompt pairs
                sims.append(_jaccard(items[i][1], items[j][1]))
        if not sims:
            return False, 0.0
        mean_sim = sum(sims) / len(sims)
        return mean_sim > self.threshold, mean_sim


def _check_jaccard_tripwire(commit_token_ids_per_prompt: List[List[int]]) -> bool:
    """Convenience wrapper used by the smoke test.

    Treats each list as belonging to a distinct prompt; window = len(input)
    so the entire batch is checked. Returns True iff mean pairwise Jaccard
    exceeds JACCARD_THRESHOLD.
    """
    if len(commit_token_ids_per_prompt) < 2:
        return False
    tw = JaccardTripwire(window=len(commit_token_ids_per_prompt))
    for i, ids in enumerate(commit_token_ids_per_prompt):
        tw.add(prompt_id=f"p{i}", commit_token_ids=ids)
    tripped, _ = tw.tripped()
    return tripped


# ----------------------------------------------------------------------------
# Phase embedding (4-vector additive bias on LoRA-A for layers 24-31)
# ----------------------------------------------------------------------------
# Sub-block index lives in `e4.phase_emb_state` (shared library) so the
# diffusion runner in `e4/diff_llada.py:_Real.denoise_block` can WRITE the
# index from inside its inner forward loop without importing anything from
# `scripts/`. The MethodType monkey-patch below READS the same thread-local
# from the LoRA-A forward.
from e4 import phase_emb_state as _pes


@contextlib.contextmanager
def phase_emb_subblock(t: Optional[int]):
    """Context manager: set the shared phase index for a sub-block.

    Usage in tests / smokes (where SCHEDULE_RL is set explicitly OR we
    want to bypass the gate):
        with phase_emb_subblock(state.sub_block + 1):
            <call into _generate for the next sub-block>

    In production rollouts the writer side of the thread-local is owned
    by `e4.diff_llada._Real.denoise_block` (which writes `b_idx` per
    sub-block when `SCHEDULE_RL=1`); this context manager exists so unit
    smokes can drive the monkey-patch without spinning the diffusion
    runner.

    `t=0` and `t=None` are both treated as "skip the phase add" (commit-LoRA
    is OFF on sub-block 0 by design). The wrapped LoRA-A forward also skips
    when the adapter is disabled (KL frozen-base pass).
    """
    prev = _pes.get_sub_block()
    _pes.force_set_sub_block(t)
    try:
        yield
    finally:
        _pes.force_set_sub_block(prev)


class PhaseEmbedding:
    """Schedule-phase embedding: 4-vector phase_emb[t] added to LoRA-A output.

    Owns:
      - nn.Parameter of shape (NUM_SUB_BLOCKS, lora_rank).
      - A `MethodType` monkey-patch installed on each LoRA-A linear in the
        targeted layers. The patched forward calls the original linear,
        then ADDS `phase_emb[t]` to its output when the thread-local
        sub-block index `t` is in {1, 2, 3} AND the adapter is not disabled.

    PEFT-version-agnostic: we don't subclass LoraLayer; we just rebind
    `forward` on each `lora_A.default` `nn.Linear` instance. The base method
    is stored as `_orig_forward_for_phase` so detach() can restore it.

    Skip rules:
      - `t is None or t == 0` → no phase add (commit-LoRA OFF on sub-block 0).
      - `module._sfumato_adapter_disabled = True` → no phase add (KL pass).
    """

    def __init__(self, num_phases: int = NUM_SUB_BLOCKS, lora_rank: int = LORA_R):
        self.num_phases = num_phases
        self.lora_rank = lora_rank
        self.param: Any = None  # set by attach()
        self._patched: List[Any] = []  # modules we monkey-patched

    def attach(self, peft_model: Any) -> int:
        """Monkey-patch lora_A.default.forward on each targeted layer.

        Returns the number of modules patched. Caller is responsible for
        adding `self.param` to the optimizer parameter list.
        """
        import torch
        import torch.nn as nn

        self.param = nn.Parameter(
            torch.zeros(self.num_phases, self.lora_rank, dtype=torch.float32),
            requires_grad=True,
        )
        peft_model._sfumato_phase_emb = self.param

        num_phases = self.num_phases
        emb = self.param

        def _wrapped_forward(self_mod: Any, x: Any) -> Any:
            out = self_mod._orig_forward_for_phase(x)
            t = _pes.get_sub_block()
            disabled = getattr(self_mod, "_sfumato_adapter_disabled", False)
            if t is None or t == 0 or disabled:
                return out
            # Clamp into bounds defensively.
            t_idx = max(0, min(num_phases - 1, int(t)))
            return out + emb[t_idx].to(out.dtype).to(out.device)

        n_patched = 0
        target_layer_set = set(LORA_LAYERS_TO_TRANSFORM)
        # PEFT stores the LoRA-A linear at `<...>.lora_A.<adapter_name>`. The
        # adapter name varies per-load (e.g. `default` for unnamed adapters,
        # `commit` for the sfumato commit-LoRA). Match by penultimate segment
        # = "lora_A" so we hit the active adapter's LoRA-A regardless of
        # which name PEFT assigned it.
        for name, module in peft_model.named_modules():
            parts = name.split(".")
            if len(parts) < 2 or parts[-2] != "lora_A":
                continue
            # Skip the ModuleDict container itself; only patch the leaf Linear.
            try:
                import torch.nn as _nn
                if not isinstance(module, _nn.Linear):
                    continue
            except ImportError:
                pass
            layer_idx = _extract_layer_idx(name)
            if layer_idx is None or layer_idx not in target_layer_set:
                continue
            module._orig_forward_for_phase = module.forward
            module.forward = MethodType(_wrapped_forward, module)
            self._patched.append(module)
            n_patched += 1
        return n_patched

    def set_adapter_disabled(self, disabled: bool) -> None:
        """Flip the skip-on-disable flag on every patched module.

        Used by the KL anchor pass: before `model.disable_adapter_layers()`
        we set this True so the phase add is also skipped (otherwise the
        frozen-base forward would still see phase noise).
        """
        for m in self._patched:
            m._sfumato_adapter_disabled = disabled

    def detach(self) -> None:
        for m in self._patched:
            orig = getattr(m, "_orig_forward_for_phase", None)
            if orig is not None:
                m.forward = orig
                try:
                    del m._orig_forward_for_phase
                except Exception:
                    pass
        self._patched = []


def _rescore_committed_logprobs(
    diff_model: Any,
    prompt_text: str,
    committed_per_sb: Sequence[Sequence[int]],
    *,
    num_sub_blocks: int = 4,
    commit_n_blocks: int = 3,
    sub_block_length: int = 32,
    mask_id: Optional[int] = None,
) -> List[Any]:
    """Re-score the committed tokens through the LIVE peft_model with grad.

    `committed_logits` captured during rollout are detached (per
    `e4/diff_llada.py:744-746`), so the grpo_loss tensor built from them
    has no autograd path back to the LoRA / phase_emb params. To make
    the policy gradient actually update the model, we run ONE forward per
    commit-active sub-block t in {1..commit_n_blocks} with phase_emb[t]
    set via the shared thread-local, and extract log-pi at the
    sub-block-t positions.

    Re-score input per sub-block t:
      [chat-templated prompt]
      [committed tokens for sub-blocks 0..t-1]
      [MASK tokens for sub-blocks t..N-1]

    This mirrors the diffusion state at sub-block t entry — model sees
    the prefix that was already committed, masks for everything else.
    log_pi_theta is then log_softmax(model_logits)[committed_token_at_p]
    for each position p in sub-block t.

    Returns: list of length num_sub_blocks; entries 1..commit_n_blocks are
    (T_t,) tensors connected to the autograd graph. Sub-block 0 stays
    empty (commit-LoRA OFF).
    """
    import torch
    import torch.nn.functional as F

    from e4 import phase_emb_state as _pes
    from e4 import diff_llada as _diff

    tokenizer = diff_model._tokenizer
    model = diff_model._model
    device = next(model.parameters()).device
    if mask_id is None:
        mask_id = _diff._LLADA_MASK_ID

    # Build chat-templated prompt — match _DENOISE_SYS exactly so the
    # re-score sees the same prompt embedding the rollout did.
    messages = [
        {"role": "system", "content": _diff._DENOISE_SYS},
        {"role": "user", "content": prompt_text},
    ]
    prompt_token_ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=True, add_generation_prompt=True
    )
    prompt_len = len(prompt_token_ids)

    # Pre-pad each sub-block to exactly sub_block_length so absolute
    # position arithmetic stays simple.
    def _pad(ids: Sequence[int]) -> List[int]:
        ids = list(ids)
        if len(ids) >= sub_block_length:
            return ids[:sub_block_length]
        return ids + [mask_id] * (sub_block_length - len(ids))

    padded_per_sb: List[List[int]] = [
        _pad(committed_per_sb[sb] if sb < len(committed_per_sb) else [])
        for sb in range(num_sub_blocks)
    ]

    out_lp: List[Any] = [torch.zeros(0, device=device) for _ in range(num_sub_blocks)]

    for t in range(1, commit_n_blocks + 1):
        committed_t = list(committed_per_sb[t]) if t < len(committed_per_sb) else []
        if not committed_t:
            continue
        # Input = prompt + sub-blocks 0..t-1 (committed) + sub-blocks t.. (masks).
        seq: List[int] = list(prompt_token_ids)
        for sb in range(num_sub_blocks):
            if sb < t:
                seq.extend(padded_per_sb[sb])
            else:
                seq.extend([mask_id] * sub_block_length)
        full_ids = torch.tensor([seq], dtype=torch.long, device=device)

        _pes.force_set_sub_block(t)
        try:
            outputs = model(full_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            sb_start = prompt_len + t * sub_block_length
            n_committed = len(committed_t)
            sb_logits = logits[0, sb_start : sb_start + n_committed]  # (n, vocab)
            log_p = F.log_softmax(sb_logits, dim=-1)
            committed_ids = torch.tensor(committed_t, dtype=torch.long, device=device)
            token_lp = log_p.gather(1, committed_ids.unsqueeze(1)).squeeze(1)
            out_lp[t] = token_lp
        finally:
            _pes.clear()

    return out_lp


def _logits_to_token_logprobs(logits: Any, token_ids: Sequence[int]) -> Any:
    """Per-position log pi(committed_token) from captured logits.

    `logits`: (T, vocab) tensor (CPU or device — caller's choice).
    `token_ids`: length-T sequence of int token ids.
    Returns: (T,) tensor of log-softmax(logits)[t, token_ids[t]].

    Used by the KL anchor pass: forward A → log_pi_theta on the committed
    positions, forward B (adapter disabled) → log_pi_frozen.
    """
    import torch
    import torch.nn.functional as F

    if logits is None:
        return torch.zeros(0)
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=logits.device)
    if logits.shape[0] == 0:
        return torch.zeros(0, device=logits.device)
    log_probs = F.log_softmax(logits, dim=-1)  # (T, vocab)
    idx = torch.arange(logits.shape[0], device=logits.device)
    return log_probs[idx, token_ids]


def kl_k3_estimator(log_pi_theta: Any, log_pi_frozen: Any) -> Any:
    """Schulman's k3 KL-divergence estimator (low-variance, always >=0).

        ratio = exp(log_pi_frozen - log_pi_theta)   # = pi_frozen / pi_theta
        kl    = (ratio - 1) - log(ratio)

    Equivalent unbiased reformulation in the spec:
        delta = log_pi_theta - log_pi_frozen
        kl    = sum( exp(-delta) - 1 + delta )      # always >= 0

    This is the gradient-stable surrogate used in TRL/GRPO trainers and
    referenced in http://joschu.net/blog/kl-approx.html (k3). Always
    non-negative, zero exactly when pi_theta == pi_frozen at the sample.

    Caller passes (T,) tensors of per-token log-probs over the committed
    positions; the loss reduces over T via .sum().
    """
    import torch

    if log_pi_theta is None or log_pi_frozen is None:
        return torch.zeros(())
    delta = log_pi_theta - log_pi_frozen
    # k3: KL = E_theta[ exp(-delta) - 1 + delta ] >= 0 with zero iff equal.
    return (torch.exp(-delta) - 1.0 + delta).sum()


def _extract_layer_idx(module_name: str) -> Optional[int]:
    """Pull the integer layer index from a dotted module name."""
    parts = module_name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h", "blocks") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


# ----------------------------------------------------------------------------
# Rollout harness
# ----------------------------------------------------------------------------
class RolloutResult(NamedTuple):
    """One cmajc-k3 rollout's payload.

    NamedTuple so existing tuple-unpacking call sites (grpo_loss, main_train,
    smoke 5) continue to work — `(committed, lp, text, reward) = result`.

    Fields:
      committed_tokens: per-sub-block list of committed token ids.
        len == num_sub_blocks; each inner list is the ids committed during
        that sub-block.
      committed_logprobs: per-sub-block list of per-token log-pi values.
        Each inner element is either:
          - a torch.Tensor of shape (n_committed,) when EMIT_LOGPROBS=1 was
            honoured by diff_llada (real GPU path);
          - a python list[float] (CPU smoke fallback path);
          - an empty list [] for sub-blocks where commit-LoRA is OFF
            (sub-block 0 by default) OR when `committed_logits` was None
            (mock mode / EMIT_LOGPROBS=0).
        Sub-blocks 1..commit_n_blocks carry gradients in the real path.
      final_text: detokenized completion text.
      reward: e4.grade.is_correct(final_text, gold) -> {0.0, 1.0}.
    """

    committed_tokens: List[List[int]]
    committed_logprobs: List[Any]
    final_text: str
    reward: float


def _zero_logprobs_like(n: int) -> Any:
    """Per-token log-prob placeholder when committed_logits is unavailable.

    Returns a torch zero-tensor of shape (n,) when torch is importable, else
    a python list. The grpo_loss path detects torch.Tensor automatically.
    """
    try:
        import torch  # type: ignore
        return torch.zeros(n)
    except ImportError:
        return [0.0] * n


def _rollout(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gold: str,
    *,
    k_steps: int = ROLLOUT_K_STEPS,
    n_branches: int = 5,
    m: int = M_ROLLOUTS,
    temperature: float = ROLLOUT_TEMPERATURE,
    seed: int = 0,
    commit_n_blocks: int = COMMIT_N_BLOCKS,
    num_sub_blocks: int = NUM_SUB_BLOCKS,
) -> List[RolloutResult]:
    """Run M cmajc-k3 rollouts for a single prompt.

    Returns a list of length M of `RolloutResult`s. The named tuple is
    iterable, so legacy `(committed, lp, text, reward)` unpacking still
    works (grpo_loss + smoke 5 / main_train).

    Real path (when `model` is a real or mock `e4.diff_llada` instance):
      For each rollout i in 0..m-1:
        - Set EMIT_LOGPROBS=1 in os.environ so StepState.committed_logits
          is populated by `_generate`.
        - Call `model.denoise_block(prompt, k_steps, seed=seed*1000+i,
          temperature, apply_commit=True, commit_n_blocks=commit_n_blocks,
          step_callback=cb)`.
        - The callback `cb` captures `state.tokens_committed` +
          `state.committed_logits` per sub-block into per-rollout lists.
        - After denoise_block returns the final text, compute log-probs
          via `_logits_to_token_logprobs(captured_logits, captured_tokens)`
          for each commit-active sub-block (1..commit_n_blocks). Sub-block 0
          stays empty (commit-LoRA OFF). When `committed_logits` is None
          (mock mode, or EMIT_LOGPROBS not honored), fall back to a zero
          tensor of the right length so downstream grpo_loss still finds a
          tensor to .stack().
        - Reward = float(e4.grade.is_correct(final_text, gold)).

    Mock-stub fallback (when `model is None` AND MOCK_MODELS=1):
      Synthesize a deterministic-ish rollout structure via random; used by
      smoke 5 (no model load at all). When `model` is a real _Mock object
      we go through the real path above and exercise the actual callback
      plumbing.

    TODO (phase_emb wrapping inside cmajc generate):
      The phase_emb additive bias on LoRA-A needs to be active for each
      sub-block forward, but `denoise_block`'s step_callback fires AT the
      sub-block BOUNDARY (after the forward completes), not as a bracket
      around the next forward. Cleanly wrapping each sub-block forward
      requires a hook inside `_generate`'s inner loop in `e4/diff_llada.py`.
      Until that hook lands, rollouts run with phase_emb at zero-init
      (effectively no schedule-conditioning); the GRPO infrastructure works
      but the schedule-conditional advantage isn't yet realized. The k3 KL
      anchor + advantage-weighted log-pi still flow correctly.
    """
    # Stub fallback (no diff-model object at all): synthetic dummy rollouts.
    # Used by smoke 5 to exercise GRPO arithmetic without any model.
    if model is None and os.environ.get("MOCK_MODELS") == "1":
        rng = random.Random(hash((prompt, gold)) & 0xFFFFFFFF)
        out_stub: List[RolloutResult] = []
        for i in range(m):
            committed = [[rng.randint(0, 32000) for _ in range(8)] for _ in range(num_sub_blocks)]
            log_probs = [
                ([math.log(rng.uniform(0.01, 0.99)) for _ in range(8)]
                 if 1 <= sb <= commit_n_blocks else [])
                for sb in range(num_sub_blocks)
            ]
            final_text = f"[mock rollout {i} for prompt[:32]={prompt[:32]!r}]"
            reward = float(rng.random() < 0.5)
            out_stub.append(RolloutResult(committed, log_probs, final_text, reward))
        return out_stub

    # Real path: requires a diff-model object exposing `.denoise_block(...)`
    # with a step_callback contract matching e4/diff_llada.StepState.
    if not hasattr(model, "denoise_block"):
        raise TypeError(
            f"_rollout real path requires a diff-model with .denoise_block; "
            f"got {type(model).__name__}. Pass a `e4.diff_llada.load(...)` "
            f"instance or set MOCK_MODELS=1 with model=None for the smoke stub."
        )

    # Late import: e4.grade is cheap and torch-free.
    from e4.grade import is_correct  # type: ignore

    # Lazy torch import — only needed for log-prob extraction. If torch isn't
    # available we still produce results (with python-list zero log-probs).
    _torch_available = True
    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError:
        _torch_available = False

    # Honor EMIT_LOGPROBS=1 contract: diff_llada._generate stashes per-sub-
    # block committed logits onto StepState only when this env var is set.
    # We set it for the duration of the rollout call and restore it after.
    prev_emit = os.environ.get("EMIT_LOGPROBS")
    os.environ["EMIT_LOGPROBS"] = "1"
    try:
        results: List[RolloutResult] = []
        for i in range(m):
            seed_i = seed * 1000 + i
            captured_tokens: List[List[int]] = []
            captured_logits: List[Any] = []  # per sub-block; None or tensor

            def _cb(state: Any) -> Any:  # noqa: ANN401
                # state is e4.diff_llada.StepState
                captured_tokens.append(list(state.tokens_committed))
                captured_logits.append(state.committed_logits)
                # Default directive: continue the LLaDA schedule unchanged.
                # We can't construct StepDirective without importing it;
                # returning None is treated as continue_llada by both real
                # and mock _generate (see e4/diff_llada.py:278-280).
                return None

            text, _flops = model.denoise_block(
                prompt=prompt,
                k_steps=k_steps,
                seed=seed_i,
                temperature=temperature,
                apply_commit=True,
                commit_n_blocks=commit_n_blocks,
                step_callback=_cb,
            )

            # Pad / truncate the captured per-sub-block lists to exactly
            # `num_sub_blocks` so downstream tripwire + grpo_loss code can
            # index without bounds checks.
            while len(captured_tokens) < num_sub_blocks:
                captured_tokens.append([])
                captured_logits.append(None)
            captured_tokens = captured_tokens[:num_sub_blocks]
            captured_logits = captured_logits[:num_sub_blocks]

            # Build per-sub-block log-prob payloads.
            #   - sub-block 0 (commit-LoRA OFF): empty list.
            #   - commit-active sub-blocks: extract per-token log-pi from
            #     the captured pre-softmax logits + committed tokens, OR
            #     fall back to zero log-probs of the right length when
            #     committed_logits is None (mock / EMIT_LOGPROBS unwired).
            lp_per_sb: List[Any] = []
            for sb_idx in range(num_sub_blocks):
                tokens_here = captured_tokens[sb_idx]
                logits_here = captured_logits[sb_idx]
                if not (1 <= sb_idx <= commit_n_blocks) or not tokens_here:
                    lp_per_sb.append([])
                    continue
                if logits_here is not None and _torch_available:
                    lp_per_sb.append(
                        _logits_to_token_logprobs(logits_here, tokens_here)
                    )
                else:
                    # Mock or EMIT_LOGPROBS-unwired: zero placeholder so
                    # downstream grpo_loss .stack() still finds a tensor.
                    lp_per_sb.append(_zero_logprobs_like(len(tokens_here)))

            reward = 1.0 if is_correct(text, gold) else 0.0
            results.append(RolloutResult(captured_tokens, lp_per_sb, text, reward))
    finally:
        if prev_emit is None:
            os.environ.pop("EMIT_LOGPROBS", None)
        else:
            os.environ["EMIT_LOGPROBS"] = prev_emit

    return results


# ----------------------------------------------------------------------------
# GRPO loss
# ----------------------------------------------------------------------------
def grpo_loss(
    rollouts: List[Tuple[List[List[int]], List[List[Any]], str, float]],
    *,
    commit_n_blocks: int = COMMIT_N_BLOCKS,
    advantage_eps: float = ADV_EPS,
) -> Tuple[Any, dict]:
    """Compute GRPO policy-gradient loss for one prompt's M rollouts.

    rollouts[i] = (committed_per_sb, log_probs_per_sb, text, reward).

    Loss = -sum_i sum_{t in 1..commit_n_blocks} sum_tau logpi(commit_{t,tau,i}) * A_i

    KL anchor (beta * KL(pi || pi_frozen)) is added by the caller (it
    requires a second forward pass with `peft.disable_adapter_layers()`
    and is GPU-side).

    Returns (loss_tensor_or_float, stats_dict).

    Pure-CPU compatible: if log_probs are floats (not torch tensors),
    returns a python float - this is the smoke path. On GPU the log_probs
    are torch.Tensors with grad; the same code returns a tensor.
    """
    rewards = [float(r[3]) for r in rollouts]
    advantages = grpo_advantages(rewards, eps=advantage_eps)

    is_torch = False
    try:
        import torch
        is_torch = isinstance(
            rollouts[0][1][1][0] if rollouts and rollouts[0][1][1] else 0.0,
            torch.Tensor,
        )
    except Exception:
        is_torch = False

    if is_torch:
        import torch
        loss_terms: List[Any] = []
        for (_, log_probs_per_sb, _, _), A in zip(rollouts, advantages):
            for sb in range(1, commit_n_blocks + 1):
                if sb >= len(log_probs_per_sb):
                    continue
                lp_list = log_probs_per_sb[sb]
                if not lp_list:
                    continue
                stacked = torch.stack(list(lp_list))  # (T,)
                loss_terms.append(-stacked.sum() * A)
        loss = torch.stack(loss_terms).sum() if loss_terms else torch.zeros((), requires_grad=True)
    else:
        loss = 0.0
        for (_, log_probs_per_sb, _, _), A in zip(rollouts, advantages):
            for sb in range(1, commit_n_blocks + 1):
                if sb >= len(log_probs_per_sb):
                    continue
                for lp in log_probs_per_sb[sb]:
                    loss = loss - float(lp) * A

    n_r = max(1, len(rewards))
    mean_r = sum(rewards) / n_r
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / n_r) if rewards else 0.0
    stats = {
        "reward_mean": mean_r,
        "reward_std": std_r,
        "advantage_max": max(advantages) if advantages else 0.0,
        "advantage_min": min(advantages) if advantages else 0.0,
    }
    return loss, stats


# ----------------------------------------------------------------------------
# MATH-500 dev check (subprocess shell-out to e4/runner.py)
# ----------------------------------------------------------------------------
def run_math500_dev_check(adapter_dir: Path, n: int = MATH500_DEV_N) -> float:
    """Run cmajc-k3 forward-only check on MATH-500 dev N rows.

    Shells out to `e4/runner.py` (the existing runner); parses the
    summary line for accuracy. Returns acc in [0, 1] or NaN on failure.

    GPU-only: this function is a no-op when MOCK_MODELS=1.
    """
    if os.environ.get("MOCK_MODELS") == "1":
        return float("nan")
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "e4" / "runner.py"),
        "--dataset", "math500",
        "--n", str(n),
        "--mechanism", "cmajc",
        "--k", "3",
        "--commit-lora", str(adapter_dir),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    except Exception as exc:
        print(f"[check] subprocess failed: {exc}", file=sys.stderr)
        return float("nan")
    for line in out.stdout.splitlines()[::-1]:
        if "accuracy:" in line.lower():
            try:
                raw = line.split(":")[-1].strip()
                if raw.endswith("%"):
                    return float(raw.rstrip("%")) / 100.0
                return float(raw)
            except Exception:
                continue
    return float("nan")


# ----------------------------------------------------------------------------
# Smoke entrypoint (no model load, runs in <5s)
# ----------------------------------------------------------------------------
def _smoke() -> int:
    print("[smoke] === Phase-4 Direction A trainer smoke ===", flush=True)
    fail = []

    # ---- 1. GRPO advantage math ----
    rewards = [1.0, 0.0, 1.0, 1.0]
    A = grpo_advantages(rewards)
    expected = [0.5773, -1.7320, 0.5773, 0.5773]  # mean=0.75, std=sqrt(0.1875)~0.4330
    print(f"[smoke] rewards={rewards} -> advantages={[f'{a:.4f}' for a in A]}", flush=True)
    if len(A) != 4:
        fail.append(f"advantages len {len(A)} != 4")
    for got, exp in zip(A, expected):
        if abs(got - exp) > 0.01:
            fail.append(f"advantage {got:.4f} != expected {exp:.4f}")
    mean_check = sum(rewards) / len(rewards)
    if abs(mean_check - 0.75) > 1e-9:
        fail.append(f"mean {mean_check} != 0.75")

    # ---- 2. Jaccard tripwire (positive case: 21 lists, last 20 identical) ----
    same = list(range(8))
    pos_input = [[42] * 8] + [list(same) for _ in range(20)]
    tripped_pos = _check_jaccard_tripwire(pos_input)
    print(f"[smoke] tripwire(21 lists, last 20 identical) -> tripped={tripped_pos}", flush=True)
    if not tripped_pos:
        fail.append("tripwire FAILED to fire on 20 identical commit-token lists")

    # ---- 3. Jaccard tripwire (negative case: all distinct) ----
    neg_input = [[i * 100 + j for j in range(8)] for i in range(21)]
    tripped_neg = _check_jaccard_tripwire(neg_input)
    print(f"[smoke] tripwire(21 disjoint lists) -> tripped={tripped_neg}", flush=True)
    if tripped_neg:
        fail.append("tripwire FALSE-POSITIVE on disjoint commit-token lists")

    # ---- 4. grpo_loss on synthetic rollouts (CPU path, float log_probs) ----
    fake_rollouts = []
    for r in rewards:
        committed = [[1, 2, 3] for _ in range(NUM_SUB_BLOCKS)]
        log_probs = [
            [-1.0, -1.0, -1.0] if 1 <= sb <= COMMIT_N_BLOCKS else []
            for sb in range(NUM_SUB_BLOCKS)
        ]
        fake_rollouts.append((committed, log_probs, "mock", r))
    loss, stats = grpo_loss(fake_rollouts)
    print(f"[smoke] grpo_loss(fake rollouts) -> loss={loss:.4f} stats={stats}", flush=True)
    if not math.isfinite(loss):
        fail.append(f"grpo_loss returned non-finite: {loss}")
    if abs(stats["reward_mean"] - 0.75) > 1e-9:
        fail.append(f"reward_mean {stats['reward_mean']} != 0.75")

    # ---- 5. _rollout returns expected shape under MOCK_MODELS ----
    os.environ["MOCK_MODELS"] = "1"
    out = _rollout(model=None, tokenizer=None, prompt="2+2?", gold="4", m=4)
    if len(out) != 4:
        fail.append(f"_rollout returned {len(out)} rollouts, expected 4")
    for i, (committed_per_sb, lp_per_sb, text, reward) in enumerate(out):
        if len(committed_per_sb) != NUM_SUB_BLOCKS:
            fail.append(
                f"rollout {i} committed_per_sb len {len(committed_per_sb)} != {NUM_SUB_BLOCKS}"
            )
        if len(lp_per_sb) != NUM_SUB_BLOCKS:
            fail.append(f"rollout {i} lp_per_sb len {len(lp_per_sb)} != {NUM_SUB_BLOCKS}")
        # Sub-block 0 (commit-LoRA OFF) should have empty log_probs.
        if lp_per_sb[0]:
            fail.append(
                f"rollout {i} sub-block 0 has nonempty log_probs (commit-LoRA should be OFF)"
            )
        if reward not in (0.0, 1.0):
            fail.append(f"rollout {i} reward {reward} not binary")

    # ---- 6. _extract_layer_idx ----
    cases = [
        ("base_model.model.transformer.layers.27.feed_forward.ff_proj.lora_A.default", 27),
        ("model.layers.31.up_proj.lora_A.default", 31),
        ("not.a.layer.path", None),
    ]
    for name, expected_idx in cases:
        got_idx = _extract_layer_idx(name)
        if got_idx != expected_idx:
            fail.append(f"_extract_layer_idx({name!r}) = {got_idx} != {expected_idx}")

    # ---- 7. committed_logits → per-token log-probs (Blocker 1) ----
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        torch.manual_seed(0)
        T, V = 8, 1000
        fake_logits = torch.randn(T, V)
        fake_tokens = [int(torch.randint(0, V, (1,)).item()) for _ in range(T)]
        lps = _logits_to_token_logprobs(fake_logits, fake_tokens)
        # Reference computation.
        ref = F.log_softmax(fake_logits, dim=-1)[range(T), torch.tensor(fake_tokens)]
        if lps.shape != (T,):
            fail.append(f"smoke7 logprobs shape {tuple(lps.shape)} != ({T},)")
        if not torch.isfinite(lps).all():
            fail.append("smoke7 non-finite log-probs")
        if not torch.allclose(lps, ref, atol=1e-6):
            fail.append("smoke7 log-probs disagree with reference")
        # All log-probs must be <= 0 (log of a probability).
        if (lps > 0).any():
            fail.append("smoke7 log-probs not all <= 0")
        print(
            f"[smoke] committed_logits→logprobs: shape={tuple(lps.shape)} "
            f"min={float(lps.min()):.3f} max={float(lps.max()):.3f}",
            flush=True,
        )
    except ImportError:
        print("[smoke] torch unavailable — smoke 7 skipped", flush=True)

    # ---- 8. PhaseEmbedding monkey-patch (Blocker 2) ----
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        torch.manual_seed(1)
        # tiny 8→4 linear with zero weights/bias so its forward returns 0.
        lin = nn.Linear(8, 4)
        with torch.no_grad():
            lin.weight.zero_()
            lin.bias.zero_()
        # Stand up a PhaseEmbedding-style monkey-patch directly (no PEFT
        # model needed; we don't even need to instantiate PhaseEmbedding —
        # we exercise the same wrapping pattern).
        emb = nn.Parameter(torch.eye(4))  # phase_emb[t] = e_t one-hot
        num_phases = 4

        def _wrapped(self_mod, x):
            out = self_mod._orig(x)
            t = _pes.get_sub_block()
            disabled = getattr(self_mod, "_sfumato_adapter_disabled", False)
            if t is None or t == 0 or disabled:
                return out
            t_idx = max(0, min(num_phases - 1, int(t)))
            return out + emb[t_idx].to(out.dtype).to(out.device)

        lin._orig = lin.forward
        lin.forward = MethodType(_wrapped, lin)

        x_in = torch.zeros(1, 8)

        # t=0 → no add (skip rule).
        with phase_emb_subblock(0):
            y0 = lin(x_in)
        if not torch.allclose(y0, torch.zeros(1, 4)):
            fail.append(f"smoke8 t=0 expected zeros, got {y0.tolist()}")

        # t=1 → output equals phase_emb[1] = [0, 1, 0, 0].
        with phase_emb_subblock(1):
            y1 = lin(x_in)
        if not torch.allclose(y1, torch.tensor([[0.0, 1.0, 0.0, 0.0]])):
            fail.append(f"smoke8 t=1 expected [0,1,0,0], got {y1.tolist()}")

        # t=2 → [0, 0, 1, 0].
        with phase_emb_subblock(2):
            y2 = lin(x_in)
        if not torch.allclose(y2, torch.tensor([[0.0, 0.0, 1.0, 0.0]])):
            fail.append(f"smoke8 t=2 expected [0,0,1,0], got {y2.tolist()}")

        # t=3 → [0, 0, 0, 1].
        with phase_emb_subblock(3):
            y3 = lin(x_in)
        if not torch.allclose(y3, torch.tensor([[0.0, 0.0, 0.0, 1.0]])):
            fail.append(f"smoke8 t=3 expected [0,0,0,1], got {y3.tolist()}")

        # disabled flag → skip even when t is set.
        lin._sfumato_adapter_disabled = True
        with phase_emb_subblock(2):
            y_dis = lin(x_in)
        if not torch.allclose(y_dis, torch.zeros(1, 4)):
            fail.append(f"smoke8 disabled expected zeros, got {y_dis.tolist()}")
        lin._sfumato_adapter_disabled = False

        # t=None outside any context → no add.
        y_none = lin(x_in)
        if not torch.allclose(y_none, torch.zeros(1, 4)):
            fail.append(f"smoke8 t=None expected zeros, got {y_none.tolist()}")

        print("[smoke] phase_emb monkey-patch: t=0/1/2/3 + disabled all OK", flush=True)
    except ImportError:
        print("[smoke] torch unavailable — smoke 8 skipped", flush=True)

    # ---- 9. KL k3 estimator (Blocker 3) ----
    try:
        import torch  # type: ignore

        torch.manual_seed(2)
        lp_theta = torch.randn(16) - 2.0  # log-probs ≤ 0 ish
        kl_zero = kl_k3_estimator(lp_theta, lp_theta.clone())
        if abs(float(kl_zero)) > 1e-6:
            fail.append(f"smoke9 KL(p||p) expected ~0, got {float(kl_zero)}")
        # Non-zero KL when frozen != theta.
        lp_frozen = lp_theta + 0.5
        kl_pos = kl_k3_estimator(lp_theta, lp_frozen)
        if float(kl_pos) <= 0:
            fail.append(f"smoke9 KL(p||q≠p) expected > 0, got {float(kl_pos)}")
        # Symmetric direction also positive.
        kl_neg = kl_k3_estimator(lp_theta, lp_theta - 0.5)
        if float(kl_neg) <= 0:
            fail.append(f"smoke9 KL(p||q-) expected > 0, got {float(kl_neg)}")
        print(
            f"[smoke] KL k3: KL(p||p)={float(kl_zero):.2e} "
            f"KL(p||p+0.5)={float(kl_pos):.4f} KL(p||p-0.5)={float(kl_neg):.4f}",
            flush=True,
        )
    except ImportError:
        print("[smoke] torch unavailable — smoke 9 skipped", flush=True)

    # ---- 10. _rollout real-path on _Mock diff-model (GPU plumbing exercised) ----
    # Drive _rollout through `e4.diff_llada.load(mock=True)` so the actual
    # denoise_block + step_callback contract is exercised. The mock doesn't
    # populate committed_logits (always None), so we expect zero-tensor /
    # empty-list log_probs; what we're verifying is the shape contract +
    # that the real-path code branches without raising.
    try:
        import e4.diff_llada as _diff  # type: ignore

        os.environ["MOCK_MODELS"] = "1"
        mock_diff = _diff.load("mock-llada", mock=True)
        out_real = _rollout(
            model=mock_diff,
            tokenizer=None,
            prompt="Mock problem 1: 2 + 1 = ?",
            gold="3",
            m=4,
            seed=0,
        )
        if len(out_real) != 4:
            fail.append(f"smoke10 _rollout returned {len(out_real)} rollouts, expected 4")
        for i, res in enumerate(out_real):
            # NamedTuple unpacks like a tuple.
            committed_per_sb, lp_per_sb, text, reward = res
            # Also confirm the named-field access works.
            if res.final_text != text:
                fail.append(f"smoke10 rollout {i} named-field/tuple mismatch")
            if len(committed_per_sb) != NUM_SUB_BLOCKS:
                fail.append(
                    f"smoke10 rollout {i} committed_per_sb len {len(committed_per_sb)} "
                    f"!= {NUM_SUB_BLOCKS}"
                )
            if len(lp_per_sb) != NUM_SUB_BLOCKS:
                fail.append(
                    f"smoke10 rollout {i} lp_per_sb len {len(lp_per_sb)} "
                    f"!= {NUM_SUB_BLOCKS}"
                )
            # Sub-block 0 (commit-LoRA OFF) must be empty.
            if lp_per_sb[0]:
                fail.append(f"smoke10 rollout {i} sub-block 0 lp not empty")
            # Sub-blocks 1..commit_n_blocks should have a tensor / list with
            # the same length as committed_per_sb[sb] (mock fills 32 tokens).
            for sb in range(1, COMMIT_N_BLOCKS + 1):
                expected_len = len(committed_per_sb[sb])
                got = lp_per_sb[sb]
                # Either a torch.Tensor or python list — both have len.
                got_len = (
                    int(got.shape[0])
                    if hasattr(got, "shape") and len(getattr(got, "shape", ())) >= 1
                    else len(got)
                )
                if got_len != expected_len:
                    fail.append(
                        f"smoke10 rollout {i} sb{sb} lp len {got_len} != "
                        f"committed len {expected_len}"
                    )
            if not isinstance(text, str) or not text:
                fail.append(f"smoke10 rollout {i} final_text empty/not-str: {text!r}")
            if reward not in (0.0, 1.0):
                fail.append(f"smoke10 rollout {i} reward {reward} not binary")
        # Determinism check: same seed → same final text across two calls.
        out_b = _rollout(
            model=mock_diff,
            tokenizer=None,
            prompt="Mock problem 1: 2 + 1 = ?",
            gold="3",
            m=4,
            seed=0,
        )
        if [r.final_text for r in out_real] != [r.final_text for r in out_b]:
            fail.append("smoke10 _rollout not deterministic for fixed seed")
        print(
            f"[smoke] _rollout real-path on _Mock: M={len(out_real)} "
            f"sub_blocks={NUM_SUB_BLOCKS} reward_set={sorted({r.reward for r in out_real})}",
            flush=True,
        )
    except ImportError:
        print("[smoke] e4.diff_llada unavailable — smoke 10 skipped", flush=True)

    # ---- 12. _rescore_committed_logprobs grad path (real torch grad) ----
    # Build a tiny stand-in model that mimics LLaDA's `model(input_ids).logits`
    # contract, plug it into a _Real instance, and verify the re-score
    # function returns gradient-bearing log-probs that touch the model's
    # trainable params. Without this the policy gradient is a no-op even
    # if optimizer.step runs.
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from types import SimpleNamespace

        class _TinyLLaDA(nn.Module):
            def __init__(self, vocab=128):
                super().__init__()
                self.embed = nn.Embedding(vocab, 16)
                self.head = nn.Linear(16, vocab)

            def forward(self, input_ids):
                h = self.embed(input_ids)
                logits = self.head(h)
                return SimpleNamespace(logits=logits)

        class _TinyTokenizer:
            def __init__(self, vocab=128):
                self.vocab = vocab

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                text = " ".join(m["content"] for m in messages) + " [GEN]"
                return [hash(w) % self.vocab for w in text.split()]

        torch.manual_seed(7)
        tiny = _TinyLLaDA(vocab=128)
        diff_stub = SimpleNamespace(_model=tiny, _tokenizer=_TinyTokenizer(128))

        # 4 sub-blocks, each with 8 committed tokens (sub_block_length=8 here).
        committed = [[(i * 7 + j) % 128 for j in range(8)] for i in range(4)]
        lp_per_sb = _rescore_committed_logprobs(
            diff_stub, "What is 2 + 3?", committed,
            num_sub_blocks=4, commit_n_blocks=3, sub_block_length=8,
            mask_id=0,  # tiny vocab, override LLaDA's 126336 default
        )
        # Verify shape contract.
        if len(lp_per_sb) != 4:
            fail.append(f"smoke12 lp_per_sb len {len(lp_per_sb)} != 4")
        for sb in range(1, 4):
            t = lp_per_sb[sb]
            if not torch.is_tensor(t) or t.dim() != 1 or t.shape[0] != 8:
                fail.append(f"smoke12 sub-block {sb}: bad shape {getattr(t, 'shape', t)!r}")
        # Verify autograd path: a sum of all log-pi should produce non-zero
        # grads on the tiny model's head bias.
        loss = sum(t.sum() for sb, t in enumerate(lp_per_sb) if torch.is_tensor(t) and t.numel())
        loss.backward()
        head_grad = tiny.head.weight.grad
        if head_grad is None or head_grad.abs().sum().item() == 0.0:
            fail.append("smoke12 backward produced no grad on tiny.head.weight")
        print(
            f"[smoke] _rescore grad path OK: lp[1..3] shapes "
            f"{[lp_per_sb[s].shape for s in (1,2,3)]} "
            f"head_grad_l1={head_grad.abs().sum().item():.3f}",
            flush=True,
        )
    except ImportError:
        print("[smoke] torch unavailable — smoke 12 skipped", flush=True)

    # ---- 11. phase_emb_state env-gate + writer wiring via _Mock ----
    # Verifies that:
    #   (a) e4.phase_emb_state.set_sub_block respects the SCHEDULE_RL gate
    #       (no-op when SCHEDULE_RL is unset; writes when set).
    #   (b) e4.diff_llada._Mock.denoise_block fires the writer at each
    #       sub-block boundary so a step_callback can OBSERVE
    #       _pes.get_sub_block() == b_idx between callbacks.
    #   (c) After denoise_block returns, the LoRA-A monkey-patch reader
    #       (smoke 8 already covered the math; this one verifies the
    #       end-to-end value path via the shared module).
    try:
        import e4.diff_llada as _diff  # type: ignore
        from e4 import phase_emb_state as _pes_t  # type: ignore

        # (a) gate off → no-op writes.
        os.environ.pop("SCHEDULE_RL", None)
        _pes_t.clear()
        _pes_t.set_sub_block(2)
        if _pes_t.get_sub_block() is not None:
            fail.append(
                f"smoke11 SCHEDULE_RL gate failed: set_sub_block(2) wrote "
                f"{_pes_t.get_sub_block()!r} when gate is off"
            )

        # (b) gate on → writer fires from inside _Mock.denoise_block; capture
        # the sub_block-index sequence the runner publishes BETWEEN sub-block
        # callbacks (i.e. just before each callback, the writer sets t=b_idx).
        os.environ["SCHEDULE_RL"] = "1"
        _pes_t.clear()
        observed: List[Optional[int]] = []
        # The writer fires BEFORE the StepState is built (see _Mock loop body
        # in e4/diff_llada.py). The callback runs AFTER, so by callback-time
        # _pes.get_sub_block() should equal state.sub_block.
        def _cb_obs(state: Any) -> Any:
            observed.append(_pes_t.get_sub_block())
            return None

        mock_diff_b = _diff.load("mock-llada-pes", mock=True)
        mock_diff_b.denoise_block(
            prompt="Mock problem 7: 2 + 7 = ?",
            k_steps=64,
            seed=42,
            temperature=0.0,
            apply_commit=True,
            commit_n_blocks=3,
            step_callback=_cb_obs,
        )
        # Expect 4 callbacks (one per sub-block) with t == sub_block index.
        if observed != [0, 1, 2, 3]:
            fail.append(
                f"smoke11 writer wiring failed: observed {observed!r} "
                f"!= [0, 1, 2, 3]"
            )

        # (c) After the loop, the thread-local should hold the LAST sub-block
        # index (no automatic clear). main_train can `_pes.clear()` between
        # rollouts if it wants the slate clean.
        if _pes_t.get_sub_block() != 3:
            fail.append(
                f"smoke11 final state expected 3, got {_pes_t.get_sub_block()!r}"
            )

        # Restore the env to a clean slate so subsequent test runs / smokes
        # don't accidentally rely on the gate.
        os.environ.pop("SCHEDULE_RL", None)
        _pes_t.clear()

        print(
            f"[smoke] phase_emb_state writer wiring: SCHEDULE_RL gate OK, "
            f"observed sub-blocks {observed}",
            flush=True,
        )
    except ImportError:
        print("[smoke] e4.diff_llada/phase_emb_state unavailable — smoke 11 skipped", flush=True)

    # ---- verdict ----
    if fail:
        print(f"[smoke] FAIL: {len(fail)} failures:", flush=True)
        for f in fail:
            print(f"   - {f}", flush=True)
        return 1
    print("[smoke] all smokes pass", flush=True)
    return 0


# ----------------------------------------------------------------------------
# Main training loop (GPU; sketched - see TODOs in module docstring)
# ----------------------------------------------------------------------------
def main_train() -> int:
    """Real GPU training entrypoint.

    Loads LLaDA + commit-LoRA via `e4.diff_llada._Real` so the rollout
    path (which calls `model.denoise_block(...)`) and the optimizer
    (which steps on `peft_model.parameters()`) both reference the SAME
    underlying peft-wrapped HF model. Rollouts are fired through the
    diff-model wrapper; backward + optimizer.step touch the underlying
    PEFT params directly. Setting `SCHEDULE_RL=1` at startup so the
    diff_llada inner-loop writer publishes the current sub-block index
    to the phase_emb thread-local on every diffusion forward.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 2

    # CRITICAL env setup: writer side of phase_emb thread-local + un-merged
    # LoRA so backward through lora_A/B is non-trivial + cuda allocator
    # tweak so 24GB GPUs don't fragment-OOM during the per-(rollout,sb)
    # rescore forward.
    os.environ["SCHEDULE_RL"] = "1"
    os.environ["EMIT_LOGPROBS"] = "1"
    os.environ["MERGE_ADAPTER"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- diff-model + LoRA (trainable!) --
    # We bypass diff_llada.load() because that path freezes everything
    # (is_trainable=False). Instead: load the HF model directly, wrap with
    # commit-LoRA via PeftModel.from_pretrained(is_trainable=True), then
    # stuff into a _Real instance so denoise_block / step_callback all work.
    print(f"Loading base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype
    )
    if torch.cuda.is_available():
        base_model = base_model.to("cuda")
    base_model.requires_grad_(False)

    if not RESUME_FROM:
        print("ERROR: RESUME_FROM is required (commit-LoRA v3 to fine-tune).", file=sys.stderr)
        return 2
    print(f"Resuming commit-LoRA from {RESUME_FROM} with is_trainable=True", flush=True)
    _hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    peft_model = PeftModel.from_pretrained(
        base_model, RESUME_FROM,
        is_trainable=True, adapter_name="commit", token=_hf_token,
    )

    # Wrap into a _Real instance so _rollout's `.denoise_block` works.
    import e4.diff_llada as _diff
    diff_model = _diff._Real(name=MODEL_NAME)
    diff_model._model = peft_model
    diff_model._tokenizer = tokenizer

    # -- phase emb (attaches to peft_model so backward flows through) --
    phase_emb = PhaseEmbedding(num_phases=NUM_SUB_BLOCKS, lora_rank=LORA_R)
    n_hooked = phase_emb.attach(peft_model)
    expected_hooks = len(LORA_LAYERS_TO_TRANSFORM) * len(LORA_TARGETS)
    print(f"Phase-emb hooks installed: {n_hooked} (expected {expected_hooks})", flush=True)
    if torch.cuda.is_available():
        phase_emb.param.data = phase_emb.param.data.to("cuda")

    # -- optimizer (LoRA params + phase_emb) --
    trainable = [p for p in peft_model.parameters() if p.requires_grad] + [phase_emb.param]
    n_train = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_train:,} ({len(trainable)} tensors)", flush=True)
    optim = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    # -- substrate --
    import json as _json
    rows = []
    with open(SUBSTRATE_JSONL) as f:
        for line in f:
            rows.append(_json.loads(line))
    # MINI_PILOT slice: first MINI_PILOT_N rows when env set.
    mini_pilot_n = _env_int("MINI_PILOT_N", 0)
    if mini_pilot_n > 0:
        rows = rows[:mini_pilot_n]
        print(f"[MINI_PILOT] using first {len(rows)} rows of substrate", flush=True)
    print(f"Substrate: {len(rows)} prompts from {SUBSTRATE_JSONL}", flush=True)
    # Use the diff-model wrapper for rollouts.
    model = diff_model

    tripwire = JaccardTripwire()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    total_steps = EPOCHS * len(rows)
    global_step = 0
    start = time.monotonic()

    for epoch in range(EPOCHS):
        rng = random.Random(SEED + epoch)
        order = list(range(len(rows)))
        rng.shuffle(order)
        for idx in order:
            row = rows[idx]
            # Rollout under no_grad so the M denoise calls don't blow VRAM.
            with torch.no_grad():
                rollouts = _rollout(
                    model, tokenizer, row["question"], row["gold"], m=M_ROLLOUTS,
                )

            # -- tripwire --
            for committed_per_sb, _, _, _ in rollouts:
                # Concatenate commit-active sub-blocks 1..3.
                commit_ids: List[int] = []
                for sb in range(1, COMMIT_N_BLOCKS + 1):
                    if sb < len(committed_per_sb):
                        commit_ids.extend(committed_per_sb[sb])
                tripwire.add(prompt_id=row["id"], commit_token_ids=commit_ids)
            tripped, sim = tripwire.tripped()
            if tripped:
                marker = SAVE_DIR / "JACCARD_TRIPPED"
                marker.write_text(
                    f"step={global_step} mean_jaccard={sim:.4f} threshold={JACCARD_THRESHOLD}\n"
                )
                print(
                    f"[tripwire] HALT: mean Jaccard={sim:.4f} > {JACCARD_THRESHOLD}",
                    flush=True,
                )
                return 3

            # -- restore adapter forward path + LoRA grads after rollout --
            # diff_llada._Real.denoise_block calls _disable_commit() at the
            # end of each rollout, which (a) disables the LoRA branch in
            # forward and (b) sets requires_grad=False on commit-LoRA params
            # under PEFT 0.19. Re-enable both so the rescore forward sees
            # the LoRA branch and backward can update lora_A/B.
            peft_model.enable_adapter_layers()
            peft_model.set_adapter("commit")
            for p_name, p in peft_model.named_parameters():
                if ".lora_" in p_name and ".commit" in p_name:
                    p.requires_grad_(True)

            # -- per-(rollout, sub-block) rescore + backward --
            # Doing one big rescore for all M rollouts × T commit-active
            # sub-blocks blows VRAM on 24GB. Instead: forward + backward +
            # free per sub-block, accumulating grads on the trainable
            # tensors. optim.step() is called after the whole prompt's M
            # rollouts have contributed.
            from e4 import phase_emb_state as _pes
            sub_block_length = 32
            mask_id = _diff._LLADA_MASK_ID
            optim.zero_grad(set_to_none=True)
            advantages = grpo_advantages(
                [float(r[3]) for r in rollouts], eps=ADV_EPS
            )

            n_backward = 0
            total_loss_scalar = 0.0
            for r_i, (committed_per_sb, _, _, _) in enumerate(rollouts):
                A = advantages[r_i]
                # Pre-build the chat-templated prompt prefix once per rollout.
                messages = [
                    {"role": "system", "content": _diff._DENOISE_SYS},
                    {"role": "user", "content": row["question"]},
                ]
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                prompt_len = len(prompt_token_ids)

                def _pad(ids, n=sub_block_length):
                    ids = list(ids)
                    return ids[:n] if len(ids) >= n else ids + [mask_id] * (n - len(ids))

                padded = [
                    _pad(committed_per_sb[sb] if sb < len(committed_per_sb) else [])
                    for sb in range(NUM_SUB_BLOCKS)
                ]
                for t in range(1, COMMIT_N_BLOCKS + 1):
                    committed_t = list(committed_per_sb[t]) if t < len(committed_per_sb) else []
                    if not committed_t:
                        continue
                    seq = list(prompt_token_ids)
                    for sb in range(NUM_SUB_BLOCKS):
                        seq.extend(padded[sb] if sb < t else [mask_id] * sub_block_length)
                    full_ids = torch.tensor([seq], dtype=torch.long, device="cuda")
                    sb_start = prompt_len + t * sub_block_length
                    n_committed = len(committed_t)
                    committed_ids = torch.tensor(committed_t, dtype=torch.long, device="cuda")

                    # Forward A: adapter ON, phase_emb ON, grad enabled.
                    _pes.force_set_sub_block(t)
                    try:
                        outputs_a = peft_model(full_ids)
                        logits_a = outputs_a.logits if hasattr(outputs_a, "logits") else outputs_a[0]
                        sb_logits_a = logits_a[0, sb_start : sb_start + n_committed]
                        log_p_a = torch.nn.functional.log_softmax(sb_logits_a, dim=-1)
                        token_lp_theta = log_p_a.gather(1, committed_ids.unsqueeze(1)).squeeze(1)
                    finally:
                        _pes.clear()

                    # Forward B (KL anchor, no grad): adapter OFF, phase_emb skip.
                    # Schulman k3 KL = (exp(-delta) - 1 + delta), delta = log_pi_theta - log_pi_frozen.
                    # KL is always >= 0 by construction. Gradient flows ONLY through
                    # log_pi_theta (forward A path); log_pi_frozen is detached.
                    if KL_BETA > 0.0:
                        peft_model.disable_adapter_layers()
                        phase_emb.set_adapter_disabled(True)
                        try:
                            with torch.no_grad():
                                outputs_b = peft_model(full_ids)
                                logits_b = outputs_b.logits if hasattr(outputs_b, "logits") else outputs_b[0]
                                sb_logits_b = logits_b[0, sb_start : sb_start + n_committed]
                                log_p_b = torch.nn.functional.log_softmax(sb_logits_b, dim=-1)
                                token_lp_frozen = log_p_b.gather(1, committed_ids.unsqueeze(1)).squeeze(1)
                            del outputs_b, logits_b, sb_logits_b, log_p_b
                        finally:
                            phase_emb.set_adapter_disabled(False)
                            peft_model.enable_adapter_layers()
                            peft_model.set_adapter("commit")
                            # disable_adapter_layers also flips requires_grad off; restore.
                            for p_name, p in peft_model.named_parameters():
                                if ".lora_" in p_name and ".commit" in p_name:
                                    p.requires_grad_(True)
                        kl_term = kl_k3_estimator(token_lp_theta, token_lp_frozen).sum()
                    else:
                        kl_term = torch.zeros((), device=token_lp_theta.device)

                    pg_loss = -(token_lp_theta.sum()) * float(A)
                    sb_loss = pg_loss + KL_BETA * kl_term
                    total_loss_scalar += float(sb_loss.detach())
                    sb_loss.backward()
                    n_backward += 1

                    del outputs_a, logits_a, sb_logits_a, log_p_a, token_lp_theta
                    if KL_BETA > 0.0:
                        del token_lp_frozen, kl_term
                    del pg_loss, sb_loss, full_ids, committed_ids
                    torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            global_step += 1

            mean_r = sum(float(r[3]) for r in rollouts) / max(1, len(rollouts))
            if global_step % LOG_INTERVAL == 0 or global_step <= 5:
                print(
                    f"step:{global_step}/{total_steps} train_loss:{total_loss_scalar:.4f} "
                    f"reward_mean:{mean_r:.3f} jaccard:{sim:.3f} n_backward:{n_backward}",
                    flush=True,
                )

            # -- early-stop check --
            if EVAL_INTERVAL > 0 and global_step % EVAL_INTERVAL == 0:
                tmp = SAVE_DIR / f"step_{global_step}"
                tmp.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(tmp))
                acc = run_math500_dev_check(tmp, n=MATH500_DEV_N)
                print(
                    f"step:{global_step}/{total_steps} val_loss:{1.0 - acc:.4f} val_bpb:{global_step}",
                    flush=True,
                )
                if math.isfinite(acc) and acc < (MATH500_BASELINE - MATH500_REGRESSION_PP):
                    print(
                        f"[earlystop] MATH-500 acc {acc:.3f} < baseline-{MATH500_REGRESSION_PP:.2f}",
                        flush=True,
                    )
                    return 4

    # -- final save --
    # `model` is the `_Real` diff_llada wrapper; save the underlying peft
    # model (which holds the LoRA adapter weights). Also dump phase_emb.
    peft_model.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    torch.save(phase_emb.param.detach().cpu(), Path(SAVE_DIR) / "phase_emb.pt")
    total_bytes = sum(p.stat().st_size for p in SAVE_DIR.rglob("*") if p.is_file())
    print(f"Serialized model {SAVE_DIR} {total_bytes} bytes", flush=True)
    elapsed = time.monotonic() - start
    print(f"done in {elapsed:.1f}s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU-only smoke (no model load)")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    return main_train()


if __name__ == "__main__":
    sys.exit(main())
