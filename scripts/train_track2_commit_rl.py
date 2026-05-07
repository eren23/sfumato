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

OPEN TODOs (GPU-blocking, not sketched here):
  - per-sub-block log-prob capture: e4/diff_llada._generate currently dumps
    a StepState at each sub-block boundary but does NOT preserve the
    pre-softmax logits over the committed positions. Needs a new field
    `committed_logits: torch.Tensor | None` on StepState plus a sampler
    branch that stashes the logits before the argmax/sample step. Wire it
    through cmajc_branch_inplace as well (the branch winner's path is
    what we want gradients on, not the losers').
  - phase_emb integration with PEFT LoraLayer: cleanest is a forward_pre_hook
    on each LoraLayer in layers 24-31 that ADDS phase_emb[t] to the input
    tensor of the LoRA-A matmul. Phase index t is set on a model attribute
    `model._sfumato_phase_idx` per sub-block by the rollout harness BEFORE
    each `_generate` sub-block boundary callback fires.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

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
class PhaseEmbedding:
    """Schedule-phase embedding: 4-vector phase_emb[t] added as bias to LoRA-A.

    Owns:
      - nn.Parameter of shape (NUM_SUB_BLOCKS, lora_rank).
      - List of forward_pre_hooks installed on each LoRA-A linear in
        layers 24-31. The hook reads `model._sfumato_phase_idx` (set by
        the rollout harness before each sub-block) and stashes the phase
        vector for the LoRA-A forward to consume.

    The phase index is threaded via a model attribute (NOT thread-local)
    because PEFT's LoraLayer forward is called from the same thread as
    `_generate`. Using an attribute keeps the data path trivial to inspect.

    TODO: PEFT's LoraLayer wraps the base linear and then applies LoRA-A
    inside its own forward. The exact handle for "input to LoRA-A" depends
    on PEFT version (we pin >=0.11). The likely target is `lora_A.default`
    (a regular nn.Linear). If `disable_adapter_layers()` is active the
    hook becomes a no-op. Smoke validates the math; GPU run validates the
    integration.
    """

    def __init__(self, num_phases: int = NUM_SUB_BLOCKS, lora_rank: int = LORA_R):
        self.num_phases = num_phases
        self.lora_rank = lora_rank
        self.param: Any = None  # set by attach()
        self._hooks: List[Any] = []

    def attach(self, peft_model: Any) -> int:
        """Install hooks on LoRA-A linears in LORA_LAYERS_TO_TRANSFORM.

        Returns the number of hooks installed. Caller is responsible for
        adding `self.param` to the optimizer parameter list.
        """
        import torch
        import torch.nn as nn

        self.param = nn.Parameter(
            torch.zeros(self.num_phases, self.lora_rank, dtype=torch.float32),
            requires_grad=True,
        )
        peft_model._sfumato_phase_emb = self.param
        peft_model._sfumato_phase_idx = 0  # default: sub-block 0 (no commit-LoRA)

        n_hooked = 0
        target_layer_set = set(LORA_LAYERS_TO_TRANSFORM)
        for name, module in peft_model.named_modules():
            # Match PEFT-style LoRA-A linears nested inside the targeted layers.
            # Module name looks like
            # "...layers.{idx}.feed_forward.{ff_proj|up_proj|ff_out}.lora_A.default"
            if not name.endswith("lora_A.default"):
                continue
            layer_idx = _extract_layer_idx(name)
            if layer_idx is None or layer_idx not in target_layer_set:
                continue
            hook = self._make_hook(peft_model)
            self._hooks.append(module.register_forward_pre_hook(hook))
            n_hooked += 1
        return n_hooked

    def _make_hook(self, peft_model: Any) -> Callable:
        emb = self.param  # closure capture

        def _pre_hook(module: Any, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
            # Skip if adapter is disabled (KL-anchor frozen-base pass).
            if getattr(peft_model, "_sfumato_adapter_disabled", False):
                return inputs
            t = int(getattr(peft_model, "_sfumato_phase_idx", 0))
            t = max(0, min(self.num_phases - 1, t))
            x = inputs[0]
            # Stash the phase bias on the module; the post-hook side (or
            # a wrapped LoRA-A forward) is what actually adds it to the
            # rank-r projection. For now just thread it through; the GPU
            # integration will switch to a forward_hook (post) once PEFT
            # internals are confirmed.
            module._sfumato_phase_bias = emb[t].to(x.dtype).to(x.device)
            return inputs

        return _pre_hook

    def detach(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []


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
# Rollout harness (signature complete; inner loop GPU-only - TODO marked)
# ----------------------------------------------------------------------------
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
    commit_n_blocks: int = COMMIT_N_BLOCKS,
    num_sub_blocks: int = NUM_SUB_BLOCKS,
) -> List[Tuple[List[List[int]], List[List[float]], str, float]]:
    """Run M cmajc-k3 rollouts for a single prompt.

    Returns a list of length M, each entry:
        (committed_token_ids_per_subblock, log_probs_per_token, final_text, reward)
      - committed_token_ids_per_subblock: list of length `num_sub_blocks`,
        each a list[int] of the token ids committed in that sub-block.
        Sub-blocks where commit-LoRA is OFF (sub-block 0 by default) still
        return the committed ids (we just don't compute gradients on them).
      - log_probs_per_token: parallel structure to committed ids; per-token
        log pi over the COMMIT-active sub-blocks (1..commit_n_blocks).
        For OFF sub-blocks this is an empty list.
      - final_text: detokenized completion.
      - reward: e4.grade.is_correct(final_text, gold) -> {0.0, 1.0}.

    GPU-only TODO:
      The current `e4/diff_llada._generate` step_callback does NOT expose
      pre-softmax logits at sub-block boundaries - only `tokens_committed`
      and `top_k_logits` (rank-k truncated). To get gradients we need a
      new optional `StepState.committed_logits` field, populated inside
      `_generate` by stashing the logits of the freshly-committed positions
      BEFORE the argmax/sample step. Once shipped, the loop below becomes:

          for i in range(m):
              model._sfumato_phase_idx = 0
              committed_per_sb, lp_per_sb = [], []
              def cb(state):
                  model._sfumato_phase_idx = state.sub_block + 1
                  if state.sub_block + 1 <= commit_n_blocks:
                      lp_per_sb.append(_logits_to_logprobs(state.committed_logits,
                                                            state.tokens_committed))
                  else:
                      lp_per_sb.append([])
                  committed_per_sb.append(state.tokens_committed)
                  return continue_llada()
              text = run_cmajc_k3(model, tokenizer, prompt, k_steps, n_branches,
                                   temperature, step_callback=cb)
              from e4.grade import is_correct
              r = float(is_correct(text, gold))
              out.append((committed_per_sb, lp_per_sb, text, r))
    """
    # Smoke / no-GPU path: synthesize a deterministic-ish dummy rollout.
    if model is None or os.environ.get("MOCK_MODELS") == "1":
        rng = random.Random(hash((prompt, gold)) & 0xFFFFFFFF)
        out: List[Tuple[List[List[int]], List[List[float]], str, float]] = []
        for i in range(m):
            committed = [[rng.randint(0, 32000) for _ in range(8)] for _ in range(num_sub_blocks)]
            log_probs = [
                ([math.log(rng.uniform(0.01, 0.99)) for _ in range(8)]
                 if 1 <= sb <= commit_n_blocks else [])
                for sb in range(num_sub_blocks)
            ]
            final_text = f"[mock rollout {i} for prompt[:32]={prompt[:32]!r}]"
            reward = float(rng.random() < 0.5)
            out.append((committed, log_probs, final_text, reward))
        return out

    # Real path: not implemented here (see TODO above + module docstring).
    raise NotImplementedError(
        "Real GPU rollout path requires e4/diff_llada._generate to expose "
        "per-sub-block committed_logits. See module-level TODO."
    )


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
    """Real GPU training entrypoint. Sketch-quality - ships when the
    diff_llada committed-logits hook is in place. Until then, --smoke is
    the only supported invocation.
    """
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 2

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- model + LoRA --
    print(f"Loading base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)

    if RESUME_FROM:
        print(f"Resuming LoRA adapter from {RESUME_FROM}", flush=True)
        model = PeftModel.from_pretrained(model, RESUME_FROM, is_trainable=True)
    else:
        cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        )
        model = get_peft_model(model, cfg)

    # -- phase emb --
    phase_emb = PhaseEmbedding(num_phases=NUM_SUB_BLOCKS, lora_rank=LORA_R)
    n_hooked = phase_emb.attach(model)
    expected_hooks = len(LORA_LAYERS_TO_TRANSFORM) * len(LORA_TARGETS)
    print(f"Phase-emb hooks installed: {n_hooked} (expected {expected_hooks})", flush=True)

    # -- optimizer (LoRA params + phase_emb) --
    trainable = [p for p in model.parameters() if p.requires_grad] + [phase_emb.param]
    optim = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    # -- substrate --
    import json as _json
    rows = []
    with open(SUBSTRATE_JSONL) as f:
        for line in f:
            rows.append(_json.loads(line))
    print(f"Substrate: {len(rows)} prompts from {SUBSTRATE_JSONL}", flush=True)

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
            rollouts = _rollout(model, tokenizer, row["question"], row["gold"], m=M_ROLLOUTS)

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

            # -- loss --
            loss, stats = grpo_loss(rollouts, commit_n_blocks=COMMIT_N_BLOCKS)
            # TODO: KL anchor - second forward pass with disable_adapter_layers,
            # KL on log pi vs log pi_frozen at committed positions, beta=KL_BETA.
            # Requires committed_logits in StepState (see module TODO).

            optim.zero_grad(set_to_none=True)
            if hasattr(loss, "backward"):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optim.step()
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                lf = float(loss) if hasattr(loss, "item") else float(loss)
                print(f"step:{global_step}/{total_steps} train_loss:{lf:.4f}", flush=True)

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
    model.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
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
