"""Qwen2.5-Instruct AR wrapper. Mock + real paths share one interface.

Real model loading is deferred to GPU pod. Local smoke tests use mock=True so
the runner can be exercised without HF downloads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from e4 import flops as flops_mod


@dataclass
class _Mock:
    name: str

    def generate_cot_and_answer(self, question: str, seed: int = 0) -> tuple[str, int]:
        random.seed(seed)
        text = f"Mock CoT for: {question}\nAnswer: {random.randint(0, 100)}"
        return text, flops_mod.qwen_forward_flops(name=self.name, n_tokens=128)

    def generate_plan(
        self, question: str, max_tokens: int = 32, seed: int = 0
    ) -> tuple[str, int]:
        random.seed(seed)
        return (
            f"plan: think step by step about {question[:30]}",
            flops_mod.qwen_forward_flops(name=self.name, n_tokens=max_tokens),
        )

    def finalize_answer(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        random.seed(seed + 1)
        return (
            f"Answer: {random.randint(0, 100)}",
            flops_mod.qwen_forward_flops(name=self.name, n_tokens=16),
        )

    def extend_cot(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        random.seed(seed + 2)
        return (
            f"\nMore reasoning {random.randint(0, 9)}",
            flops_mod.qwen_forward_flops(name=self.name, n_tokens=24),
        )


@dataclass
class _Real:
    """Real Qwen2.5-Instruct path. Implementation lives on the pod (day 1)."""

    name: str
    _model: object | None = None
    _tokenizer: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Lazy import: only when actually loading on pod.
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype="auto", device_map="auto"
        )

    def generate_cot_and_answer(self, question: str, seed: int = 0) -> tuple[str, int]:
        raise NotImplementedError("Filled in on pod (day 1).")

    def generate_plan(
        self, question: str, max_tokens: int = 32, seed: int = 0
    ) -> tuple[str, int]:
        raise NotImplementedError("Filled in on pod (day 1).")

    def finalize_answer(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        raise NotImplementedError("Filled in on pod (day 1).")

    def extend_cot(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        raise NotImplementedError("Filled in on pod (day 1).")


def load(name: str, mock: bool = False) -> _Mock | _Real:
    return _Mock(name=name) if mock else _Real(name=name)
