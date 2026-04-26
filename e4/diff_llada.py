"""LLaDA diffusion-LM wrapper. Mock + real paths share one interface.

Real loading deferred to GPU pod (day 1). Mock returns deterministic strings.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from e4 import flops as flops_mod


@dataclass
class _Mock:
    name: str
    block_len: int = 256

    def denoise_block(
        self, prompt: str, k_steps: int, seed: int = 0
    ) -> tuple[str, int]:
        h = hashlib.sha256(f"{prompt}|{k_steps}|{seed}".encode()).hexdigest()[:8]
        text = f"Mock diffusion CoT (k={k_steps}) hash={h}"
        used = flops_mod.llada_forward_flops(
            name=self.name, n_tokens=self.block_len, n_steps=k_steps
        )
        return text, used


@dataclass
class _Real:
    """Real LLaDA-Instruct path. Implementation lives on the pod (day 1)."""

    name: str
    block_len: int = 256
    _model: object | None = None
    _tokenizer: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

    def denoise_block(
        self, prompt: str, k_steps: int, seed: int = 0
    ) -> tuple[str, int]:
        # Reference sampler: see https://github.com/ML-GSAI/LLaDA chat_demo.py
        raise NotImplementedError("Filled in on pod (day 1).")


def load(name: str, mock: bool = False, block_len: int = 256) -> _Mock | _Real:
    return _Mock(name=name, block_len=block_len) if mock else _Real(
        name=name, block_len=block_len
    )
