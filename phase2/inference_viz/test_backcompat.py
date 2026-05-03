"""Backwards-compat regression for the LLaDA `_generate` callback refactor.

Workstream C (Phase 2) added an optional `step_callback` to
``e4.diff_llada._Real._generate``. This test asserts that calling
``denoise_block`` WITHOUT a callback produces bit-identical mock output
versus a stored fixture, so existing ``runner.py`` callers (c2, c2c, c3,
c4, cmaj, cmajc, ...) can never silently regress.

We use ``MOCK_MODELS=1``-style mock paths because the real LLaDA model
needs a GPU and weights — but the mock's ``denoise_block`` signature is
also extended (it accepts the same kwargs and the same ``step_callback``
parameter), so the bit-equality check covers the same compatibility
contract: a None callback = legacy behavior.

Run:
    pytest phase2/inference_viz/test_backcompat.py -q
or:
    python phase2/inference_viz/test_backcompat.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e4 import diff_llada  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mock_c2c_v3.json"


def _make_call_args() -> dict:
    """Pinned call signature: matches runner.py:c2c invocation under MOCK."""
    return dict(
        prompt="Mock problem 42: 2 + 42 = ?",
        k_steps=32,
        seed=0,
        temperature=0.0,
        apply_commit=True,
        commit_n_blocks=3,
    )


def _run_legacy() -> tuple[str, int]:
    """Call denoise_block with NO callback — must match the fixture."""
    model = diff_llada.load(name="mock-llada", mock=True)
    return model.denoise_block(**_make_call_args())


def _run_with_noop_callback() -> tuple[str, int]:
    """Call denoise_block WITH a no-op callback — must also match the fixture
    (the callback returns continue_llada() at every sub-block, which is the
    documented bit-equivalent of None)."""

    states = []

    def cb(state: diff_llada.StepState) -> diff_llada.StepDirective:
        states.append(state.sub_block)
        return diff_llada.StepDirective.continue_llada()

    model = diff_llada.load(name="mock-llada", mock=True)
    text, used = model.denoise_block(step_callback=cb, **_make_call_args())
    # Must have visited all 4 sub-blocks under the mock schedule.
    assert states == [0, 1, 2, 3], f"expected visits [0,1,2,3], got {states}"
    return text, used


def maybe_write_fixture() -> dict:
    """Write the fixture if missing; called once on first run, then locked."""
    text, used = _run_legacy()
    payload = {"text": text, "used": int(used)}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def test_legacy_path_matches_fixture() -> None:
    if not FIXTURE.exists():
        expected = maybe_write_fixture()
    else:
        expected = json.loads(FIXTURE.read_text())
    text, used = _run_legacy()
    assert text == expected["text"], (
        f"legacy mock output drifted!\n  expected: {expected['text']!r}\n"
        f"  got:      {text!r}"
    )
    assert int(used) == int(expected["used"]), (
        f"flops drift: expected {expected['used']}, got {used}"
    )


def test_callback_path_is_bit_equivalent() -> None:
    expected = json.loads(FIXTURE.read_text())
    text, used = _run_with_noop_callback()
    assert text == expected["text"], (
        f"callback path drifted vs fixture!\n  expected: {expected['text']!r}\n"
        f"  got:      {text!r}"
    )
    assert int(used) == int(expected["used"])


def test_directive_stop_terminates_early() -> None:
    """Passing a stop directive at sub-block 1 must NOT crash; output is
    a partial mock string but the function must return cleanly."""

    def cb(state: diff_llada.StepState) -> diff_llada.StepDirective:
        if state.sub_block >= 1:
            return diff_llada.StepDirective.stop()
        return diff_llada.StepDirective.continue_llada()

    model = diff_llada.load(name="mock-llada", mock=True)
    text, used = model.denoise_block(step_callback=cb, **_make_call_args())
    assert isinstance(text, str)
    assert used > 0


if __name__ == "__main__":
    if not FIXTURE.exists():
        print(f"[backcompat] writing first-time fixture -> {FIXTURE}")
        maybe_write_fixture()
    test_legacy_path_matches_fixture()
    print("[backcompat] legacy path matches fixture: PASS")
    test_callback_path_is_bit_equivalent()
    print("[backcompat] callback path bit-equivalent:  PASS")
    test_directive_stop_terminates_early()
    print("[backcompat] stop directive terminates:     PASS")
    print("[backcompat] all 3 tests passed.")
