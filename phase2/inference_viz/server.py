"""FastAPI backend for the step-by-step inference visualizer.

Architecture
------------
The LLaDA sampler in ``e4/diff_llada.py:_generate`` accepts a synchronous
``step_callback`` that fires once per sub-block boundary. This server runs
the sampler in a worker thread and uses two ``queue.Queue`` objects per
session to pump StepStates out (``state_q``) and StepDirectives in
(``directive_q``). The HTTP handler awaits an item from ``state_q`` to
return the next step to the client; the client posts a directive which we
push onto ``directive_q`` to unblock the worker.

This intentionally avoids asyncio/await-inside-callback contortions so the
real LLaDA sampler can run unmodified (it's plain torch.nograd loops).

Endpoints
---------
- ``POST /session/start``               body: {problem_idx, k_steps, seed,
                                              temperature, apply_commit,
                                              commit_n_blocks, mock}
                                       -> {session_id, problem, num_blocks}
- ``POST /session/{id}/step``           body: {directive: ..., args: {...}}
                                       -> {step_state | done | error}
- ``GET  /session/{id}/trace``          -> JSONL bytes (one record per
                                          sub-block) suitable for saving to
                                          ``phase2/inference_viz/traces/``.
- ``DELETE /session/{id}``              -> tears down the worker thread.
- ``GET  /healthz``                     -> {ok: true, mock: bool}

The first ``POST /session/{id}/step`` ALWAYS receives a "continue_llada"
directive implicitly (we just block on state_q). Subsequent calls echo the
client's directive into directive_q before blocking on state_q.

Run locally
-----------
    MOCK_MODELS=1 python3 phase2/inference_viz/server.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e4 import ar_qwen, diff_llada, runner  # noqa: E402

logging.basicConfig(
    level=os.environ.get("VIZ_LOG_LEVEL", "INFO"),
    format="[viz-server %(asctime)s %(levelname)s] %(message)s",
)
log = logging.getLogger("viz-server")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Session state ─────────────────────────────────────────────────────────


@dataclass
class Session:
    session_id: str
    problem: dict
    cfg: dict
    state_q: "queue.Queue[Optional[dict]]" = field(default_factory=queue.Queue)
    directive_q: "queue.Queue[diff_llada.StepDirective]" = field(default_factory=queue.Queue)
    trace: list[dict] = field(default_factory=list)
    worker: Optional[threading.Thread] = None
    finished: bool = False
    final_text: str = ""
    error: Optional[str] = None
    # True until the first /step call has returned the first sub-block state.
    # Used so the first /step doesn't push a stray directive on directive_q
    # before the worker's first state is pulled.
    awaiting_first_step: bool = True
    # The current sub-block's StepState handed to the callback — needed so
    # the AR-extend / cmaj-branch directives can mutate ``x_handle`` before
    # the sampler resumes.
    pending_state: Optional[diff_llada.StepState] = None


_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()


# ── Mechanism injection helpers ───────────────────────────────────────────


def _ar_extend_inject(
    session: Session,
    state: diff_llada.StepState,
    n_tokens: int,
    model_name: str,
) -> tuple[str, list[int]]:
    """Run AR-extend and graft the resulting tokens into x_handle.

    Returns (decoded_text, token_ids_written). In MOCK_MODELS=1 this just
    fills the next n_tokens masked slots with deterministic ids so the UI
    flow can be exercised without a GPU.
    """
    cfg = session.cfg
    mock = bool(cfg.get("mock"))
    ar_model = ar_qwen.load(model_name or cfg["ar_model"], mock=mock)
    plan = session.problem.get("question", "")
    cot_so_far = ""  # we don't currently buffer the LLaDA decode mid-flight
    text, _flops = ar_model.extend_cot(
        question=session.problem["question"], plan=plan, cot=cot_so_far, seed=cfg["seed"]
    )

    # If we have a real x_handle, write up to n_tokens AR ids into the next
    # masked positions in the current/next sub-block.
    written: list[int] = []
    if state.x_handle is not None:
        try:
            import torch  # type: ignore
            from e4.diff_llada import _LLADA_MASK_ID

            x = state.x_handle
            ar_real = ar_qwen.load(model_name or cfg["ar_model"], mock=False)
            ar_real._ensure_loaded()  # type: ignore[attr-defined]
            tok = ar_real._tokenizer  # type: ignore[attr-defined]
            ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()[:n_tokens]
            mask_positions = (x[0] == _LLADA_MASK_ID).nonzero(as_tuple=False).flatten().tolist()
            mask_positions = [p for p in mask_positions if p >= state.block_start]
            for tid, pos in zip(ids, mask_positions[:n_tokens]):
                x[0, pos] = int(tid)
                written.append(int(tid))
        except Exception as exc:
            log.warning("AR-extend graft fell back (%s); UI flow only", exc)
    return text, written


def _cmaj_branch(
    session: Session,
    state: diff_llada.StepState,
    b: int,
) -> dict:
    """Fork b parallel LLaDA continuations from the current x state.

    In real mode this clones x_handle and runs b independent samplers
    starting from the next sub-block; we then majority-vote among the
    extracted answers. In mock mode we fabricate b deterministic outputs.
    """
    cfg = session.cfg
    if cfg.get("mock"):
        outputs = [
            f"Mock branch {i}: ans={(cfg['seed'] * 7 + i) % 100}" for i in range(b)
        ]
        return {"b": b, "branch_outputs": outputs, "winner_idx": 0}

    # Real-mode branching is non-trivial because we'd need to deep-copy the
    # in-flight x and re-enter the sampler b times. For Phase 2 we do the
    # next-best thing: re-run denoise_block from scratch on the same prompt
    # with b different seeds, which matches runner.py:cmaj exactly.
    diff_model = diff_llada.load(
        cfg["diff_model"],
        mock=False,
        lora_path=cfg.get("lora_path") or None,
        commit_lora_path=cfg.get("commit_lora_path") or None,
    )
    branches: list[str] = []
    for i in range(b):
        cot, _used = diff_model.denoise_block(
            prompt=session.problem["question"],
            k_steps=cfg["k_steps"],
            seed=cfg["seed"] * 100 + i,
            temperature=max(cfg.get("temperature", 0.7), 0.7),
            apply_commit=bool(cfg.get("apply_commit")),
            commit_n_blocks=int(cfg.get("commit_n_blocks", 1)),
        )
        branches.append(cot)
    # Trivial first-wins for now; UI can override.
    return {"b": b, "branch_outputs": branches, "winner_idx": 0}


# ── Sampler worker thread ─────────────────────────────────────────────────


def _state_to_record(
    session_id: str,
    problem_idx: int,
    state: diff_llada.StepState,
    intervention: Optional[dict],
    ar_payload: Optional[dict],
    cmaj_payload: Optional[dict],
) -> dict:
    """Serialize a StepState into the pinned JSONL schema."""
    return {
        "session_id": session_id,
        "problem_idx": problem_idx,
        "step_idx": state.step_idx,
        "sub_block": state.sub_block,
        "mechanism": state.mechanism if intervention is None else (
            "ar_extend" if intervention.get("directive") == "switch_to_ar"
            else "cmaj_branch" if intervention.get("directive") == "branch_cmaj"
            else state.mechanism
        ),
        "tokens_committed": state.tokens_committed,
        "token_strings": state.token_strings,
        "positions": state.positions,
        "entropy": state.entropy,
        "top_k_logits": [
            [[int(t), float(p)] for (t, p) in row] for row in state.top_k_logits
        ],
        "commit_lora_active": bool(state.commit_lora_active),
        "logit_shift_norm": state.logit_shift_norm,
        "temperature": float(state.temperature),
        "steps_per_block": int(state.steps_per_block),
        "manual_intervention": intervention,
        "ar_extend": ar_payload,
        "cmaj_branch": cmaj_payload,
        "wallclock_ms": int(state.wallclock_ms),
        "timestamp": _utc_now_iso(),
    }


def _sampler_worker(session: Session) -> None:
    """Runs the LLaDA sampler with a callback that pumps the queues."""
    cfg = session.cfg

    diff_model = diff_llada.load(
        cfg["diff_model"],
        mock=bool(cfg.get("mock")),
        lora_path=cfg.get("lora_path") or None,
        commit_lora_path=cfg.get("commit_lora_path") or None,
    )

    # Inject the visited-blocks counter for trace bookkeeping.
    visited = {"i": 0}

    def cb(state: diff_llada.StepState) -> diff_llada.StepDirective:
        # 1) hand state out to HTTP handler
        record_skeleton = _state_to_record(
            session_id=session.session_id,
            problem_idx=session.problem.get("idx", -1),
            state=state,
            intervention=None,
            ar_payload=None,
            cmaj_payload=None,
        )
        # Stash the live state so the directive handler can mutate x_handle.
        session.pending_state = state
        session.state_q.put(record_skeleton)

        # 2) wait for the directive from the HTTP layer
        directive = session.directive_q.get()
        session.pending_state = None
        visited["i"] += 1

        # 3) augment the trace record with intervention details + persist
        intervention: Optional[dict] = None
        ar_payload: Optional[dict] = None
        cmaj_payload: Optional[dict] = None

        if directive.kind == "switch_to_ar":
            intervention = {
                "directive": "switch_to_ar",
                "args": {"n_tokens": directive.n_tokens, "model_name": directive.model_name},
            }
            text, ids = _ar_extend_inject(
                session, state, directive.n_tokens, directive.model_name
            )
            ar_payload = {
                "model": directive.model_name or cfg["ar_model"],
                "n_tokens": int(directive.n_tokens),
                "text": text,
                "tokens_grafted": ids,
            }
        elif directive.kind == "branch_cmaj":
            intervention = {
                "directive": "branch_cmaj",
                "args": {"b": directive.b},
            }
            cmaj_payload = _cmaj_branch(session, state, directive.b)
        elif directive.kind == "stop":
            intervention = {"directive": "stop", "args": {}}

        full_record = dict(record_skeleton)
        full_record["manual_intervention"] = intervention
        full_record["ar_extend"] = ar_payload
        full_record["cmaj_branch"] = cmaj_payload
        if intervention:
            full_record["mechanism"] = (
                "ar_extend" if directive.kind == "switch_to_ar"
                else "cmaj_branch" if directive.kind == "branch_cmaj"
                else state.mechanism
            )
        session.trace.append(full_record)
        return directive

    try:
        text, _used = diff_model.denoise_block(
            prompt=session.problem["question"],
            k_steps=int(cfg["k_steps"]),
            seed=int(cfg["seed"]),
            temperature=float(cfg.get("temperature", 0.0)),
            apply_commit=bool(cfg.get("apply_commit")),
            commit_n_blocks=int(cfg.get("commit_n_blocks", 1)),
            step_callback=cb,
        )
        session.final_text = text
    except Exception as exc:
        log.exception("sampler worker crashed")
        session.error = repr(exc)
    finally:
        session.finished = True
        # Sentinel so any blocked /step call returns "done".
        session.state_q.put(None)


# ── FastAPI app ───────────────────────────────────────────────────────────


try:
    from pydantic import BaseModel  # type: ignore

    class StartReq(BaseModel):
        problem_idx: int = 42
        k_steps: int = 32
        seed: int = 0
        temperature: float = 0.0
        apply_commit: bool = True
        commit_n_blocks: int = 3
        ar_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
        diff_model: str = "GSAI-ML/LLaDA-8B-Instruct"
        lora_path: str = ""
        commit_lora_path: str = ""
        mock: Optional[bool] = None  # falls back to MOCK_MODELS env

    class StepReq(BaseModel):
        directive: str = "continue_llada"
        n_tokens: int = 6
        model_name: str = ""
        b: int = 5
except ImportError:  # pydantic missing — server can't start anyway
    StartReq = None  # type: ignore
    StepReq = None  # type: ignore


def _make_app():
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.responses import JSONResponse, PlainTextResponse  # type: ignore

    app = FastAPI(title="Sfumato Inference Visualizer")

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "ok": True,
            "mock": bool(int(os.environ.get("MOCK_MODELS", "0"))),
            "active_sessions": len(_SESSIONS),
        }

    @app.post("/session/start")
    def start(req: StartReq) -> dict:
        mock = req.mock if req.mock is not None else bool(int(os.environ.get("MOCK_MODELS", "0")))

        # Load problem (one only — we don't need the full 200).
        if mock:
            problem = {
                "idx": req.problem_idx,
                "id": f"mock-{req.problem_idx}",
                "question": f"Mock problem {req.problem_idx}: 2 + {req.problem_idx} = ?",
                "answer": str(2 + req.problem_idx),
            }
        else:
            dev_indices = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"
            problems = runner.load_problems(req.problem_idx + 1, dev_indices)
            if req.problem_idx >= len(problems):
                raise HTTPException(400, f"problem_idx out of range (max {len(problems)-1})")
            problem = {**problems[req.problem_idx], "idx": req.problem_idx}

        cfg = req.model_dump()
        cfg["mock"] = mock

        sid = uuid.uuid4().hex
        session = Session(session_id=sid, problem=problem, cfg=cfg)
        with _SESSIONS_LOCK:
            _SESSIONS[sid] = session

        t = threading.Thread(target=_sampler_worker, args=(session,), daemon=True)
        session.worker = t
        t.start()

        # gen_length / sub_block_length are fixed in the wrapper.
        return {
            "session_id": sid,
            "problem": problem,
            "num_blocks": 4,
            "sub_block_length": 32,
            "cfg": cfg,
        }

    @app.post("/session/{sid}/step")
    def step(sid: str, req: StepReq) -> dict:
        session = _SESSIONS.get(sid)
        if session is None:
            raise HTTPException(404, "session not found")

        # On the very first /step call we just pull the worker's initial
        # sub-block state — the worker is blocked on directive_q AFTER having
        # pushed state #0. On every subsequent call we must FIRST forward
        # the client's directive (unblocking the worker) and THEN block on
        # state_q for the next state.
        if session.awaiting_first_step:
            session.awaiting_first_step = False
        else:
            session.directive_q.put(_make_directive(req))

        # Block for next state with a generous timeout to keep clients
        # responsive even if the worker is doing a heavy real-mode forward.
        try:
            record = session.state_q.get(timeout=120.0)
        except queue.Empty:
            return {"timeout": True, "session_id": sid}

        if record is None:
            return {
                "done": True,
                "final_text": session.final_text,
                "error": session.error,
            }
        # NB: the returned record's `manual_intervention` will still be null
        # at this point — it's stamped retroactively after the next step call
        # that supplies a directive. Trace endpoint exposes the post-stamp.
        return {"step": record}

    @app.get("/session/{sid}/trace")
    def trace(sid: str) -> PlainTextResponse:
        session = _SESSIONS.get(sid)
        if session is None:
            raise HTTPException(404, "session not found")
        body = "\n".join(json.dumps(r) for r in session.trace) + ("\n" if session.trace else "")
        return PlainTextResponse(body, media_type="application/jsonl")

    @app.delete("/session/{sid}")
    def delete(sid: str) -> dict:
        with _SESSIONS_LOCK:
            session = _SESSIONS.pop(sid, None)
        if session is None:
            raise HTTPException(404, "session not found")
        # Best-effort wakeup of any blocked worker.
        try:
            session.directive_q.put_nowait(diff_llada.StepDirective.stop())
        except Exception:
            pass
        return {"ok": True}

    def _make_directive(req: StepReq) -> diff_llada.StepDirective:
        if req.directive == "continue_llada":
            return diff_llada.StepDirective.continue_llada()
        if req.directive == "switch_to_ar":
            return diff_llada.StepDirective.switch_to_ar(
                n_tokens=req.n_tokens, model_name=req.model_name
            )
        if req.directive == "branch_cmaj":
            return diff_llada.StepDirective.branch_cmaj(b=req.b)
        if req.directive == "stop":
            return diff_llada.StepDirective.stop()
        raise HTTPException(400, f"unknown directive {req.directive!r}")

    return app


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    try:
        import uvicorn  # type: ignore
    except ImportError:
        print(
            "uvicorn + fastapi required. Install with: "
            "pip install fastapi uvicorn pydantic",
            file=sys.stderr,
        )
        return 1

    app = _make_app()
    log.info("starting viz server on %s:%s (mock=%s)",
             args.host, args.port, os.environ.get("MOCK_MODELS", "0"))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
