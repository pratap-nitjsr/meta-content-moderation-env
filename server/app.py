# app/main.py
"""
FastAPI server exposing MetaContentModerationEnv via OpenEnv HTTP spec.

Endpoints:
    POST /reset          → ModerationObservation
    POST /step           → StepResult
    GET  /state          → ModerationState
    GET  /health         → {"status": "ok"}
    GET  /tasks          → list of available task names
    GET  /openenv.yaml   → serve the openenv.yaml metadata file
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from server.env import MetaContentModerationEnv, VALID_TASKS
from server.models import (
    ModerationDecision,
    ModerationObservation,
    ModerationState,
    StepResult,
)

# Try to import OpenEnv core package at runtime and log availability.
# Prefer the new `openenv.core` import; fall back to legacy `openenv_core` if needed.
_oe_imported = False
for _mod in ("openenv.core", "openenv_core"):
    try:
        _openenv_core = __import__(_mod, fromlist=["*"])
        _oe_ver = getattr(_openenv_core, "__version__", None)
        if _oe_ver:
            print(f"[startup] {_mod} available, version={_oe_ver}")
        else:
            print(f"[startup] {_mod} imported (version unknown)")
        _oe_imported = True
        break
    except Exception as _e:
        # continue to next candidate
        _oe_last_exc = _e

if not _oe_imported:
    print(f"[startup] openenv package not importable: {_oe_last_exc}")

# ─── App Init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MetaContentModerationEnv",
    description="OpenEnv environment for AI content moderation — Meta-inspired",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance — task and seed set via env vars or /reset body
_env: MetaContentModerationEnv | None = None


# ─── Request/Response Models ──────────────────────────────────────────────────

from pydantic import BaseModel, Field

class ResetRequest(BaseModel):
    task: str = Field(default="single-label-classify", description="Task name to run")
    seed: int = Field(default=42, description="Random seed for reproducibility")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect home page to Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """Health check — required for HF Spaces ping."""
    return {"status": "ok", "env": "MetaContentModerationEnv", "version": "0.1.0"}


@app.get("/tasks")
def list_tasks() -> dict:
    """List all available task names."""
    return {
        "tasks": list(VALID_TASKS),
        "descriptions": {
            "single-label-classify": "Easy — classify single content item into one violation category",
            "multi-label-classify": "Medium — assign multiple violation labels to content",
            "ad-policy-compliance": "Medium-Hard — review ad copy against ad policies, cite rule IDs",
            "thread-moderation-hard": "Hard — moderate full WhatsApp thread with cultural context + policy conflicts",
        }
    }


@app.post("/reset", response_model=ModerationObservation)
def reset(request: ResetRequest = ResetRequest()) -> ModerationObservation:
    """
    Reset the environment to a new episode.
    Returns the first observation.
    """
    global _env
    if request.task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task}'. Valid tasks: {list(VALID_TASKS)}"
        )
    _env = MetaContentModerationEnv(task=request.task, seed=request.seed)
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(action: ModerationDecision) -> StepResult:
    """
    Submit a moderation decision and advance the episode.
    Returns next observation, reward, done flag, and reward breakdown.
    """
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    try:
        result = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=ModerationState)
def state() -> ModerationState:
    """Return current environment state snapshot."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset to initialize the environment")
    return _env.state()


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_openenv_yaml() -> str:
    """Serve the openenv.yaml metadata file."""
    yaml_path = Path(__file__).parent.parent / "openenv.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return yaml_path.read_text(encoding="utf-8")


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event() -> None:
    """Pre-warm the default task on startup."""
    global _env
    default_task = os.getenv("MODERATION_TASK", "single-label-classify")
    default_seed = int(os.getenv("MODERATION_SEED", "42"))
    if default_task in VALID_TASKS:
        _env = MetaContentModerationEnv(task=default_task, seed=default_seed)
        _env.reset()
        print(f"[startup] Pre-warmed task='{default_task}' seed={default_seed}")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()


