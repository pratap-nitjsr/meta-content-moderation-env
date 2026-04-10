# server/app.py
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv is required. Install with 'uv sync'") from e

try:
    from ..models import ModerationDecision, ModerationObservation, ModerationState
    from .env import MetaContentModerationEnv, VALID_TASKS
except ImportError:
    from models import ModerationDecision, ModerationObservation, ModerationState
    from server.env import MetaContentModerationEnv, VALID_TASKS

# Create the internal openenv interface
app = create_app(
    MetaContentModerationEnv,
    ModerationDecision,
    ModerationObservation,
    env_name="meta-content-moderation-env",
    max_concurrent_envs=1
)

# Custom extra routes specific to this environment
@app.get("/", include_in_schema=False)
def root():
    """Redirect home page to Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/tasks")
def list_tasks() -> dict:
    """List all available task names."""
    return {
        "tasks": list(VALID_TASKS),
        "descriptions": {
            "single-label-classify": "Easy \u2014 classify single content item into one violation category",
            "multi-label-classify": "Medium \u2014 assign multiple violation labels to content",
            "ad-policy-compliance": "Medium-Hard \u2014 review ad copy against ad policies, cite rule IDs",
            "thread-moderation-hard": "Hard \u2014 moderate full WhatsApp thread with cultural context + policy conflicts",
        }
    }

@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_openenv_yaml() -> str:
    """Serve the openenv.yaml metadata file."""
    yaml_path = Path(__file__).parent.parent / "openenv.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return yaml_path.read_text(encoding="utf-8")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
