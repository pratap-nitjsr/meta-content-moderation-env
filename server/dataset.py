# app/dataset.py
"""
Dataset loader for MetaContentModerationEnv.
All data is loaded from local JSON files under data/.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_posts(seed: int = 42) -> list[dict]:
    items = load_json(DATA_DIR / "posts.json")
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def get_image_descriptions(seed: int = 42) -> list[dict]:
    items = load_json(DATA_DIR / "image_descriptions.json")
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def get_ad_copies(seed: int = 42) -> list[dict]:
    items = load_json(DATA_DIR / "ad_copies.json")
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def get_whatsapp_threads(seed: int = 42) -> list[dict]:
    items = load_json(DATA_DIR / "whatsapp_threads.json")
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def get_community_standards() -> dict:
    return load_json(DATA_DIR / "policies" / "community_standards.json")


def get_ad_policies() -> dict:
    return load_json(DATA_DIR / "policies" / "ad_policies.json")


def get_policy_excerpt(content_type: str, policies: dict) -> str:
    """Return a short relevant policy excerpt for the given content type."""
    relevant = [
        p["description"]
        for p in policies.get("policies", [])
    ]
    return " | ".join(relevant[:3])
