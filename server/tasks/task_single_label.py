# app/tasks/task_single_label.py
"""
Task: single-label-classify
Difficulty: Easy
Content types: text_post, image_description
Steps per episode: 10
Description:
    Agent receives one content item per step.
    Must return exactly ONE ViolationCategory label.
    Action must be one of: approve (if CLEAN) or remove/restrict.
"""
from __future__ import annotations
from models import (
    ContentItem, ModerationObservation, ContentType
)
from server.dataset import get_posts, get_image_descriptions, get_policy_excerpt, get_community_standards

TASK_NAME = "single-label-classify"
MAX_STEPS = 10
INSTRUCTIONS = (
    "You are a content moderator. For each item, assign EXACTLY ONE violation category "
    "from the list, or mark it as CLEAN if no violation exists. "
    "Then choose the appropriate enforcement action: approve (clean), remove (clear violation), "
    "restrict (borderline), or escalate (unclear). "
    "Respond with a JSON ModerationDecision."
)


def build_episode(seed: int = 42) -> list[ContentItem]:
    """Return a shuffled mix of posts and image descriptions for one episode."""
    posts = get_posts(seed)[:7]
    images = get_image_descriptions(seed)[:3]
    items = posts + images
    return [ContentItem(**{k: v for k, v in item.items()
                           if k not in ("ground_truth_labels", "ground_truth_action", "difficulty")})
            for item in items]


def build_observation(step: int, item: ContentItem) -> ModerationObservation:
    policies = get_community_standards()
    return ModerationObservation(
        step=step,
        content_item=item,
        policy_excerpt=get_policy_excerpt(item.content_type.value, policies),
        task_name=TASK_NAME,
        instructions=INSTRUCTIONS,
    )
