# app/tasks/task_multi_label.py
"""
Task: multi-label-classify
Difficulty: Medium
Content types: text_post, ad_copy, whatsapp_message
Steps per episode: 12
Description:
    Agent receives content that may have 1, 2, or 3 violations simultaneously.
    Must return ALL applicable labels (or [CLEAN]).
    Grader uses precision/recall — partial credit for partial label matches.
    Agent is penalized more for false positives than false negatives.
"""
from __future__ import annotations
from server.models import ContentItem, ModerationObservation
from server.dataset import get_posts, get_ad_copies, get_community_standards, get_policy_excerpt

TASK_NAME = "multi-label-classify"
MAX_STEPS = 12
INSTRUCTIONS = (
    "You are an advanced content moderator. Each item may violate MULTIPLE policies simultaneously. "
    "Assign ALL applicable violation categories. If the content is clean, return only [CLEAN]. "
    "Provide your confidence score and briefly cite which policy rule applies. "
    "Respond with a JSON ModerationDecision."
)


def build_episode(seed: int = 42) -> list[ContentItem]:
    """Episode focuses on multi-label and borderline items."""
    posts = [p for p in get_posts(seed) if len(p.get("ground_truth_labels", [])) > 1 or
             p.get("difficulty") in ("medium", "hard")]
    ads = get_ad_copies(seed)[:4]
    all_items = (posts + ads)[:MAX_STEPS]
    return [ContentItem(**{k: v for k, v in item.items()
                           if k not in ("ground_truth_labels", "ground_truth_action",
                                        "difficulty", "ad_targeting", "violated_policies")})
            for item in all_items]


def build_observation(step: int, item: ContentItem) -> ModerationObservation:
    policies = get_community_standards()
    return ModerationObservation(
        step=step,
        content_item=item,
        policy_excerpt=get_policy_excerpt(item.content_type.value, policies),
        task_name=TASK_NAME,
        instructions=INSTRUCTIONS,
    )
