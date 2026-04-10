# app/tasks/task_ad_policy.py
"""
Task: ad-policy-compliance
Difficulty: Medium-Hard
Content types: ad_copy only
Steps per episode: 10
Description:
    Agent reviews ad copy and must:
    1. Identify violation labels
    2. Cite the specific AD_POLICY_* rule IDs that are violated
    3. Choose correct enforcement action (remove vs restrict vs approve)
    Partial credit for correct label even without correct policy citation.
    Full credit requires both correct label AND correct policy_citations.
"""
from __future__ import annotations
from models import ContentItem, ModerationObservation
from server.dataset import get_ad_copies, get_ad_policies

TASK_NAME = "ad-policy-compliance"
MAX_STEPS = 10
INSTRUCTIONS = (
    "You are a Meta Ads policy reviewer. For each ad, you must: "
    "1. Identify ALL violation categories present. "
    "2. Cite the specific policy rule IDs (e.g. AD_POLICY_HEALTH_CLAIMS_001) that are violated. "
    "3. Choose the correct action: approve (compliant), remove (clear violation), restrict (borderline). "
    "The full policy document is provided in policy_excerpt. "
    "Respond with a JSON ModerationDecision including policy_citations."
)


def build_episode(seed: int = 42) -> list[ContentItem]:
    ads = get_ad_copies(seed)[:MAX_STEPS]
    return [ContentItem(**{k: v for k, v in ad.items()
                           if k not in ("ground_truth_labels", "ground_truth_action",
                                        "difficulty", "violated_policies", "ad_targeting")})
            for ad in ads]


def build_observation(step: int, item: ContentItem) -> ModerationObservation:
    policies = get_ad_policies()
    full_policy_text = "\n".join(
        f"[{p['id']}] {p['name']}: {p['description']}"
        for p in policies["policies"]
    )
    return ModerationObservation(
        step=step,
        content_item=item,
        policy_excerpt=full_policy_text,
        task_name=TASK_NAME,
        instructions=INSTRUCTIONS,
    )
