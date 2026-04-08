# app/models.py
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class ViolationCategory(str, Enum):
    """All possible content violation categories."""
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    NUDITY = "nudity"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    TERRORISM = "terrorism"
    SCAM = "scam"
    CLEAN = "clean"  # No violation


class ContentType(str, Enum):
    """Type of content being moderated."""
    TEXT_POST = "text_post"
    IMAGE_DESCRIPTION = "image_description"
    AD_COPY = "ad_copy"
    WHATSAPP_MESSAGE = "whatsapp_message"


class ModerationAction(str, Enum):
    """Actions the agent can take on content."""
    APPROVE = "approve"
    REMOVE = "remove"
    ESCALATE = "escalate"
    RESTRICT = "restrict"         # Limit reach, don't remove
    REQUEST_REVIEW = "request_review"


class ContentItem(BaseModel):
    """A single piece of content to be moderated."""
    content_id: str = Field(..., description="Unique content identifier")
    content_type: ContentType
    text: str = Field(..., description="Raw content text or image description")
    author_region: str = Field(default="US", description="ISO 3166-1 alpha-2 country code")
    author_history: list[str] = Field(
        default_factory=list,
        description="Prior violation categories for this author"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context: thread_id, reply_to, ad_targeting etc."
    )
    language: str = Field(default="en", description="ISO 639-1 language code")
    media_urls: list[str] = Field(
        default_factory=list, 
        description="List of direct URLs to associated media (e.g. JPG, PNG)"
    )
    media_types: list[str] = Field(
        default_factory=list, 
        description="Corresponding media types (e.g. 'image', 'video')"
    )


class ModerationObservation(BaseModel):
    """
    What the agent sees at each step.
    Contains the content item to moderate plus any policy references.
    """
    step: int = Field(..., ge=0, description="Current step index")
    content_item: ContentItem
    policy_excerpt: str = Field(
        default="",
        description="Relevant policy text snippet for this content type"
    )
    thread_history: list[ContentItem] = Field(
        default_factory=list,
        description="Prior messages in thread (for thread tasks only)"
    )
    conflicting_policies: list[str] = Field(
        default_factory=list,
        description="Names of policies that may conflict for this item (hard task)"
    )
    task_name: str = Field(..., description="Active task identifier")
    instructions: str = Field(..., description="Natural language task instructions for the agent")


class ModerationDecision(BaseModel):
    """
    The agent's moderation decision for a single content item.
    This is what the agent submits via step().
    """
    content_id: str = Field(..., description="Must match the observed content_item.content_id")
    labels: list[ViolationCategory] = Field(
        ...,
        min_length=1,
        description="One or more violation categories. Use [CLEAN] if no violation."
    )
    action: ModerationAction = Field(..., description="Enforcement action to take")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agent confidence in this decision (0.0–1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Optional free-text reasoning (used in hard task grader)"
    )
    policy_citations: list[str] = Field(
        default_factory=list,
        description="Policy rule IDs cited for this decision"
    )

    @model_validator(mode="after")
    def clean_label_exclusivity(self) -> ModerationDecision:
        """CLEAN label must be the only label if present."""
        if ViolationCategory.CLEAN in self.labels and len(self.labels) > 1:
            raise ValueError("CLEAN label cannot be combined with violation labels")
        return self


class ModerationReward(BaseModel):
    """
    Structured reward breakdown for one step.
    Provides partial credit signals at every step.
    """
    total: float = Field(..., ge=-1.0, le=1.0, description="Composite reward for this step")

    # Partial credit components
    label_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    label_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    action_correct: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_citation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    false_positive_penalty: float = Field(default=0.0, le=0.0, description="Negative if clean content flagged")
    reasoning_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Hard task only")

    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Raw component scores for debugging"
    )


class ModerationState(BaseModel):
    """
    Full internal state of the environment.
    Returned by state() endpoint.
    """
    task_name: str
    episode_id: str
    current_step: int
    max_steps: int
    done: bool
    cumulative_reward: float
    items_seen: int
    items_remaining: int
    decisions_log: list[dict[str, Any]] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class StepResult(BaseModel):
    """Return type of env.step() — mirrors OpenEnv spec."""
    observation: ModerationObservation
    reward: float
    reward_breakdown: ModerationReward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
