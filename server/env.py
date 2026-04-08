# app/env.py
"""
MetaContentModerationEnv — OpenEnv compliant environment.

Implements:
    reset() -> ModerationObservation
    step(action: ModerationDecision) -> StepResult
    state() -> ModerationState

Supports 4 tasks:
    single-label-classify   (easy)
    multi-label-classify    (medium)
    ad-policy-compliance    (medium-hard)
    thread-moderation-hard  (hard)
"""
from __future__ import annotations

import uuid
from typing import Any

from server.models import (
    ModerationDecision,
    ModerationObservation,
    ModerationReward,
    ModerationState,
    StepResult,
    ContentItem,
)
from server.dataset import (
    get_posts,
    get_image_descriptions,
    get_ad_copies,
    get_whatsapp_threads,
    get_community_standards,
    get_ad_policies,
)
from server.graders import (
    grade_single_label,
    grade_multi_label,
    grade_ad_policy,
    grade_thread_hard,
    get_ground_truth,
)
from server.tasks.task_single_label import (
    build_episode as build_single_label_episode,
    build_observation as build_single_label_obs,
    MAX_STEPS as SINGLE_MAX,
    TASK_NAME as SINGLE_TASK,
)
from server.tasks.task_multi_label import (
    build_episode as build_multi_label_episode,
    build_observation as build_multi_label_obs,
    MAX_STEPS as MULTI_MAX,
    TASK_NAME as MULTI_TASK,
)
from server.tasks.task_ad_policy import (
    build_episode as build_ad_episode,
    build_observation as build_ad_obs,
    MAX_STEPS as AD_MAX,
    TASK_NAME as AD_TASK,
)
from server.tasks.task_thread_hard import (
    build_episode as build_thread_episode,
    build_observation as build_thread_obs,
    MAX_STEPS as THREAD_MAX,
    TASK_NAME as THREAD_TASK,
)

VALID_TASKS = {SINGLE_TASK, MULTI_TASK, AD_TASK, THREAD_TASK}


class MetaContentModerationEnv:
    """
    OpenEnv environment for Meta-inspired content moderation.

    Usage:
        env = MetaContentModerationEnv(task="single-label-classify", seed=42)
        obs = env.reset()
        while True:
            decision = agent.act(obs)
            result = env.step(decision)
            if result.done:
                break
    """

    def __init__(self, task: str = "single-label-classify", seed: int = 42) -> None:
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {VALID_TASKS}")
        self.task = task
        self.seed = seed

        # Episode state — all reset in reset()
        self._episode_id: str = ""
        self._step: int = 0
        self._max_steps: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._decisions_log: list[dict[str, Any]] = []

        # Episode data (loaded per task)
        self._items: list[ContentItem] = []
        self._ground_truth_all: list[dict] = []
        self._thread_steps: list[Any] = []  # Used only for thread task

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> ModerationObservation:
        """Reset environment to initial state. Returns first observation."""
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._decisions_log = []

        # Load episode data
        self._load_episode_data()

        return self._make_observation()

    def step(self, action: ModerationDecision) -> StepResult:
        """
        Process one moderation decision.

        Args:
            action: Agent's ModerationDecision for the current content item.

        Returns:
            StepResult with observation, reward, done, info.

        Raises:
            RuntimeError: If called before reset() or after episode ends.
        """
        if not self._episode_id:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Grade the action
        reward_obj = self._grade(action)
        reward = reward_obj.total

        self._cumulative_reward += reward
        self._decisions_log.append({
            "step": self._step,
            "content_id": action.content_id,
            "labels": [l.value for l in action.labels],
            "action": action.action.value,
            "reward": reward,
            "breakdown": reward_obj.breakdown,
        })

        # Advance step
        self._step += 1
        self._done = self._step >= self._max_steps

        # Build next observation (or terminal)
        if self._done:
            next_obs = self._make_terminal_observation()
        else:
            next_obs = self._make_observation()

        return StepResult(
            observation=next_obs,
            reward=reward,
            reward_breakdown=reward_obj,
            done=self._done,
            info={
                "episode_id": self._episode_id,
                "cumulative_reward": self._cumulative_reward,
                "step": self._step,
            },
        )

    def state(self) -> ModerationState:
        """Return current internal state snapshot."""
        score = self._compute_score()
        return ModerationState(
            task_name=self.task,
            episode_id=self._episode_id,
            current_step=self._step,
            max_steps=self._max_steps,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            items_seen=self._step,
            items_remaining=max(0, self._max_steps - self._step),
            decisions_log=self._decisions_log,
            score=score,
        )

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _load_episode_data(self) -> None:
        """Load and prepare episode data based on active task."""
        if self.task == SINGLE_TASK:
            self._items = build_single_label_episode(self.seed)
            self._max_steps = min(SINGLE_MAX, len(self._items))
            raw_all = get_posts(self.seed) + get_image_descriptions(self.seed)
            self._ground_truth_all = raw_all

        elif self.task == MULTI_TASK:
            self._items = build_multi_label_episode(self.seed)
            self._max_steps = min(MULTI_MAX, len(self._items))
            self._ground_truth_all = get_posts(self.seed) + get_ad_copies(self.seed)

        elif self.task == AD_TASK:
            self._items = build_ad_episode(self.seed)
            self._max_steps = min(AD_MAX, len(self._items))
            self._ground_truth_all = get_ad_copies(self.seed)

        elif self.task == THREAD_TASK:
            self._thread_steps = build_thread_episode(self.seed)
            self._max_steps = min(THREAD_MAX, len(self._thread_steps))
            # Flatten for ground truth lookup
            threads = get_whatsapp_threads(self.seed)
            self._ground_truth_all = [
                msg for t in threads for msg in t["messages"]
            ]
            # Build items list from thread steps for current item lookup
            self._items = [step[0] for step in self._thread_steps]

    def _make_observation(self) -> ModerationObservation:
        """Build observation for current step."""
        if self.task == THREAD_TASK:
            item, history, conflicts = self._thread_steps[self._step]
            return build_thread_obs(self._step, item, history, conflicts)

        item = self._items[self._step]

        if self.task == SINGLE_TASK:
            return build_single_label_obs(self._step, item)
        elif self.task == MULTI_TASK:
            return build_multi_label_obs(self._step, item)
        elif self.task == AD_TASK:
            return build_ad_obs(self._step, item)

        raise ValueError(f"Unknown task: {self.task}")

    def _make_terminal_observation(self) -> ModerationObservation:
        """Return a terminal (end-of-episode) observation."""
        from server.models import ContentItem, ContentType
        dummy = ContentItem(
            content_id="__terminal__",
            content_type=ContentType.TEXT_POST,
            text="Episode complete.",
        )
        from server.tasks.task_single_label import INSTRUCTIONS
        return ModerationObservation(
            step=self._step,
            content_item=dummy,
            task_name=self.task,
            instructions="Episode complete. No more items.",
        )

    def _grade(self, action: ModerationDecision) -> ModerationReward:
        """Route grading to correct grader based on active task."""
        gt = get_ground_truth(action.content_id, self._ground_truth_all)

        if self.task == SINGLE_TASK:
            return grade_single_label(action, gt["labels"], gt["action"])

        elif self.task == MULTI_TASK:
            return grade_multi_label(action, gt["labels"], gt["action"])

        elif self.task == AD_TASK:
            return grade_ad_policy(action, gt["labels"], gt["action"], gt["policy_ids"])

        elif self.task == THREAD_TASK:
            _, _, conflicts = self._thread_steps[self._step]
            is_final = (self._step == self._max_steps - 1)
            return grade_thread_hard(
                action, gt["labels"], gt["action"],
                has_policy_conflict=bool(conflicts),
                is_final_message=is_final,
            )

        raise ValueError(f"Unknown task: {self.task}")

    def _compute_score(self) -> float:
        """Normalize cumulative reward to [0.0, 1.0]."""
        if not self._decisions_log:
            return 0.0
        # Max possible reward: 1.0 per step × max_steps
        max_possible = self._max_steps * 1.0
        raw = self._cumulative_reward / max_possible if max_possible > 0 else 0.0
        return round(min(max(raw, 0.0), 1.0), 4)
