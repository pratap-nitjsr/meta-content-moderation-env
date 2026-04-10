# server/env.py
from __future__ import annotations
import uuid
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ModerationDecision, ModerationObservation, ModerationReward, ModerationState, ContentItem
    )
except ImportError:
    from models import (
        ModerationDecision, ModerationObservation, ModerationReward, ModerationState, ContentItem
    )

from server.dataset import (
    get_posts, get_image_descriptions, get_ad_copies, get_whatsapp_threads,
    get_community_standards, get_ad_policies,
)
from server.graders import (
    grade_single_label, grade_multi_label, grade_ad_policy, grade_thread_hard, get_ground_truth,
)
from server.tasks.task_single_label import build_episode as build_single_label_episode, build_observation as build_single_label_obs, MAX_STEPS as SINGLE_MAX, TASK_NAME as SINGLE_TASK
from server.tasks.task_multi_label import build_episode as build_multi_label_episode, build_observation as build_multi_label_obs, MAX_STEPS as MULTI_MAX, TASK_NAME as MULTI_TASK
from server.tasks.task_ad_policy import build_episode as build_ad_episode, build_observation as build_ad_obs, MAX_STEPS as AD_MAX, TASK_NAME as AD_TASK
from server.tasks.task_thread_hard import build_episode as build_thread_episode, build_observation as build_thread_obs, MAX_STEPS as THREAD_MAX, TASK_NAME as THREAD_TASK

VALID_TASKS = {SINGLE_TASK, MULTI_TASK, AD_TASK, THREAD_TASK}

class MetaContentModerationEnv(Environment[ModerationDecision, ModerationObservation, ModerationState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, task: str = "single-label-classify", seed: int = 42) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {VALID_TASKS}")
        self.task = task
        self.seed = seed

        self._episode_id: str = ""
        self._step: int = 0
        self._max_steps: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._decisions_log: list[dict[str, Any]] = []

        self._items: list[ContentItem] = []
        self._ground_truth_all: list[dict] = []
        self._thread_steps: list[Any] = []

    def reset(self, task: str = None, seed: int = None) -> ModerationObservation:
        if task is not None:
            if task not in VALID_TASKS:
                raise ValueError(f"Unknown task '{task}'. Valid: {VALID_TASKS}")
            self.task = task
        if seed is not None:
            self.seed = seed
            
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._decisions_log = []

        self._load_episode_data()
        
        obs = self._make_observation()
        obs.reward = 0.0
        obs.done = False
        obs.metadata = {
            "episode_id": self._episode_id,
            "step": self._step,
            "cumulative_reward": 0.0
        }
        return obs

    def step(self, action: ModerationDecision) -> ModerationObservation:
        if not self._episode_id:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

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

        self._step += 1
        self._done = self._step >= self._max_steps

        if self._done:
            next_obs = self._make_terminal_observation()
        else:
            next_obs = self._make_observation()

        next_obs.reward = reward
        next_obs.done = self._done
        next_obs.metadata = {
            "episode_id": self._episode_id,
            "cumulative_reward": self._cumulative_reward,
            "step": self._step,
            "reward_breakdown": reward_obj.model_dump(),
        }
        return next_obs

    @property
    def state(self) -> ModerationState:
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
            ground_truth_data=self._ground_truth_all,
            has_policy_conflict=bool(self._thread_steps[self._step][2]) if self.task == THREAD_TASK and self._step < len(self._thread_steps) else False,
            is_final_message=(self._step == self._max_steps - 1) if self.task == THREAD_TASK else False,
        )

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _load_episode_data(self) -> None:
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
            threads = get_whatsapp_threads(self.seed)
            self._ground_truth_all = [
                msg for t in threads for msg in t["messages"]
            ]
            self._items = [step[0] for step in self._thread_steps]

    def _make_observation(self) -> ModerationObservation:
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
        try:
            from ..models import ContentItem, ContentType
        except ImportError:
            from models import ContentItem, ContentType
            
        dummy = ContentItem(
            content_id="__terminal__",
            content_type=ContentType.TEXT_POST,
            text="Episode complete.",
        )
        return ModerationObservation(
            step=self._step,
            content_item=dummy,
            task_name=self.task,
            instructions="Episode complete. No more items.",
        )

    def _grade(self, action: ModerationDecision) -> ModerationReward:
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
        if not self._decisions_log:
            return 0.01
        max_possible = self._max_steps * 1.0
        if max_possible <= 0:
            return 0.01
            
        avg_reward = self._cumulative_reward / max_possible
        # Map avg_reward from [-1.0, 1.0] to [0.0, 1.0]
        normalized = (avg_reward + 1.0) / 2.0
        
        # Clamp strictly between 0.01 and 0.99 for OpenEnv
        score = min(max(normalized, 0.01), 0.99)
        return round(score, 4)

