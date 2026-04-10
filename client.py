from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ModerationDecision, ModerationObservation, ModerationState

class MetaContentModerationClient(EnvClient[ModerationDecision, ModerationObservation, State]):
    """
    Client for the Meta Content Moderation Environment.
    """

    def _step_payload(self, action: ModerationDecision) -> Dict[str, Any]:
        """Convert ModerationDecision to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ModerationObservation]:
        """
        Parse server response into StepResult containing ModerationObservation.
        """
        obs_data = payload.get("observation", {})
        
        observation = ModerationObservation(
            step=obs_data.get("step", 0),
            content_item=obs_data.get("content_item", {}),
            policy_excerpt=obs_data.get("policy_excerpt", ""),
            thread_history=obs_data.get("thread_history", []),
            conflicting_policies=obs_data.get("conflicting_policies", []),
            task_name=obs_data.get("task_name", ""),
            instructions=obs_data.get("instructions", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("metadata", {})
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
