# app/tasks/task_thread_hard.py
"""
Task: thread-moderation-hard
Difficulty: Hard
Content types: whatsapp_message (full threads)
Steps per episode: 15
Description:
    This is the hardest task. Per step, the agent sees ONE message in a thread
    PLUS the full prior thread history (context window grows).
    Agent must:
    1. Multi-label classify the current message
    2. Account for cultural/regional context (author_region, language)
    3. Resolve conflicting policies (e.g. political speech vs hate speech)
    4. Decide action for the THREAD as a whole at the final message
    Grader checks:
    - Per-message label accuracy (partial credit)
    - Thread-level action accuracy (full credit)
    - Reasoning quality on conflicting policy cases (scored by keyword check)
    - False positive rate on political speech protected content
"""
from __future__ import annotations
from server.models import ContentItem, ModerationObservation
from server.dataset import get_whatsapp_threads, get_community_standards

TASK_NAME = "thread-moderation-hard"
MAX_STEPS = 15
INSTRUCTIONS = (
    "You are a senior content moderator handling a WhatsApp message thread. "
    "You see messages one at a time, with full prior context. "
    "For EACH message: assign all applicable violation labels, choose an action. "
    "For the FINAL message in a thread: also decide the thread-level action. "
    "IMPORTANT: Political speech is protected under CS_POLITICAL_001 even if it sounds harsh. "
    "Hate speech that uses political framing is NOT protected — use your judgment. "
    "When policies conflict, cite both rule IDs in policy_citations and explain in reasoning. "
    "Respond with a JSON ModerationDecision."
)


def build_episode(seed: int = 42) -> list[tuple[ContentItem, list[ContentItem], list[str]]]:
    """
    Returns list of (current_message, thread_history_so_far, conflicting_policy_ids).
    One entry per step across all threads in the episode.
    """
    threads = get_whatsapp_threads(seed)[:3]  # 3 threads, ~5 messages each = 15 steps
    steps = []
    policies = get_community_standards()
    conflict_policy_ids = ["CS_HATE_001", "CS_POLITICAL_001"]

    for thread in threads:
        messages = thread["messages"]
        history = []
        for msg in messages:
            item = ContentItem(**{k: v for k, v in msg.items() if k != "position"})
            conflicts = conflict_policy_ids if thread.get("difficulty") == "hard" else []
            steps.append((item, list(history), conflicts))
            history.append(item)

    return steps[:MAX_STEPS]


def build_observation(
    step: int,
    item: ContentItem,
    history: list[ContentItem],
    conflicting_policies: list[str],
) -> ModerationObservation:
    policies = get_community_standards()
    policy_text = "\n".join(
        f"[{p['id']}] {p['name']}: {p['description']}"
        for p in policies["policies"]
    )
    return ModerationObservation(
        step=step,
        content_item=item,
        policy_excerpt=policy_text,
        thread_history=history,
        conflicting_policies=conflicting_policies,
        task_name=TASK_NAME,
        instructions=INSTRUCTIONS,
    )
