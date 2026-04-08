# app/graders.py
"""
Determnistic mathematical graders for all 4 tasks.
Features Novel Graph-Distance Label Matching, Brier-Score Confidence Calibration, 
and Dense NLP Keyword Reasoning evaluation.
"""
from __future__ import annotations
import re
from server.models import ModerationDecision, ModerationReward, ViolationCategory, ModerationAction


# ─── Novel Grading Math ───────────────────────────────────────────────────────

VIOLATION_DISTANCE_MATRIX = {
    "clean": {"clean": 0.0, "spam": 0.8, "scam": 0.9, "misinformation": 0.9, "hate_speech": 1.0, "harassment": 1.0, "violence": 1.0, "nudity": 1.0, "self_harm": 1.0, "terrorism": 1.0},
    "hate_speech": {"hate_speech": 0.0, "harassment": 0.3, "violence": 0.5, "terrorism": 0.6, "clean": 1.0},
    "harassment": {"harassment": 0.0, "hate_speech": 0.3, "violence": 0.6, "clean": 1.0},
    "violence": {"violence": 0.0, "terrorism": 0.4, "hate_speech": 0.5, "self_harm": 0.5, "clean": 1.0},
    "spam": {"spam": 0.0, "scam": 0.3, "misinformation": 0.6, "clean": 0.8},
    "scam": {"scam": 0.0, "spam": 0.3, "misinformation": 0.5, "clean": 0.9},
    "misinformation": {"misinformation": 0.0, "scam": 0.5, "spam": 0.6, "clean": 0.9},
    "nudity": {"nudity": 0.0, "clean": 1.0},
    "self_harm": {"self_harm": 0.0, "violence": 0.5, "clean": 1.0},
    "terrorism": {"terrorism": 0.0, "violence": 0.4, "hate_speech": 0.6, "clean": 1.0}
}

def get_distance(l1: str, l2: str) -> float:
    # Ensure symmetric lookup
    d1 = VIOLATION_DISTANCE_MATRIX.get(l1, {}).get(l2, 1.0)
    d2 = VIOLATION_DISTANCE_MATRIX.get(l2, {}).get(l1, 1.0)
    return min(d1, d2)


def _graph_f1_score(pred: set[str], truth: set[str]) -> tuple[float, float, float]:
    """Calculates continuous F1 score based on graph distance similarities mapped between predicted and true sets."""
    if not truth and not pred:
        return 1.0, 1.0, 1.0
    if not pred or not truth:
        return 0.0, 0.0, 0.0
    
    # Recall: For each truth label, find max semantic similarity in pred
    recall_sum = sum(max((1.0 - get_distance(t, p)) for p in pred) for t in truth)
    recall = recall_sum / len(truth)

    # Precision: For each pred label, find max semantic similarity in truth
    precision_sum = sum(max((1.0 - get_distance(t, p)) for t in truth) for p in pred)
    precision = precision_sum / len(pred)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _brier_calibration_penalty(f1: float, confidence: float) -> float:
    """Calculates Continuous Ranked Probability penalty. Penalizes overconfidence on wrong answers and underconfidence on right answers."""
    # Scale max penalty appropriately (max -0.2)
    return -0.2 * ((f1 - confidence) ** 2)


def _reasoning_density_score(reasoning: str, context: str) -> float:
    """Extracts high-value NLP signals from the context string and measures dense recall within the agent's reasoning bounds."""
    stop_words = {"the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "on", "at", "by", "this", "that", "it"}
    context_words = set(re.findall(r'\b\w+\b', context.lower())) - stop_words
    if not context_words: return 0.0
    reasoning_words = set(re.findall(r'\b\w+\b', reasoning.lower()))
    
    intersection = context_words & reasoning_words
    # Maxes out after capturing 50% density of core contextual tokens
    overlap = len(intersection) / max(len(context_words) * 0.5, 1.0)
    return min(overlap, 1.0)


# ─── Task 1: Single Label Grader ──────────────────────────────────────────────

def grade_single_label(
    decision: ModerationDecision,
    ground_truth_labels: list[str],
    ground_truth_action: str,
) -> ModerationReward:
    truth_set = set(ground_truth_labels)
    pred_set = {l.value for l in decision.labels}

    precision, recall, f1 = _graph_f1_score(pred_set, truth_set)
    action_correct = 1.0 if decision.action.value == ground_truth_action else 0.3
    
    brier_penalty = _brier_calibration_penalty(f1, decision.confidence)

    fp_penalty = 0.0
    if ViolationCategory.CLEAN.value in truth_set and ViolationCategory.CLEAN.value not in pred_set:
        fp_penalty = -0.5  # Heavy penalty for flagging clean content

    total = round(min(max(0.5 * f1 + 0.3 * action_correct + 0.2 * precision + fp_penalty + brier_penalty, -1.0), 1.0), 4)

    return ModerationReward(
        total=total,
        label_precision=precision,
        label_recall=recall,
        action_correct=action_correct,
        false_positive_penalty=fp_penalty + brier_penalty,
        breakdown={"f1": f1, "action": action_correct, "fp": fp_penalty, "brier": brier_penalty},
    )


# ─── Task 2: Multi-Label Grader ───────────────────────────────────────────────

def grade_multi_label(
    decision: ModerationDecision,
    ground_truth_labels: list[str],
    ground_truth_action: str,
) -> ModerationReward:
    truth_set = set(ground_truth_labels)
    pred_set = {l.value for l in decision.labels}

    precision, recall, f1 = _graph_f1_score(pred_set, truth_set)
    action_correct = 1.0 if decision.action.value == ground_truth_action else 0.2

    brier_penalty = _brier_calibration_penalty(f1, decision.confidence)

    fp_penalty = 0.0
    if ViolationCategory.CLEAN.value in truth_set and pred_set != {ViolationCategory.CLEAN.value}:
        fp_penalty = -0.4

    total = round(min(max(0.4 * f1 + 0.3 * precision + 0.2 * action_correct + fp_penalty + brier_penalty, -1.0), 1.0), 4)

    return ModerationReward(
        total=total,
        label_precision=precision,
        label_recall=recall,
        action_correct=action_correct,
        false_positive_penalty=fp_penalty + brier_penalty,
        breakdown={"f1": f1, "precision": precision, "action": action_correct, "brier": brier_penalty},
    )


# ─── Task 3: Ad Policy Grader ─────────────────────────────────────────────────

def grade_ad_policy(
    decision: ModerationDecision,
    ground_truth_labels: list[str],
    ground_truth_action: str,
    ground_truth_policy_ids: list[str],
) -> ModerationReward:
    truth_set = set(ground_truth_labels)
    pred_set = {l.value for l in decision.labels}

    precision, recall, f1 = _graph_f1_score(pred_set, truth_set)
    action_correct = 1.0 if decision.action.value == ground_truth_action else 0.1

    # Exact string matching for standard policy IDs
    cited = set(decision.policy_citations)
    truth_citations = set(ground_truth_policy_ids)
    
    cite_p, cite_r, cite_f1 = _graph_f1_score(cited, truth_citations) if truth_citations else (1.0, 1.0, 1.0)

    brier_penalty = _brier_calibration_penalty(f1, decision.confidence)

    fp_penalty = 0.0
    if ViolationCategory.CLEAN.value in truth_set and pred_set != {ViolationCategory.CLEAN.value}:
        fp_penalty = -0.3

    total = round(min(max(
        0.35 * f1 + 0.25 * action_correct + 0.3 * cite_f1 + 0.1 * precision + fp_penalty + brier_penalty,
        -1.0), 1.0), 4)

    return ModerationReward(
        total=total,
        label_precision=precision,
        label_recall=recall,
        action_correct=action_correct,
        policy_citation_score=cite_f1,
        false_positive_penalty=fp_penalty,
        breakdown={"f1": f1, "action": action_correct, "citation_f1": cite_f1, "brier": brier_penalty},
    )


# ─── Task 4: Thread Hard Grader ───────────────────────────────────────────────

def grade_thread_hard(
    decision: ModerationDecision,
    ground_truth_labels: list[str],
    ground_truth_action: str,
    has_policy_conflict: bool,
    is_final_message: bool,
    cultural_context: str = ""
) -> ModerationReward:
    truth_set = set(ground_truth_labels)
    pred_set = {l.value for l in decision.labels}

    precision, recall, f1 = _graph_f1_score(pred_set, truth_set)
    action_correct = 1.0 if decision.action.value == ground_truth_action else 0.15

    brier_penalty = _brier_calibration_penalty(f1, decision.confidence)

    # Reasoning quality via Semantic Trace Check
    reasoning_score = 0.0
    if has_policy_conflict or cultural_context:
         reasoning_score = _reasoning_density_score(decision.reasoning, cultural_context)

    fp_penalty = 0.0
    if ViolationCategory.CLEAN.value in truth_set and pred_set != {ViolationCategory.CLEAN.value}:
        fp_penalty = -0.5  # Hard penalty for protected speech flagged incorrectly

    # Final message bonus: thread-level action validation
    thread_bonus = 0.1 if is_final_message and action_correct == 1.0 else 0.0

    total = round(min(max(
        0.3 * f1 + 0.25 * action_correct + 0.25 * reasoning_score + 0.1 * precision + fp_penalty + thread_bonus + brier_penalty,
        -1.0), 1.0), 4)

    return ModerationReward(
        total=total,
        label_precision=precision,
        label_recall=recall,
        action_correct=action_correct,
        reasoning_quality=reasoning_score,
        false_positive_penalty=fp_penalty + brier_penalty,
        breakdown={
            "f1": f1, "action": action_correct,
            "reasoning": reasoning_score, "thread_bonus": thread_bonus, "brier": brier_penalty
        },
    )

def get_ground_truth(content_id: str, all_data: list[dict]) -> dict:
    for item in all_data:
        if item.get("content_id") == content_id:
            return {
                "labels": item.get("ground_truth_labels", ["clean"]),
                "action": item.get("ground_truth_action", "approve"),
                "policy_ids": item.get("violated_policies", []),
                "cultural_context": item.get("cultural_context", "") 
            }
    return {"labels": ["clean"], "action": "approve", "policy_ids": [], "cultural_context": ""}


# ─── OpenEnv Grader Interface Wrappers ────────────────────────────────────────

def _extract_data(state: Any) -> list[dict]:
    """Helper to extract ground_truth_data from state dict or object."""
    if isinstance(state, dict):
        return state.get("ground_truth_data", [])
    return getattr(state, "ground_truth_data", [])


def single_label_entry(state: Any, action: Any) -> ModerationReward:
    data = _extract_data(state)
    gt = get_ground_truth(action.content_id, data)
    return grade_single_label(action, gt["labels"], gt["action"])


def multi_label_entry(state: Any, action: Any) -> ModerationReward:
    data = _extract_data(state)
    gt = get_ground_truth(action.content_id, data)
    return grade_multi_label(action, gt["labels"], gt["action"])


def ad_policy_entry(state: Any, action: Any) -> ModerationReward:
    data = _extract_data(state)
    gt = get_ground_truth(action.content_id, data)
    return grade_ad_policy(action, gt["labels"], gt["action"], gt["policy_ids"])


def thread_hard_entry(state: Any, action: Any) -> ModerationReward:
    data = _extract_data(state)
    gt = get_ground_truth(action.content_id, data)
    
    # Thread task needs extra flags usually passed from env.step
    # If these aren't in state, we use defaults
    has_conflict = False
    is_final = False
    
    if isinstance(state, dict):
        has_conflict = state.get("has_policy_conflict", False)
        is_final = state.get("is_final_message", False)
    else:
        has_conflict = getattr(state, "has_policy_conflict", False)
        is_final = getattr(state, "is_final_message", False)

    return grade_thread_hard(
        action, gt["labels"], gt["action"],
        has_policy_conflict=has_conflict,
        is_final_message=is_final,
        cultural_context=gt["cultural_context"]
    )
