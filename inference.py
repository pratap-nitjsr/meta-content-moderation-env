#!/usr/bin/env python3
"""
inference.py — Baseline inference script for MetaContentModerationEnv.

STDOUT FORMAT (strict — do not modify):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

Environment Variables (Additional):
    MODERATION_TASK      Task name (default: "single-label-classify")
    MODERATION_SEED      Episode seed (default: 42)
    MAX_STEPS_OVERRIDE   Override max steps per task (optional)
    INFERENCE_MODE       'cot' or 'multi-agent' (default: 'cot')
"""
from __future__ import annotations

import json
import os
import sys
import dotenv
import textwrap
import traceback
from typing import Optional

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

dotenv.load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is a compulsory environment variable.")

TASK_NAME = os.getenv("MODERATION_TASK", "single-label-classify")
BENCHMARK = os.getenv("BENCHMARK", "meta-content-moderation")
SEED = int(os.getenv("MODERATION_SEED", "42"))
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "cot").lower()
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 512

TASK_MAX_STEPS = {
    "single-label-classify": 10,
    "multi-label-classify": 12,
    "ad-policy-compliance": 10,
    "thread-moderation-hard": 15,
}
MAX_STEPS = int(os.getenv("MAX_STEPS_OVERRIDE", str(TASK_MAX_STEPS.get(TASK_NAME, 10))))
SUCCESS_SCORE_THRESHOLD = 0.4  # score >= 0.4 counts as success

# ─── Stdout Logging ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string: no newlines, truncate to 120 chars
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─── System Prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_COT = textwrap.dedent("""
    You are an expert AI content moderator working for a major social media platform.
    You will be shown content items one at a time and must make moderation decisions.

    For EACH content item, you MUST first conduct a thorough policy analysis inside a <scratchpad> block. 
    Then, output exactly one JSON object matching this schema inside a Markdown block:

    <scratchpad>
    Analyze the text, context, thread history, and any conflicting policies.
    Debate whether the user's intent violates policy boundaries.
    </scratchpad>
    ```json
    {
        "content_id": "<exact content_id from the observation>",
        "labels": ["<violation_category>"],
        "action": "<approve|remove|escalate|restrict|request_review>",
        "confidence": <float 0.0-1.0>,
        "reasoning": "<brief explanation summarizing the scratchpad>",
        "policy_citations": ["<POLICY_ID>"]
    }
    ```

    Valid violation categories: hate_speech, violence, nudity, spam, misinformation,
    harassment, self_harm, terrorism, scam, clean

    Rules:
    - Use ["clean"] if no violation exists, with action "approve"
    - "clean" cannot be combined with other labels
    - For multi-label tasks: include ALL applicable labels
    - For ad-policy tasks: always include policy_citations
    - For thread tasks: consider full conversation context and cultural nuance

    FEW-SHOT EXAMPLE:
    User: 
    CONTENT TO MODERATE:
      content_id: post_100
      content_type: text_post
      text: The moon landing was completely faked by NASA in a Hollywood basement. WAKE UP!
      author_region: US
      language: en
      author_history: []
    
    Assistant:
    <scratchpad>
    The user is claiming the moon landing is faked. This is a classic conspiracy theory. Checking the misinformation policy, while false, this does not pose direct real-world harm or medical danger. Therefore, by standard severity scales, it may technically be 'clean' or fall under generic misinformation tracking without immediate removal. I will mark it clean to avoid free-speech false-positive penalties.
    </scratchpad>
    ```json
    {
        "content_id": "post_100",
        "labels": ["clean"],
        "action": "approve",
        "confidence": 0.85,
        "reasoning": "Harmless conspiracy theory lacking real-world harm metrics.",
        "policy_citations": []
    }
    ```
""").strip()

prompt_analyst = textwrap.dedent("""
    You are the PROSECUTOR. You represent platform safety.
    Analyze the provided content and vigorously argue why it Violates the policies.
    Point out any underlying dog whistles, potential harms, scams, or hateful intent.
    Be extremely critical. Do not write JSON. Output your argument concisely.
""").strip()

prompt_defender = textwrap.dedent("""
    You are the DEFENSE COUNSEL. You represent free speech and cultural nuance.
    Analyze the provided content and vigorously argue why it is CLEAN and does NOT violate policies.
    Point out cultural context, sarcasm, harmless intents, or missing direct threats.
    Defend the user's right to post. Do not write JSON. Output your argument concisely.
""").strip()

SYSTEM_PROMPT_JUDGE = textwrap.dedent("""
    You are an expert AI content moderator (The Final Judge) working for a major social media platform.
    You will be shown content items along with a Debate from two previous agents.

    For EACH content item, you MUST first conduct a thorough policy analysis inside a <scratchpad> block, weighing both the Prosecutor's and Defender's arguments.
    Then, output exactly one JSON object matching this schema inside a Markdown block:

    <scratchpad>
    Analyze the text, thread history, conflicting policies, and the debate.
    Decide whether the Prosecutor or Defender is correct based strictly on severity.
    </scratchpad>
    ```json
    {
        "content_id": "<exact content_id from the observation>",
        "labels": ["<violation_category>"],
        "action": "<approve|remove|escalate|restrict|request_review>",
        "confidence": <float 0.0-1.0>,
        "reasoning": "<brief explanation summarizing the scratchpad>",
        "policy_citations": ["<POLICY_ID>"]
    }
    ```

    Valid violation categories: hate_speech, violence, nudity, spam, misinformation,
    harassment, self_harm, terrorism, scam, clean

    Rules:
    - Use ["clean"] if no violation exists, with action "approve"
    - "clean" cannot be combined with other labels
    - For multi-label tasks: include ALL applicable labels
    - For thread tasks: consider full conversation context and cultural nuance
""").strip()


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_user_prompt(observation: dict) -> str:
    """Convert observation dict into a clear prompt for the model."""
    item = observation.get("content_item", {})
    thread_history = observation.get("thread_history", [])
    conflicting = observation.get("conflicting_policies", [])
    policy = observation.get("policy_excerpt", "")
    instructions = observation.get("instructions", "")

    parts = [
        f"TASK: {observation.get('task_name', '')}",
        f"STEP: {observation.get('step', 0)}",
        "",
        f"INSTRUCTIONS: {instructions}",
        "",
        "CONTENT TO MODERATE:",
        f"  content_id: {item.get('content_id', '')}",
        f"  content_type: {item.get('content_type', '')}",
        f"  text: {item.get('text', '')}",
        f"  author_region: {item.get('author_region', 'US')}",
        f"  language: {item.get('language', 'en')}",
        f"  author_history: {item.get('author_history', [])}",
    ]

    if thread_history:
        parts += ["", "THREAD CONTEXT (prior messages):"]
        for i, msg in enumerate(thread_history[-5:], 1):  # Last 5 for context
            parts.append(f"  [{i}] [{msg.get('author_region','?')}] {msg.get('text','')}")

    if conflicting:
        parts += ["", f"⚠️  CONFLICTING POLICIES DETECTED: {', '.join(conflicting)}"]
        parts.append("   Resolve the conflict in your reasoning field.")

    if policy:
        # Truncate policy to avoid token overflow
        policy_excerpt = policy[:800] + ("..." if len(policy) > 800 else "")
        parts += ["", "RELEVANT POLICIES:", policy_excerpt]

    parts += ["", "Respond with a JSON ModerationDecision object only."]
    return "\n".join(parts)


# ─── Model Call ───────────────────────────────────────────────────────────────

def get_model_decision(client: OpenAI, observation: dict) -> tuple[str, dict]:
    """
    Call the model and return (raw_text, parsed_decision_dict).
    Falls back to a safe default on any error.
    """
    user_prompt = build_user_prompt(observation)
    content_id = observation.get("content_item", {}).get("content_id", "unknown")

    safe_default = {
        "content_id": content_id,
        "labels": ["clean"],
        "action": "approve",
        "confidence": 0.5,
        "reasoning": "Model fallback — defaulting to approve",
        "policy_citations": [],
    }

    try:
        if INFERENCE_MODE == "multi-agent":
            # 1. Analyst Phase
            res_analyst = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt_analyst}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            arg_analyst = (res_analyst.choices[0].message.content or "").strip()

            # 2. Defender Phase
            res_defender = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt_defender}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            arg_defender = (res_defender.choices[0].message.content or "").strip()

            # 3. Judge Phase
            judge_prompt = f"{user_prompt}\n\n[PROSECUTOR ARGUMENT]\n{arg_analyst}\n\n[DEFENDER ARGUMENT]\n{arg_defender}\n\nEvaluate the arguments and output your final Decision."
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        else:
            # Standard CoT + Few-Shot Phase
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

        raw = (completion.choices[0].message.content or "").strip()

        # Extract JSON using robust bracket bounds since CoT scratchpad pushes it down
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw[start_idx:end_idx+1]
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON block found in generation")

        # Ensure content_id matches
        if "content_id" not in parsed or not parsed["content_id"]:
            parsed["content_id"] = content_id

        return raw, parsed

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        return str(safe_default), safe_default
    except Exception as e:
        print(f"[DEBUG] Model call failed: {e}", flush=True)
        return str(safe_default), safe_default


# ─── HTTP Client for Env ──────────────────────────────────────────────────────

import httpx

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

def env_reset(task: str, seed: int) -> dict:
    resp = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(decision: dict) -> dict:
    resp = httpx.post(
        f"{ENV_BASE_URL}/step",
        json={"action": decision},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    resp = httpx.get(f"{ENV_BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str, seed: int) -> None:
    """Run inference for a specific task and log results."""
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Get max steps for this specific task
    max_steps = int(os.getenv("MAX_STEPS_OVERRIDE", str(TASK_MAX_STEPS.get(task_name, 10))))
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        obs = env_reset(task=task_name, seed=seed)

        for step in range(1, max_steps + 1):
            if obs.get("content_item", {}).get("content_id") == "__terminal__":
                break

            # Get model decision
            raw_action, decision_dict = get_model_decision(client, obs)
            error_msg = None

            # Step environment
            try:
                result = env_step(decision_dict)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                obs = result.get("observation", obs)
            except Exception as e:
                error_msg = str(e)[:100]
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(decision_dict, separators=(",", ":")),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # Final state
        try:
            final_state = env_state()
            score = float(final_state.get("score", 0.0))
        except Exception:
            # Compute score manually if state endpoint fails
            max_possible = steps_taken * 1.0
            score = min(max(sum(rewards) / max_possible, 0.0), 1.0) if max_possible > 0 else 0.0

        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error in task {task_name}: {e}\n{traceback.format_exc()}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # List of tasks to iterate through
    tasks_to_run = [
        "single-label-classify",
        "multi-label-classify",
        "ad-policy-compliance",
        "thread-moderation-hard"
    ]
    
    # If MODERATION_TASK is set and valid, we could prioritize it, 
    # but the requirement is to iterate through all.
    for task in tasks_to_run:
        run_task(client, task, SEED)


if __name__ == "__main__":
    main()
