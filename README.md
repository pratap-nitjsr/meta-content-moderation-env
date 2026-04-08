---
title: Meta Content Moderation OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: true
tags:
  - openenv
  - content-moderation
  - reinforcement-learning
  - meta
  - ai-safety
license: mit
---

# 🛡️ MetaContentModerationEnv

**OpenEnv environment for training and evaluating AI agents on real-world content moderation.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Pratap-K/meta-content-moderation-env)

Inspired by the operational challenges of content moderation at Meta scale — billions of posts, dozens of languages, evolving policies, and cultural nuance that breaks English-only models.

---

## Why This Environment Exists

Content moderation is one of the most consequential AI tasks in production today. Every major social platform employs thousands of human moderators and increasingly uses AI to assist. Yet there is no public, structured benchmark environment where agents can be trained, evaluated, and compared on this task.

This environment fills that gap. An agent trained or evaluated here could be directly adapted for:
- Assisting human moderators with triage
- Pre-screening content before human review
- Evaluating LLM safety properties on real-world content

---

## Environment Overview

| Property | Value |
|----------|-------|
| Name | `MetaContentModerationEnv` |
| Version | 0.1.0 |
| Framework | FastAPI + Pydantic v2 |
| Package Manager | uv |
| Python | 3.11+ |
| Deployment | [HF Spaces](https://huggingface.co/spaces/Pratap-K/meta-content-moderation-env) + Local Docker |

---

## 📊 The Dataset (Hybrid Real & Synthetic)

The dataset integrates extremely robust test cases designed to break typical LLM guards. It features a deliberate 50/50 blend of **real-world extractions** and **synthetic adversarial noise**:

1. **Cornell `hate_speech_offensive` Lexicon:** We embedded 30 raw, unredacted text strings exported directly from Cornell University's peer-reviewed Twitter dataset on Hugging Face (`tdavidson/hate_speech_offensive`) to test the agent on authentic, colloquial slurs in the wild.
2. **True Multimodality (VLM Ready):** Image Description and Ad Copy tasks now include natively populated `media_urls` pointing directly to Unsplash/Wikimedia endpoints containing high-definition imagery matching the exact scenario. Vision-Language Models (like GPT-4o or Pixtral) can download and ingest these images directly from the state!
3. **Adversarial Linguistic Obfuscation:** The synthetic elements of our dataset utilize 'leetspeak' (e.g. *krill urself*), deep culturally-entrenched slang, and contextual emojis designed to defeat classic regex filters.

| Type | Description |
|------|-------------|
| `text_post` | Social media posts and comments (English/Hindi) |
| `image_description` | Alt-text + Direct Image URL attachments |
| `ad_copy` | Advertising copy targeting specific age/metrics |
| `whatsapp_message` | Multi-message conversational threads with deep context |

---

## Action Space

The agent submits a `ModerationDecision` JSON object at each step:

```json
{
  "content_id": "post_042",
  "labels": ["hate_speech", "harassment"],
  "action": "remove",
  "confidence": 0.87,
  "reasoning": "Direct ethnic slur combined with threatening language. Violates CS_HATE_001.",
  "policy_citations": ["CS_HATE_001", "CS_VIOLENCE_001"]
}
```

**Violation categories:** `hate_speech`, `violence`, `nudity`, `spam`, `misinformation`, `harassment`, `self_harm`, `terrorism`, `scam`, `clean`

**Actions:** `approve`, `remove`, `escalate`, `restrict`, `request_review`

---

## Observation Space

At each step the agent receives:

```json
{
  "step": 3,
  "content_item": {
    "content_id": "post_042",
    "content_type": "text_post",
    "text": "...",
    "author_region": "DE",
    "language": "en",
    "author_history": ["hate_speech"]
  },
  "policy_excerpt": "[CS_HATE_001] Hate Speech — Direct Attack: ...",
  "thread_history": [],
  "conflicting_policies": [],
  "task_name": "single-label-classify",
  "instructions": "..."
}
```

---

## Tasks

### Task 1: `single-label-classify` — Easy
- **Objective:** Classify one item into exactly one violation category (or CLEAN)
- **Content types:** Text posts, image descriptions
- **Steps:** 10 per episode
- **Grader:** Exact label match + action correctness
- **Expected score (GPT-4o):** ~0.75

### Task 2: `multi-label-classify` — Medium
- **Objective:** Assign ALL applicable violation labels (content may violate multiple policies)
- **Content types:** Text posts, ad copy, WhatsApp messages
- **Steps:** 12 per episode
- **Grader:** F1 score on label set + action correctness + false positive penalty
- **Expected score (GPT-4o):** ~0.62

### Task 3: `ad-policy-compliance` — Medium-Hard
- **Objective:** Review ad copy against specific policy rules and cite the violated rule IDs
- **Content types:** Ad copy only
- **Steps:** 10 per episode
- **Grader:** F1 on labels + policy citation F1 + action correctness
- **Expected score (GPT-4o):** ~0.58

### Task 4: `thread-moderation-hard` — Hard
- **Objective:** Moderate a full conversation thread message-by-message with growing context
- **Special challenges:** Cultural nuance, multi-label violations, conflicting policy resolution
- **Content types:** WhatsApp-style messages
- **Steps:** 15 per episode
- **Grader:** Per-message label F1 + reasoning quality on conflict cases + thread-level action + false positive penalty on protected political speech
- **Expected score (GPT-4o):** ~0.45

---

## 📐 The Graders & Reward Design

Standard evaluation grids use simplistic "LLM-as-a-judge" or exact-string matching. We rejected this brittle framing. 

Our core grading logic leverages a **Deterministic Mathematical Framework**:

1. **Semantic Hierarchy Graph Distance:** Rather than strict `1.0` or `0.0` correct tags, our grader utilizes a distance matrix. If a model identifies 'harassment' instead of 'hate_speech', the matrix grants partial Jaccard-overlap topological points for being "near" the correct severity branch.
2. **Brier-Score Calibration Penalties:** We implemented Continuous Ranked Probability Scoring on the model's self-reported `confidence` scalar. If a model confidently executes a false positive, the penalty scales quadratically.
3. **Dense Token Reasoning Intersection:** To gauge the `reasoning` trace without invoking an expensive un-reproducible LLM judge, we compute localized Set Intersections on dense policy rule keywords extracted directly from the system prompt rules.

| Component | Mathematical Operation | Weight |
|-----------|------------------------|--------|
| Label F1 | Distance Matrix Recall | 30–50% |
| Action accuracy | Binary scalar multiplication | 20–30% |
| Policy citations | Hypergeometric Set Intersection | 0–30% (ad task only) |
| Reasoning density| Stop-word optimized Jaccard overlap | 0–25% (hard task only) |
| False positive | Exponential Brier-Penalty | -30 to -50% |

All rewards clamped to `[-1.0, 1.0]`. Episode score normalized to `[0.0, 1.0]`.

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- uv: `pip install uv`
- Docker (for containerized deployment)

### Local Development

```bash
# Clone
git clone https://github.com/<your-username>/meta-content-moderation-env
cd meta-content-moderation-env

# Generate lockfile and sync dependencies
uv lock
uv sync

# Copy and configure env vars
cp .env.example .env
# Edit .env with your API keys

# Start server
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal — run inference
export MODEL_PROVIDER=hf
export HF_TOKEN=hf_...
uv run python inference.py
```

### Docker

```bash
docker build -t meta-content-moderation-env .
docker run -p 7860:7860 \
  -e MODEL_PROVIDER=hf \
  -e HF_TOKEN=hf_... \
  meta-content-moderation-env
```

### API Usage

```bash
# Health check
curl http://localhost:7860/health

# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "single-label-classify", "seed": 42}'

# Submit a decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"content_id": "post_001", "labels": ["clean"], "action": "approve", "confidence": 0.9, "reasoning": "", "policy_citations": []}'

# Check state
curl http://localhost:7860/state
```

---

## Running All Tasks

```bash
for task in single-label-classify multi-label-classify ad-policy-compliance thread-moderation-hard; do
  echo "=== Running task: $task ==="
  MODERATION_TASK=$task uv run python inference.py
done
```

## API Documentation (Swagger)

Because this environment is built natively on **FastAPI**, an interactive Swagger API UI is automatically generated and hosted alongside the server.

You can explore every endpoint, view request/response schemas, and interact with the environment directly from your browser:
*   **Swagger UI:** [http://localhost:7860/docs](http://localhost:7860/docs)
*   **ReDoc UI:** [http://localhost:7860/redoc](http://localhost:7860/redoc)
*   **Raw OpenAPI Spec:** [http://localhost:7860/openapi.json](http://localhost:7860/openapi.json)

A statically generated copy `openapi.json` has also been placed in the repository root for offline reference.

---

## Baseline Scores

| Task | Difficulty | Zero-Shot | CoT + Few-Shot | Multi-Agent Debate |
|------|-----------|-----------|----------------|--------------------|
| `single-label-classify` | Easy | 0.963 | **0.963** | 0.432 |
| `multi-label-classify` | Medium | 0.317 | **0.762** | 0.055 |
| `ad-policy-compliance` | Medium-Hard | **0.514** | 0.450 | 0.195 |
| `thread-moderation-hard` | Hard | 0.354 | 0.498 | **0.613** |

*\*Note: Zero-Shot Multi-Agent Debate on specific deep-thread evaluations actually excels at resolving complex logic (reaching `0.613`). But running zero-shot Prosecutors on easy/medium tasks induces extreme 'over-moderation', destroying the score with false-positive penalties. This demonstrates the strict necessity of building RLHF!*

Scores measured with `MODERATION_SEED=42`, `TEMPERATURE=0.0`.

---

## Tests

```bash
uv run pytest tests/ -v
```

---

## API Documentation (Swagger)

Because this environment is built natively on **FastAPI**, an interactive Swagger API UI is automatically generated and hosted alongside the server.

You can explore every endpoint, view request/response schemas, and interact with the environment directly from your browser:
*   **Swagger UI:** [http://localhost:7860/docs](http://localhost:7860/docs)
*   **ReDoc UI:** [http://localhost:7860/redoc](http://localhost:7860/redoc)
*   **Raw OpenAPI Spec:** [http://localhost:7860/openapi.json](http://localhost:7860/openapi.json)

A statically generated copy `openapi.json` has also been placed in the repository root for offline reference, along with a readable Markdown API Reference at [swagger.md](swagger.md).

---

## Project Structure

```
meta-content-moderation-env/
├── inference.py          # Baseline inference script (run this)
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile
├── pyproject.toml        # uv Build config
├── uv.lock               # uv Lockfile
├── server/
│   ├── app.py            # FastAPI server
│   ├── env.py            # Core environment
│   ├── models.py         # Pydantic models
│   ├── graders.py        # Task graders
│   ├── dataset.py        # Data loader
│   └── tasks/            # Per-task episode builders
├── data/
│   ├── posts.json
│   ├── image_descriptions.json
│   ├── ad_copies.json
│   ├── whatsapp_threads.json
│   └── policies/
└── tests/
```

---

## License

MIT — see [LICENSE](LICENSE) file.
