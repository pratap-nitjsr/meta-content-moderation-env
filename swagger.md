# Meta Content Moderation Environment - API Documentation

This API powers the `meta-content-moderation-env` benchmark for OpenEnv autonomous agents.

## Endpoints

### 1. `POST /reset`
Initializes or resets a specific moderation task episode.

**Request Body:** `application/json`
```json
{
  "task": "single-label-classify",
  "seed": 42
}
```

**Response:** `200 OK` (Returns the initial Observation schema)
```json
{
  "step": 0,
  "task_name": "single-label-classify",
  "instructions": "Determine the highest severity violation...",
  "content_item": {
    "content_id": "post_1",
    "content_type": "text_post",
    "text": "User generated text...",
    "media_urls": [],
    "media_types": []
  },
  "thread_history": [],
  "conflicting_policies": [],
  "policy_excerpt": "Standard policies..."
}
```

---

### 2. `POST /step`
Submits a moderation action to the environment.

**Request Body:** `application/json` (`ModerationDecision`)
```json
{
  "content_id": "post_1",
  "labels": ["hate_speech"],
  "action": "remove",
  "confidence": 0.95,
  "reasoning": "Direct attack on protected group.",
  "policy_citations": []
}
```

**Response:** `200 OK`
```json
{
  "observation": { ... next ContentItem ... },
  "reward": 0.85,
  "done": false,
  "info": {}
}
```

---

### 3. `GET /state`
Retrieves the current episode state, providing live telemetry of the metrics algorithm.

**Response:** `200 OK`
```json
{
  "score": 0.85,
  "metrics": {
    "total_rewards": 0.85,
    "steps": 1
  }
}
```
