import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "healthy")

def test_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "tasks" in response.json()

def test_reset_and_state():
    # Test reset
    response = client.post("/reset", json={"task": "single-label-classify", "seed": 42})
    assert response.status_code == 200
    res_data = response.json()
    obs = res_data.get("observation", res_data)
    assert "step" in obs
    assert "content_item" in obs

    # Test state
    response = client.get("/state")
    assert response.status_code == 200
    state = response.json()
    assert state["task_name"] == "single-label-classify"
    assert state["current_step"] == 0

def test_step():
    client.post("/reset", json={"task": "single-label-classify", "seed": 42})
    
    # Needs a real content_id from the dataset, but the env allows any currently.
    response = client.post("/step", json={"action": {
        "content_id": "dummy_content",
        "labels": ["clean"],
        "action": "approve",
        "confidence": 1.0,
        "reasoning": "Looks good",
        "policy_citations": []
    }})
    assert response.status_code == 200
    result = response.json()
    assert "reward" in result
    assert "observation" in result
