import pytest
from fastapi.testclient import TestClient
from src.safety_eval.api import app, review_queue, monitor

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_evaluate_endpoint():
    response = client.post("/evaluate", json={"text": "I love sunshine and flowers."})
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "calibrated_score" in data
    assert "flagged" in data
    assert "latency_ms" in data
    assert isinstance(data["flagged"], bool)

def test_monitor_endpoint():
    client.post("/evaluate", json={"text": "Test comment 1"})
    response = client.get("/monitor")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "flagged_rate" in data
    assert "alert" in data

def test_review_endpoints():
    # Enqueue an item directly for testing
    item_id = review_queue.maybe_enqueue(
        text="borderline sample text",
        score=0.52,
        threshold=0.5,
        margin=0.1,
        model_version="test-model",
    )
    assert item_id is not None

    # Get pending reviews
    pending = client.get("/review/pending")
    assert pending.status_code == 200
    items = pending.json()
    assert any(i["id"] == item_id for i in items)

    # Submit review
    rev_resp = client.post(f"/review/{item_id}", json={"human_label": 1})
    assert rev_resp.status_code == 200

    # Export to CSV
    exp_resp = client.post("/review/export", json={"output_path": "data/test_reviews.csv"})
    assert exp_resp.status_code == 200
    assert exp_resp.json()["count"] >= 1

def test_calibrated_path_and_threshold_loading():
    response = client.post("/evaluate", json={"text": "This is completely abhorrent violence and hatred."})
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data
    assert data["threshold"] == pytest.approx(0.1161, abs=0.01)
    assert data["model_version"] == "baseline-tfidf-logreg-v1"
    assert "calibrated_score" in data
    assert isinstance(data["flagged"], bool)

