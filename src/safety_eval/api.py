"""FastAPI application for safety evaluation, monitoring, and human-in-the-loop review."""
import os
import time
import yaml
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.safety_eval.monitor import SafetyMonitor
from src.safety_eval.review import ReviewQueue
from src.safety_eval.calibrate import TemperatureScaler

app = FastAPI(title="LLM Safety Evaluation Service", version="0.1.0")

# Load configuration
CONFIG_PATH = Path("configs/thresholds.yaml")
def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {
        "threshold": 0.5,
        "flagged_tolerance_margin": 0.1,
        "rolling_window_size": 100,
        "flagged_rate_alert_threshold": 0.20,
        "model_version": "baseline-tfidf-logreg-v1",
        "temperature": 1.0,
    }

config = load_config()
monitor = SafetyMonitor(
    window_size=config.get("rolling_window_size", 100),
    alert_threshold=config.get("flagged_rate_alert_threshold", 0.20),
)
review_queue = ReviewQueue(db_path="review_queue.db")

# Model state
MODEL_PATH = Path("models/baseline_model.joblib")
CALIBRATOR_PATH = Path("models/calibrator.joblib")

_model = None
_calibrator = None

def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model

def get_calibrator():
    global _calibrator
    if _calibrator is None and CALIBRATOR_PATH.exists():
        _calibrator = joblib.load(CALIBRATOR_PATH)
    elif _calibrator is None:
        _calibrator = TemperatureScaler(temperature=config.get("temperature", 1.0))
    return _calibrator

class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to evaluate for toxicity/safety")

class EvaluateResponse(BaseModel):
    score: float
    calibrated_score: float
    flagged: bool
    threshold: float
    model_version: str
    latency_ms: float
    review_enqueued: bool

class ReviewRequest(BaseModel):
    human_label: int = Field(..., ge=0, le=1, description="0=safe, 1=toxic/violating")

class ExportRequest(BaseModel):
    output_path: str = Field(default="data/reviewed_labels.csv")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": config.get("model_version", "unknown")}

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_text(req: EvaluateRequest):
    start_t = time.perf_counter()
    model = get_model()
    calibrator = get_calibrator()
    threshold = float(config.get("threshold", 0.5))
    margin = float(config.get("flagged_tolerance_margin", 0.1))
    model_version = str(config.get("model_version", "baseline-tfidf-logreg-v1"))

    # Compute raw score
    if model is not None:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([req.text])
            raw_score = float(probs[0][1])
            # logit for temperature scaling
            raw_score_clipped = max(min(raw_score, 1.0 - 1e-7), 1e-7)
            logit = float(np_logit(raw_score_clipped))
            calibrated_score = float(calibrator.predict_proba(np.array([logit]))[0])
        else:
            raw_score = 0.5
            calibrated_score = 0.5
    else:
        # Fallback heuristic if model not yet trained
        raw_score = 0.0
        calibrated_score = 0.0

    latency_ms = (time.perf_counter() - start_t) * 1000.0
    flagged = bool(calibrated_score >= threshold)

    # Monitor
    monitor.record(flagged)

    # Human-in-the-loop review enqueue if borderline
    enqueue_id = review_queue.maybe_enqueue(
        text=req.text,
        score=calibrated_score,
        threshold=threshold,
        margin=margin,
        model_version=model_version,
    )

    return EvaluateResponse(
        score=round(raw_score, 4),
        calibrated_score=round(calibrated_score, 4),
        flagged=flagged,
        threshold=round(threshold, 4),
        model_version=model_version,
        latency_ms=round(latency_ms, 2),
        review_enqueued=enqueue_id is not None,
    )

import numpy as np
def np_logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))

@app.get("/monitor")
def get_monitor_metrics():
    return monitor.get_status()

@app.get("/review/pending")
def get_pending_reviews(limit: int = 50):
    return review_queue.get_pending(limit=limit)

@app.post("/review/export")
def export_reviews(req: ExportRequest):
    count = review_queue.export_to_csv(req.output_path)
    return {"message": f"Exported {count} records", "path": req.output_path, "count": count}

@app.post("/review/{item_id}")
def submit_review(item_id: int, req: ReviewRequest):
    success = review_queue.store_review(item_id, req.human_label)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Review item {item_id} not found")
    return {"message": "Review recorded successfully", "id": item_id, "human_label": req.human_label}

