"""Calibration and threshold optimization module using temperature scaling (Platt scaling)."""
import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml

from src.safety_eval.metrics import compute_ece, compute_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
CONFIG_PATH = Path("configs/thresholds.yaml")

IDENTITY_COLS = [
    "male", "female", "lgbtq", "christian", "muslim",
    "jewish", "black", "white", "psychiatric_or_mental_illness"
]

class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration (Guo et al. 2017)."""
    __module__ = "src.safety_eval.calibrate"

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(float(temperature), 1e-3)

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        def nll_loss(T: float) -> float:
            scaled_logits = logits / max(T, 1e-3)
            # Binary cross entropy with numerical clipping
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -50.0, 50.0)))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            bce = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
            return float(bce)

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.clip(logits / max(self.temperature, 1e-3), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-scaled))

def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    n_thresholds: int = 500,
) -> Tuple[float, float]:
    """Find the threshold minimizing expected cost = FN * fn_cost + FP * fp_cost."""
    best_thresh = 0.5
    min_cost = float("inf")
    candidate_thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    for thresh in candidate_thresholds:
        preds = (y_prob >= thresh).astype(int)
        fn = np.sum((y_true == 1) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        cost = fn * fn_cost + fp * fp_cost
        if cost < min_cost:
            min_cost = float(cost)
            best_thresh = float(thresh)

    return float(round(best_thresh, 4)), float(min_cost)

def compute_subgroup_fpr(
    y_true: np.ndarray,
    preds: np.ndarray,
    df: pd.DataFrame
) -> Dict[str, float]:
    """Computes False Positive Rate (FP / (FP + TN)) for each demographic subgroup."""
    fpr_dict = {}
    for col in IDENTITY_COLS:
        if col in df.columns:
            sub_mask = df[col].values.astype(bool)
            neg_mask = sub_mask & (y_true == 0)
            neg_count = int(np.sum(neg_mask))
            if neg_count > 0:
                fp_count = int(np.sum((preds == 1) & neg_mask))
                fpr_dict[col] = round(fp_count / neg_count, 4)
            else:
                fpr_dict[col] = None
    return fpr_dict

def calibrate_and_evaluate_baseline() -> Dict[str, Any]:
    """Fits temperature scaling and cost-optimal threshold on val; evaluates and logs on test."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load thresholds config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    fn_cost = float(cfg.get("fn_cost", 5.0))
    fp_cost = float(cfg.get("fp_cost", 1.0))
    model_path = Path(cfg.get("model_path", "models/baseline_model.joblib"))

    logger.info("Loading baseline model and datasets...")
    model = joblib.load(model_path)
    val_df = pd.read_parquet(DATA_DIR / "civil_comments_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "civil_comments_test.parquet")
    hate_df = pd.read_parquet(DATA_DIR / "hatexplain_test.parquet")

    # Fit on val only
    logger.info("Extracting val logits and fitting TemperatureScaler...")
    val_logits = model.decision_function(val_df["text"].fillna(""))
    val_labels = val_df["label"].values
    val_raw_probs = model.predict_proba(val_df["text"].fillna(""))[:, 1]
    val_ece_before = compute_ece(val_labels, val_raw_probs)

    calibrator = TemperatureScaler()
    calibrator.fit(val_logits, val_labels)
    val_cal_probs = calibrator.predict_proba(val_logits)
    val_ece_after = compute_ece(val_labels, val_cal_probs)

    # Optimal threshold on val only
    opt_thresh, min_cost = find_cost_optimal_threshold(
        val_labels, val_cal_probs, fn_cost=fn_cost, fp_cost=fp_cost, n_thresholds=500
    )
    logger.info(f"Optimal threshold on val: {opt_thresh} (temperature={calibrator.temperature:.4f})")

    # Save calibrator
    calibrator_path = MODELS_DIR / "calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)

    # Evaluate on test split
    logger.info("Evaluating on Civil Comments test split...")
    test_logits = model.decision_function(test_df["text"].fillna(""))
    test_labels = test_df["label"].values
    test_raw_probs = model.predict_proba(test_df["text"].fillna(""))[:, 1]
    test_cal_probs = calibrator.predict_proba(test_logits)

    test_ece_before = compute_ece(test_labels, test_raw_probs)
    test_ece_after = compute_ece(test_labels, test_cal_probs)

    # Precision, Recall, F1 at chosen threshold
    test_preds = (test_cal_probs >= opt_thresh).astype(int)
    tp = int(np.sum((test_labels == 1) & (test_preds == 1)))
    fp = int(np.sum((test_labels == 0) & (test_preds == 1)))
    fn = int(np.sum((test_labels == 1) & (test_preds == 0)))
    tn = int(np.sum((test_labels == 0) & (test_preds == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_opt = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    test_fpr_overall = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    subgroup_fprs = compute_subgroup_fpr(test_labels, test_preds, test_df)

    # Update configs/thresholds.yaml
    cfg["threshold"] = opt_thresh
    cfg["temperature"] = round(calibrator.temperature, 4)
    cfg["model_path"] = str(model_path)
    cfg["calibrator_path"] = str(calibrator_path)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Load existing metrics
    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    baseline_metrics = {
        "model": "TF-IDF + LogisticRegression",
        "civil_comments_val": all_metrics.get("civil_comments_val", {}),
        "civil_comments_test": {
            **all_metrics.get("civil_comments_test", {}),
            "optimal_threshold": opt_thresh,
            "precision_at_optimal_threshold": round(prec, 4),
            "recall_at_optimal_threshold": round(rec, 4),
            "f1_at_optimal_threshold": round(f1_opt, 4),
            "overall_fpr_at_optimal_threshold": round(test_fpr_overall, 4),
            "subgroup_fpr_at_optimal_threshold": subgroup_fprs,
            "ece_uncalibrated": round(test_ece_before, 4),
            "ece_calibrated": round(test_ece_after, 4),
        },
        "hatexplain_test_m1": all_metrics.get("hatexplain_test_m1", all_metrics.get("hatexplain_test_domain_shift", {})),
        "hatexplain_test_m2": all_metrics.get("hatexplain_test_m2", {}),
        "calibration": {
            "temperature": round(calibrator.temperature, 4),
            "optimal_threshold": opt_thresh,
            "val_cost": min_cost,
            "ece_val_before": round(val_ece_before, 4),
            "ece_val_after": round(val_ece_after, 4),
            "ece_test_before": round(test_ece_before, 4),
            "ece_test_after": round(test_ece_after, 4),
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
        }
    }

    all_metrics["baseline"] = baseline_metrics
    # Also update top-level test metrics with calibrated figures
    all_metrics["civil_comments_test"] = baseline_metrics["civil_comments_test"]
    all_metrics["calibration"] = baseline_metrics["calibration"]

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Baseline calibration and threshold evaluation complete.")
    logger.info(f"Test ECE: {test_ece_before:.4f} -> {test_ece_after:.4f}")
    logger.info(f"Test P/R/F1 at {opt_thresh}: {prec:.4f} / {rec:.4f} / {f1_opt:.4f}")
    return baseline_metrics

if __name__ == "__main__":
    calibrate_and_evaluate_baseline()
