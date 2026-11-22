"""Baseline model training and evaluation: TF-IDF + LogisticRegression."""
import os
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.safety_eval.metrics import (
    compute_auroc,
    compute_pr_auc,
    compute_f1,
    compute_ece,
    compute_subgroup_auc,
    compute_bpsn_auc,
    compute_bnsp_auc,
    compute_borkan_bias_metric,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
IDENTITY_COLS = [
    "male", "female", "lgbtq", "christian", "muslim",
    "jewish", "black", "white", "psychiatric_or_mental_illness"
]

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    overall_auc = compute_auroc(y_true, y_prob)
    overall_pr_auc = compute_pr_auc(y_true, y_prob)
    overall_f1 = compute_f1(y_true, y_prob, threshold=threshold)
    overall_ece = compute_ece(y_true, y_prob)

    subgroup_aucs = {}
    bpsn_aucs = {}
    bnsp_aucs = {}

    for col in IDENTITY_COLS:
        if col in df.columns:
            mask = df[col].values.astype(bool)
            if np.sum(mask) >= 10:  # Require minimum samples for reliable AUC
                sub_auc = compute_subgroup_auc(y_true, y_prob, mask)
                bpsn = compute_bpsn_auc(y_true, y_prob, mask)
                bnsp = compute_bnsp_auc(y_true, y_prob, mask)
                subgroup_aucs[col] = sub_auc
                bpsn_aucs[col] = bpsn
                bnsp_aucs[col] = bnsp

    bias_metric = compute_borkan_bias_metric(
        overall_auc=overall_auc,
        subgroup_aucs=subgroup_aucs,
        bpsn_aucs=bpsn_aucs,
        bnsp_aucs=bnsp_aucs,
    )

    return {
        "auroc": round(overall_auc, 4),
        "pr_auc": round(overall_pr_auc, 4),
        "f1": round(overall_f1, 4),
        "ece": round(overall_ece, 4),
        "borkan_bias_metric": round(bias_metric, 4) if not np.isnan(bias_metric) else None,
        "subgroup_auc": {k: round(v, 4) if not np.isnan(v) else None for k, v in subgroup_aucs.items()},
        "bpsn_auc": {k: round(v, 4) if not np.isnan(v) else None for k, v in bpsn_aucs.items()},
        "bnsp_auc": {k: round(v, 4) if not np.isnan(v) else None for k, v in bnsp_aucs.items()},
    }

def train_and_eval():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading prepared parquet datasets...")
    train_df = pd.read_parquet(DATA_DIR / "civil_comments_train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "civil_comments_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "civil_comments_test.parquet")
    hate_df = pd.read_parquet(DATA_DIR / "hatexplain_test.parquet")

    logger.info(f"Training on {len(train_df)} samples, evaluating on {len(val_df)} val, {len(test_df)} test, {len(hate_df)} HateXplain...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1)),
    ])

    logger.info("Fitting TF-IDF + LogisticRegression pipeline...")
    pipeline.fit(train_df["text"].fillna(""), train_df["label"].values)

    logger.info("Saving trained baseline model...")
    joblib.dump(pipeline, MODELS_DIR / "baseline_model.joblib")

    logger.info("Computing predictions on Civil Comments test...")
    test_probs = pipeline.predict_proba(test_df["text"].fillna(""))[:, 1]
    civil_metrics = evaluate_predictions(test_df["label"].values, test_probs, test_df)

    logger.info("Computing predictions on HateXplain external test (domain shift)...")
    hate_probs = pipeline.predict_proba(hate_df["text"].fillna(""))[:, 1]
    hate_m1_label = hate_df["label_m1"].values if "label_m1" in hate_df.columns else hate_df["label"].values
    hate_m1_metrics = evaluate_predictions(hate_m1_label, hate_probs, hate_df)

    # Mapping 2: hateful -> 1, normal -> 0, offensive excluded
    if "label_m2" in hate_df.columns:
        mask_m2 = hate_df["label_m2"] >= 0
        hate_m2_df = hate_df[mask_m2]
        hate_m2_probs = hate_probs[mask_m2]
        hate_m2_metrics = evaluate_predictions(hate_m2_df["label_m2"].values.astype(int), hate_m2_probs, hate_m2_df)
    else:
        hate_m2_metrics = None

    logger.info("Computing predictions on Civil Comments val (for calibration baseline)...")
    val_probs = pipeline.predict_proba(val_df["text"].fillna(""))[:, 1]
    val_metrics = evaluate_predictions(val_df["label"].values, val_probs, val_df)

    results = {
        "model": "TF-IDF + LogisticRegression",
        "civil_comments_val": val_metrics,
        "civil_comments_test": civil_metrics,
        "hatexplain_test_domain_shift": hate_m1_metrics,
        "hatexplain_test_m1": hate_m1_metrics,
        "hatexplain_test_m2": hate_m2_metrics,
    }

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Civil Comments Test AUROC: {civil_metrics['auroc']}, PR-AUC: {civil_metrics['pr_auc']}, Borkan Bias Metric: {civil_metrics['borkan_bias_metric']}")
    logger.info(f"HateXplain M1 AUROC: {hate_m1_metrics['auroc']}, PR-AUC: {hate_m1_metrics['pr_auc']}")
    if hate_m2_metrics:
        logger.info(f"HateXplain M2 AUROC: {hate_m2_metrics['auroc']}, PR-AUC: {hate_m2_metrics['pr_auc']}")
    logger.info("Metrics successfully saved to metrics.json")

    return results

if __name__ == "__main__":
    train_and_eval()
