"""Fine-tunes DistilBERT on Civil Comments with evaluation on test, val, and HateXplain M1/M2."""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import yaml

from src.safety_eval.train_baseline import evaluate_predictions
from src.safety_eval.calibrate import TemperatureScaler, find_cost_optimal_threshold, compute_subgroup_fpr
from src.safety_eval.metrics import compute_ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
CONFIG_PATH = Path("configs/thresholds.yaml")

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

def get_predictions_and_logits(model, tokenizer, texts, batch_size=64, device="cuda"):
    model.eval()
    all_logits = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            # Binary classification: logit difference (pos - neg)
            pos_logits = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_logits.extend(pos_logits)
            all_probs.extend(probs)
    return np.array(all_logits, dtype=np.float32), np.array(all_probs, dtype=np.float32)

def train_and_eval_distilbert(
    subsample_train: bool = False,
    max_train_samples: int = 75000,
    num_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
) -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load datasets
    train_df = pd.read_parquet(DATA_DIR / "civil_comments_train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "civil_comments_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "civil_comments_test.parquet")
    hate_df = pd.read_parquet(DATA_DIR / "hatexplain_test.parquet")

    if subsample_train and len(train_df) > max_train_samples:
        logger.info(f"Subsampling train set from {len(train_df)} to {max_train_samples} to cap runtime under 40 min...")
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    train_dataset = TextClassificationDataset(train_df["text"].fillna(""), train_df["label"].values, tokenizer)
    val_dataset = TextClassificationDataset(val_df["text"].fillna(""), val_df["label"].values, tokenizer)

    training_args = TrainingArguments(
        output_dir="./distilbert_checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("Starting DistilBERT fine-tuning...")
    trainer.train()

    # Save fine-tuned transformer
    output_dir = MODELS_DIR / "distilbert_safety"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Inference on val, test, HateXplain
    logger.info("Computing predictions on val set...")
    val_logits, val_probs = get_predictions_and_logits(model, tokenizer, val_df["text"].fillna("").tolist(), device=device)
    val_metrics = evaluate_predictions(val_df["label"].values, val_probs, val_df)

    logger.info("Computing predictions on test set...")
    test_logits, test_probs = get_predictions_and_logits(model, tokenizer, test_df["text"].fillna("").tolist(), device=device)
    test_metrics = evaluate_predictions(test_df["label"].values, test_probs, test_df)

    logger.info("Computing predictions on HateXplain...")
    hate_logits, hate_probs = get_predictions_and_logits(model, tokenizer, hate_df["text"].fillna("").tolist(), device=device)
    hx_m1 = evaluate_predictions(hate_df["label_m1"].values, hate_probs, hate_df)

    mask_m2 = hate_df["label_m2"] >= 0
    hate_m2_df = hate_df[mask_m2]
    hate_m2_probs = hate_probs[mask_m2]
    hx_m2 = evaluate_predictions(hate_m2_df["label_m2"].values.astype(int), hate_m2_probs, hate_m2_df)

    # Save predictions parquets
    pd.DataFrame({"logit": val_logits, "prob": val_probs}).to_parquet(DATA_DIR / "preds_distilbert_val.parquet", index=False)
    pd.DataFrame({"logit": test_logits, "prob": test_probs}).to_parquet(DATA_DIR / "preds_distilbert_test.parquet", index=False)
    pd.DataFrame({"logit": hate_logits, "prob": hate_probs}).to_parquet(DATA_DIR / "preds_distilbert_hatexplain.parquet", index=False)

    # Calibration on val
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_df["label"].values)
    val_cal_probs = scaler.predict_proba(val_logits)
    test_cal_probs = scaler.predict_proba(test_logits)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    fn_cost = float(cfg.get("fn_cost", 5.0))
    fp_cost = float(cfg.get("fp_cost", 1.0))

    opt_thresh, min_cost = find_cost_optimal_threshold(
        val_df["label"].values, val_cal_probs, fn_cost=fn_cost, fp_cost=fp_cost, n_thresholds=500
    )

    test_ece_before = compute_ece(test_df["label"].values, test_probs)
    test_ece_after = compute_ece(test_df["label"].values, test_cal_probs)

    test_preds = (test_cal_probs >= opt_thresh).astype(int)
    test_labels = test_df["label"].values
    tp = int(np.sum((test_labels == 1) & (test_preds == 1)))
    fp = int(np.sum((test_labels == 0) & (test_preds == 1)))
    fn = int(np.sum((test_labels == 1) & (test_preds == 0)))
    tn = int(np.sum((test_labels == 0) & (test_preds == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_opt = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    subgroup_fprs = compute_subgroup_fpr(test_labels, test_preds, test_df)

    distilbert_results = {
        "model": "DistilBERT Fine-Tuned",
        "civil_comments_val": val_metrics,
        "civil_comments_test": {
            **test_metrics,
            "optimal_threshold": opt_thresh,
            "precision_at_optimal_threshold": round(prec, 4),
            "recall_at_optimal_threshold": round(rec, 4),
            "f1_at_optimal_threshold": round(f1_opt, 4),
            "overall_fpr_at_optimal_threshold": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
            "subgroup_fpr_at_optimal_threshold": subgroup_fprs,
            "ece_uncalibrated": round(test_ece_before, 4),
            "ece_calibrated": round(test_ece_after, 4),
        },
        "hatexplain_test_m1": hx_m1,
        "hatexplain_test_m2": hx_m2,
        "calibration": {
            "temperature": round(scaler.temperature, 4),
            "optimal_threshold": opt_thresh,
            "val_cost": min_cost,
            "ece_test_before": round(test_ece_before, 4),
            "ece_test_after": round(test_ece_after, 4),
        }
    }

    # Save to metrics.json
    metrics_path = Path("metrics.json")
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    all_metrics["distilbert"] = distilbert_results
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("DistilBERT evaluation saved to metrics.json.")
    return distilbert_results

if __name__ == "__main__":
    train_and_eval_distilbert()
