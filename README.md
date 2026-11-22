# LLM Safety & Toxicity Evaluation Service

An end-to-end, production-grade safety evaluation service for text toxicity and content moderation, featuring subgroup fairness evaluations, temperature calibration, real-time alert monitoring, and a human-in-the-loop review workflow.

## Architecture

```
User Request / Upstream LLM Output
                │
                ▼
      ┌──────────────────┐
      │  POST /evaluate  │
      └─────────┬────────┘
                │
                ▼
   ┌────────────────────────┐
   │ Toxicity Classifier    │ (DistilBERT / TF-IDF Logistic Regression)
   └────────────┬───────────┘
                │ Raw score
                ▼
   ┌────────────────────────┐
   │ Temperature Scaling    │ (Calibrated probability)
   └────────────┬───────────┘
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
┌──────────────┐                 ┌───────────────┐
│ Flagged /    │                 │ Rolling Window│
│ Decision     │                 │ Monitor       │
└───────┬──────┘                 └───────┬───────┘
        │ |score - threshold| < ε        │ flagged_rate > limit
        ▼                                ▼
┌──────────────┐                 ┌───────────────┐
│ Review Queue │ (SQLite DB)     │ Alert System  │
└──────────────┘                 └───────────────┘
```

## Results Summary

*Metrics populated from saved experimental artifacts (`metrics.json`).*

| Model | Civil Comments AUROC | Civil Comments PR-AUC | HateXplain AUROC (Domain Shift) | ECE (Uncalibrated) | ECE (Calibrated) | Subgroup Metric (Borkan et al.) |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| DistilBERT Fine-Tuned | TBD | TBD | TBD | TBD | TBD | TBD |

## How to Run

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation & Training
```bash
python -m src.safety_eval.data
python -m src.safety_eval.train_baseline
python -m src.safety_eval.calibrate
```

### API Service
```bash
uvicorn src.safety_eval.api:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
pytest tests/ -v
```

## Ethical Considerations & Limitations
- **Identity Bias**: Content moderation models can exhibit unintended bias against specific identity demographic mentions.
- **Domain Shift**: Performance degradation is expected when deployed outside the training distribution (e.g., forum comments vs. social media microblogs).
- **Nuance & Sarcasm**: Keyword-correlated false positives require continuous human-in-the-loop audit.

## References
1. Borkan et al. (2019) *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification*.
2. Guo et al. (2017) *On Calibration of Modern Neural Networks*.
3. Mathew et al. (2021) *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*.
