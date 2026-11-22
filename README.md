# LLM Safety & Toxicity Evaluation Service

![CI](https://github.com/kinshuk18/llm-safety-eval-service/actions/workflows/ci.yml/badge.svg)

A production-grade REST service for real-time text toxicity detection and content moderation. Given any input text the service returns a calibrated toxicity probability, applies an asymmetric cost-optimal decision threshold, routes borderline predictions into a human-in-the-loop review queue, and streams rolling flagged-rate alerts to downstream observers. All models are evaluated against subgroup fairness metrics (Borkan et al. 2019) and out-of-domain benchmarks (HateXplain) to surface demographic bias before deployment.

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
   │ Toxicity Classifier    │ (TF-IDF + Logistic Regression; DistilBERT offline)
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

## Results

| Model | Civil AUROC | PR-AUC | Borkan | ECE (cal) | Worst BPSN | HX M1 AUROC | HX M2 AUROC |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.9196 | 0.6439 | 0.8681 | 0.0153 | 0.7285 (white) | 0.5666 | 0.5564 |
| DistilBERT | 0.9562 | 0.7472 | 0.9189 | 0.0069 | 0.8537 (black) | 0.6329 | 0.6219 |

*Civil = 20k Civil Comments test samples; HX M1 = hateful+offensive→1; HX M2 = hateful→1 only.*
*Baseline model weights (2.3 MB total) are tracked in git under `models/`.*

## Quickstart (fresh clone)

```bash
git clone https://github.com/kinshuk18/llm-safety-eval-service.git
cd llm-safety-eval-service

# Requires Python 3.12+ (scipy 1.18 needs it).
# 1. Create virtual environment
uv venv --python 3.12 .venv && source .venv/bin/activate

# 2. Install serving dependencies
uv pip install -r requirements.txt

# 3. Install training dependencies (data prep + model training)
uv pip install -r requirements-train.txt

# 4. Download and prepare data (~150k train, 20k val/test)
python -m src.safety_eval.data

# 5. Train TF-IDF + Logistic Regression baseline
python -m src.safety_eval.train_baseline

# 6. Fit temperature calibrator
python -m src.safety_eval.calibrate

# 7. Run the API
uvicorn src.safety_eval.api:app --host 0.0.0.0 --port 8000

# 8. Test
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/evaluate \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test comment."}'

# 9. Docker (serving image, ~200 MB; no torch)
docker build -t safety-eval .
docker run -p 8000:8000 safety-eval
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check; returns model version |
| POST | `/evaluate` | Score text; returns calibrated probability, flagged, threshold, latency |
| GET | `/monitor` | Rolling window flagged-rate and alert status |
| GET | `/review/pending` | List borderline items awaiting human review |
| POST | `/review/{id}` | Submit human label (0=safe, 1=toxic) for item |
| POST | `/review/export` | Export reviewed labels to CSV |

## DistilBERT Model

Trained on Colab T4 GPU (1 epoch, 150k train, lr=2e-5, batch=32, max_len=128, fp16, transformers<5). **Weights are not stored in this repository** — the service serves the baseline TF-IDF model; DistilBERT results are offline evaluation only. To reproduce, run `src/safety_eval/train_transformer.py` with `requirements-train.txt`.

## Limitations

- **Domain shift**: HateXplain AUROC collapses to 0.5666 (M1) — barely above chance. The model is trained on news-forum comments and does not generalize to social microblog speech without retraining.
- **Identity-term FPR**: At the cost-optimal threshold, FPR for identity-mentioning comments is 3–5× overall for DistilBERT (lgbtq 0.3294, black 0.3177, white 0.2994 vs. overall 0.0645). Improved BPSN AUC does not eliminate this.
- **Threshold cost assumption**: Thresholds are tuned with a 5:1 FN:FP cost ratio. Deployments with different cost profiles must re-tune on their own validation data.
- **Binary only**: The service predicts a single toxic/safe score; nuanced categories (threats, identity attacks, etc.) are not distinguished.
- **Not a compliance guarantee**: This tool surfaces risk signals; human review is required before any punitive enforcement.
- **Small subgroup sizes**: BPSN/BNSP AUCs for rare identity groups have high variance due to limited test samples.

## Ethics

Training datasets (Civil Comments, HateXplain) contain toxic, hateful, and abusive text. This system is intended for assisting human moderators, not for fully automated punitive decisions. Demographic bias persists even in fine-tuned models; deploy with appropriate oversight and audit.

## References

1. **Borkan et al. 2019** — *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.* Introduced the generalized mean (p=−5) Borkan bias metric and the BPSN/BNSP AUC framework used in all subgroup fairness evaluations here.
2. **Guo et al. 2017** — *On Calibration of Modern Neural Networks.* Temperature scaling post-hoc calibration method applied to reduce ECE from 0.0228 (uncalibrated) to 0.0153.
3. **Mathew et al. 2021** — *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* Used as the out-of-domain evaluation benchmark to measure domain-shift degradation under two annotation mappings (M1, M2).
