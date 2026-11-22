# Experimental Results

All numbers recorded in this document are strictly sourced from saved experimental artifacts (`metrics.json`, evaluation logs).

## Metric Definitions
- **AUROC**: Area Under the Receiver Operating Characteristic Curve.
- **PR-AUC**: Area Under the Precision-Recall Curve.
- **F1@0.5**: Binary F1 score at default uncalibrated decision threshold (0.50).
- **F1@Cost-Opt**: Binary F1 score at cost-optimal threshold (minimizing expected cost with 5:1 FN:FP ratio on validation).
- **ECE**: Expected Calibration Error (10 bins).
- **Subgroup AUC**: Subgroup-specific AUROC for demographic identity mentions.
- **BPSN AUC**: Background Positive, Subgroup Negative AUROC (measuring false positive bias on non-toxic identity mentions).
- **BNSP AUC**: Background Negative, Subgroup Positive AUROC (measuring false negative bias on toxic identity mentions).
- **Borkan Bias Metric**: Weighted generalized mean ($p=-5$) fairness metric across overall, Subgroup, BPSN, and BNSP AUCs (Borkan et al., 2019).

---

## Model Benchmark Comparison Table

| Model | Split / Condition | AUROC | PR-AUC | F1 @ 0.5 | F1 @ Cost-Opt | ECE (Uncal -> Cal) | Borkan Bias Metric | Cost-Opt Threshold |
|---|---|---|---|---|---|---|---|---|
| **TF-IDF + LR** | Civil Comments Val | 0.9188 | 0.6491 | 0.4631 | — | 0.0234 | 0.8654 | — |
| **TF-IDF + LR** | Civil Comments Test | 0.9196 | 0.6439 | 0.4702 | 0.5717 | 0.0228 -> 0.0153 | 0.8681 | 0.1161 |
| **TF-IDF + LR** | HateXplain M1 (Hateful+Offensive -> 1) | 0.5666 | 0.6644 | 0.3401 | — | 0.3745 | 0.5637 | — |
| **TF-IDF + LR** | HateXplain M2 (Hateful -> 1, Off Excl) | 0.5564 | 0.5063 | 0.3058 | — | 0.2639 | 0.5622 | — |
| **DistilBERT** | Civil Comments Test | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* |
| **DistilBERT** | HateXplain M1 / M2 | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* | *(Pending Colab GPU)* |

---

## Detailed Subgroup Metrics (Civil Comments Test Split)

*Overall Test AUROC: 0.9196 | Borkan Bias Metric: 0.8681*

| Subgroup | Subgroup AUC | BPSN AUC | BNSP AUC | False Positive Rate @ $\tau=0.1161$ |
|---|---|---|---|---|
| **male** | 0.8779 | 0.8415 | 0.9470 | 0.1584 |
| **female** | 0.8698 | 0.8578 | 0.9363 | 0.1556 |
| **lgbtq** | 0.6784 | 0.7820 | 0.8898 | 0.3059 |
| **christian** | 0.8759 | 0.9159 | 0.8838 | 0.0774 |
| **muslim** | 0.8502 | 0.8086 | 0.9492 | 0.2206 |
| **jewish** | 0.8124 | 0.8298 | 0.8930 | 0.1829 |
| **black** | 0.8097 | 0.7521 | 0.9570 | 0.3125 |
| **white** | 0.8122 | 0.7285 | 0.9647 | 0.3796 |
| **psychiatric_or_mental_illness** | 0.9400 | 0.8913 | 0.9500 | 0.1000 |
| **Overall / Generalized Mean ($p=-5$)** | **0.8078** | **0.7936** | **0.9255** | **0.0647** |

---

## Latency Benchmark (CPU Serving)

| Model | Sample Size | Batch Size | p50 Latency (ms) | p95 Latency (ms) |
|---|---|---|---|---|
| **TF-IDF + LR Baseline** | 500 | 1 | 0.37 ms | 0.49 ms |
