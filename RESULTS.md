# Experimental Results

All numbers recorded in this document are strictly sourced from saved experimental artifacts (`metrics.json`, evaluation logs).

## Metric Definitions
- **AUROC**: Area Under the Receiver Operating Characteristic Curve.
- **PR-AUC**: Area Under the Precision-Recall Curve.
- **F1@Threshold**: Binary F1 score at cost-optimal or target operational threshold (default 0.50).
- **ECE**: Expected Calibration Error (10 bins).
- **Subgroup AUC**: Subgroup-specific AUROC for demographic identity mentions.
- **BPSN AUC**: Background Positive, Subgroup Negative AUROC (measuring false positive bias on non-toxic identity mentions).
- **BNSP AUC**: Background Negative, Subgroup Positive AUROC (measuring false negative bias on toxic identity mentions).
- **Borkan Bias Metric**: Weighted generalized mean ($p=-5$) fairness metric across overall, Subgroup, BPSN, and BNSP AUCs (Borkan et al., 2019).

---

## Model Benchmark Table

| Model | Split | AUROC | PR-AUC | F1 @ 0.5 | ECE | Borkan Bias Metric |
|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | Civil Comments Val | 0.9188 | 0.6491 | 0.4631 | 0.0234 | 0.8654 |
| **TF-IDF + Logistic Regression** | Civil Comments Test | 0.9196 | 0.6439 | 0.4702 | 0.0228 | 0.8681 |
| **TF-IDF + Logistic Regression** | HateXplain (Domain Shift) | 0.5666 | 0.6644 | 0.3401 | 0.3745 | 0.5637 |
| **DistilBERT Fine-Tuned** | Civil Comments Test | *(Pending P3)* | *(Pending P3)* | *(Pending P3)* | *(Pending P3)* | *(Pending P3)* |

---

## Detailed Subgroup Metrics (TF-IDF Baseline on Civil Comments Test)

*Overall AUROC: 0.9196*

| Subgroup | Subgroup AUC | BPSN AUC | BNSP AUC |
|---|---|---|---|
| **male** | 0.8779 | 0.8415 | 0.9470 |
| **female** | 0.8698 | 0.8578 | 0.9363 |
| **lgbtq** | 0.6784 | 0.7820 | 0.8898 |
| **christian** | 0.8759 | 0.9159 | 0.8838 |
| **muslim** | 0.8502 | 0.8086 | 0.9492 |
| **jewish** | 0.8124 | 0.8298 | 0.8930 |
| **black** | 0.8097 | 0.7521 | 0.9570 |
| **white** | 0.8122 | 0.7285 | 0.9647 |
| **psychiatric_or_mental_illness** | 0.9400 | 0.8913 | 0.9500 |
| **Generalized Mean ($p=-5$)** | **0.8078** | **0.7936** | **0.9255** |
| **Final Borkan et al. Bias Metric** | — | — | **0.8681** |
