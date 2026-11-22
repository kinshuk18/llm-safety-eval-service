# Architectural and Experimental Decisions

This document records key engineering and modeling decisions backed strictly by empirical artifacts and experimental runs. All entries follow a uniform structure: Question, What I tried, Result (with real numbers), Decision, and What failed.

---

## Decisions

### 1. Baseline Text Representation and Model Architecture
- **Question**: What baseline model architecture balances high-throughput CPU inference latency (<1ms) with competitive discriminatory performance on real-world toxicity moderation?
- **What I tried**: Sublinear term-frequency TF-IDF with unigram and bigram representations (`max_features=50000`, `ngram_range=(1,2)`) combined with L-BFGS Logistic Regression ($C=2.0$).
- **Result (with number)**: Achieved **0.9196 AUROC** and **0.6439 PR-AUC** on 20,000 real Civil Comments test samples, with real CPU serving latency of **0.37 ms p50** and **0.49 ms p95** over 500 test evaluations.
- **Decision**: Adopt TF-IDF + LogisticRegression as the default low-latency, CPU-deployable baseline service for online inference.
- **What failed**: Unigram-only counts without sublinear term frequency yielded lower PR-AUC (0.589) due to extreme sensitivity to high-frequency comment vocabulary and inability to capture negated phrases (e.g., "not bad").

### 2. Operational Decision Threshold vs F1@0.5 Under Class Imbalance
- **Question**: Why does default binary classification at $\tau=0.50$ fail for online toxicity moderation, and how should an operating threshold be chosen?
- **What I tried**: Compared default threshold $\tau=0.50$ against cost-sensitive threshold search minimizing expected loss $L = 5.0 \cdot \text{FN} + 1.0 \cdot \text{FP}$ (5:1 penalty reflecting the operational severity of letting toxic content through), optimized on validation data only.
- **Result (with number)**: At default $\tau=0.50$, recall is only **0.3479** and F1 is **0.4702**. The cost-optimal threshold converged to $\tau=0.1161$ on validation (cost: 3468.0), which on test improved recall to **0.6984** and F1 to **0.5717** (+10.15 points) at a test false positive rate of **0.0647**.
- **Decision**: Reject $\tau=0.50$; configure the production service to load the empirical cost-optimal threshold ($\tau=0.1161$) from `configs/thresholds.yaml`.
- **What failed**: Using standard F1-maximization without an asymmetric cost matrix selected $\tau \approx 0.32$, which still suffered from an unacceptably high false-negative rate on subtle abusive content.

### 3. Probability Calibration via Platt Temperature Scaling
- **Question**: How well do raw model scores represent true posterior toxicity probabilities, and how can calibration error be minimized post-hoc?
- **What I tried**: Fitted post-hoc Temperature Scaling (Guo et al. 2017) on validation set logits using negative log-likelihood minimization, then validated Expected Calibration Error (ECE, 10 bins) on the unseen test split.
- **Result (with number)**: The optimal temperature parameter converged to $T=0.9068$ on validation. On the 20,000 test set, ECE improved from **0.0228 (uncalibrated)** down to **0.0153 (calibrated)**, a 32.9% reduction in calibration error.
- **Decision**: Persist `models/calibrator.joblib` and route all online scoring through `TemperatureScaler.predict_proba()` before applying operational thresholds and review margins.
- **What failed**: Fitting temperature scaling directly on training split logits caused severe overconfidence ($T \approx 0.45$) because training cross-entropy was already over-optimized, yielding worse test ECE (0.0381).

### 4. Demographic Subgroup Bias and BPSN AUC Degradation
- **Question**: How does the toxicity model behave when evaluating benign mentions of protected demographic identity groups?
- **What I tried**: Evaluated 9 demographic identity groups (`male`, `female`, `lgbtq`, `christian`, `muslim`, `jewish`, `black`, `white`, `psychiatric_or_mental_illness`) on Civil Comments test set, computing Subgroup AUC, Background Positive Subgroup Negative (BPSN) AUC, Background Negative Subgroup Positive (BNSP) AUC, and False Positive Rates at $\tau=0.1161$.
- **Result (with number)**: Despite a high overall AUROC of **0.9196**, BPSN AUC severely collapsed to **0.7285 for `white`**, **0.7521 for `black`**, and **0.7820 for `lgbtq`**, reducing the overall Borkan et al. generalized mean ($p=-5$) bias metric to **0.8681**. Correspondingly, subgroup FPR at $\tau=0.1161$ reached **0.3796 for `white`**, **0.3125 for `black`**, and **0.3059 for `lgbtq`**, compared to **0.0647 overall**.
- **Decision**: Expose subgroup false positive rates and BPSN AUC metrics alongside standard discrimination metrics to prevent biased models from being deployed based solely on aggregate AUROC.
- **What failed**: In plain words, a low BPSN score means non-toxic comments mentioning demographic terms are systematically misclassified as toxic because the model associates demographic identity tokens with toxicity. Mitigations that were NOT tried in this baseline include: subgroup-weighted cross-entropy loss (Borkan et al. 2019), counterfactual identity-term token swapping during data augmentation, and adversarial representation debiasing.

### 5. Out-of-Domain Generalization and Annotation Label Mismatch
- **Question**: How well does a classifier trained on civil news forum comments transfer to social microblog content, and how sensitive is it to label definitions?
- **What I tried**: Evaluated zero-shot transfer on 1,924 test samples from HateXplain (Twitter/Gab) across two annotation mappings: Mapping 1 (M1: hateful + offensive $\rightarrow$ 1, normal $\rightarrow$ 0; $N=1,924$) and Mapping 2 (M2: hateful $\rightarrow$ 1, normal $\rightarrow$ 0, offensive excluded; $N=1,376$).
- **Result (with number)**: Under M1, AUROC collapsed from **0.9196 (Civil Comments)** to **0.5666 (HateXplain)**, PR-AUC was **0.6644**, and ECE spiked to **0.3745**. Under M2, AUROC dropped to **0.5564**, PR-AUC dropped to **0.5063**, and ECE was **0.2639**.
- **Decision**: Out-of-domain transfer without contextual adaptation performs barely better than random chance. Both M1 and M2 are tracked in `metrics.json` to capture the fundamental label mismatch between news forum moderation and social media speech.
- **What failed**: Assuming that excluding ambiguous offensive slang (M2) would resolve the domain shift; the baseline bag-of-words architecture fundamentally lacks the contextual capacity to distinguish targeted abuse from colloquial ingroup vernacular.

### 6. Dataset Script Deprecation in Hugging Face Datasets
- **Question**: How can HateXplain be loaded reproducibly in modern Python environments given upstream Hugging Face API deprecations?
- **What I tried**: Attempted `load_dataset("hatexplain")` using `datasets>=3.0.0`.
- **Result (with number)**: Failed with `RuntimeError: Dataset scripts are no longer supported, but found hatexplain.py` because Hugging Face Datasets 3.0+ permanently removed support for remote dataset loading scripts.
- **Decision**: Pinned `datasets==2.21.0` in `requirements.txt` for dataset preparation, and persisted the extracted splits locally as standardized Parquet tables (`data/hatexplain_test.parquet`).
- **What failed**: Attempting to load using `datasets>=3.0.0` with `trust_remote_code=True` still failed because upstream removed the script runner mechanism entirely.

### 7. FastAPI Route Order Precedence and Review Queue Export
- **Question**: Why did POST requests to `/review/export` initially return HTTP 422 Unprocessable Entity during review queue operations?
- **What I tried**: Inspected FastAPI routing table order when defining human-in-the-loop review queue endpoints (`/review/pending`, `/review/export`, and `/review/{item_id}`).
- **Result (with number)**: When `/review/{item_id}` was declared prior to `/review/export`, Starlette's route matcher matched `/review/export` against `{item_id}` with `"export"`, triggering a FastAPI validation error (`value is not a valid integer`) with HTTP 422.
- **Decision**: Strictly declare static path endpoints (`/review/pending`, `/review/export`) before dynamic parameterized path endpoints (`/review/{item_id}`) in `src/safety_eval/api.py`.
- **What failed**: Relying on framework-level routing heuristics to distinguish string literal paths from typed path parameters; Starlette evaluates route patterns sequentially.

### 8. Term Association in Linear Model False Positives
- **Question**: What linguistic patterns drive false positive classifications in the linear baseline model?
- **What I tried**: Analyzed the 1,190 false positive comments on the Civil Comments test split against model logistic regression coefficients and token frequency.
- **Result (with number)**: Beyond general argument words (`stupid` with coef 28.18, `ridiculous` with coef 13.93), demographic identity terms carried high positive weights: `white` (coef +7.50, present in 169 false positives), `black` (coef +6.91, present in 81 false positives), and `racist` (coef +6.38, present in 54 false positives).
- **Decision**: Deploy an active human-in-the-loop review queue with a $\pm 0.10$ tolerance margin around the decision threshold to intercept borderline comments before automated enforcement.
- **What failed**: Stopword removal alone failed to resolve the issue because identity terms and discussion nouns are legitimate content words, not standard stopwords.
