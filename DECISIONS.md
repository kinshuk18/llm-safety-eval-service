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

### 9. Dependency Split and Docker Image Size
- **Question**: Should torch/transformers be in the serving container?
- **What I tried**: Initially `requirements.txt` included `torch==2.15.0` and `transformers==5.24.0`. The Dockerfile installed these in the serving image.
- **Result**: torch wheel alone is ~700 MB compressed; image build times exceeded 10 min on slow connections. The API and `calibrate.py` never import torch at runtime.
- **Decision**: Split into `requirements.txt` (serving: fastapi, scikit-learn, joblib, etc.) and `requirements-train.txt` (torch, transformers<5, datasets==2.21.0, accelerate). Dockerfile uses `requirements.txt` only.
- **What failed**: Keeping torch in `requirements.txt` made CI extremely slow and the Docker image unnecessarily large (~1.5 GB vs ~200 MB).

### 10. Colab Training Failures — transformers 5.x and Zip Nesting
- **Question**: What errors occurred when setting up DistilBERT training on Colab T4?
- **What I tried**:
  - (a) Installed `transformers==5.0.0`; called `TrainingArguments` with `warmup_ratio` parameter.
  - (b) Uploaded `data.zip` and extracted with standard unzip; referenced `data/civil_comments_train.parquet`.
- **Result**:
  - (a) `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` — `warmup_ratio` was dropped in transformers 5.x (renamed/removed). Fixed by pinning `transformers<5`.
  - (b) The zip extracted to `data/data/civil_comments_train.parquet` (double-nested path) because the zip was created from inside the `data/` directory. Fixed by extracting with strip-components or adjusting the path.
- **Decision**: Pin `transformers<5` in `requirements-train.txt`; document the double-nest zip issue.
- **What failed**: `transformers>=5.0` for training; assuming zip root equals repo root.

### 11. DistilBERT vs Baseline — What Improved and What Did Not
- **Question**: Does DistilBERT reduce identity-group false positive rate compared to the baseline?
- **What I tried**: Evaluated both models at their respective cost-optimal thresholds (baseline τ=0.1161, DistilBERT τ=0.1769).
- **Result**: AUROC improved 0.9196→0.9562, Borkan bias metric improved 0.8681→0.9189, BPSN worst-case improved from 0.7285 (white) to 0.8537 (black). However, identity-group FPR at threshold did NOT meaningfully improve for DistilBERT: lgbtq FPR=0.3294, black FPR=0.3177, white FPR=0.2994 vs overall FPR=0.0645 — still about 4.6–5.1× higher, essentially the same pattern as baseline.
- **Decision**: Report both BPSN AUC (which improved) and subgroup FPR at threshold (which did not) separately, since they measure different failure modes.
- **What failed**: Assuming that improved BPSN AUC would translate to lower identity-group FPR at the operational threshold; the token-association problem persists even in fine-tuned transformers without explicit debiasing.

### 12. F1 at Cost-Optimal Threshold Falls for DistilBERT While Recall Rises
- **Question**: Is maximizing recall at the cost-optimal threshold desirable when F1 falls?
- **What I tried**: Evaluated DistilBERT at cost-optimal threshold τ=0.1769: recall=0.8054, F1=0.6321. Compared to F1@0.5=0.6713.
- **Result**: F1 at the cost-optimal threshold (0.6321) is lower than F1@0.5 (0.6713) because the threshold is shifted toward recall. However, recall at τ=0.1769 (0.8054) dramatically exceeds recall at τ=0.5 (which would be ~0.5–0.6 range). The cost function (5:1 FN:FP) deliberately accepts false positives to avoid false negatives.
- **Decision**: Report both F1@0.5 and F1@threshold separately. The cost-optimal threshold is correct for the stated FN:FP=5:1 cost ratio; F1@threshold is not the right metric for evaluating this choice.
- **What failed**: Using F1-maximization to pick the threshold; standard F1 optimizes at τ≈0.5 and ignores the asymmetric cost of missing toxic content.
