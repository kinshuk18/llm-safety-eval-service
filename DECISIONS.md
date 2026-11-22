# Architectural and Experimental Decisions

This document records key decisions backed strictly by empirical artifacts and experimental runs.

## Decisions

### 1. Baseline Text Representation and Classification
- **Question**: What baseline model architecture balances fast CPU inference latency (<10ms) with strong discriminatory power on real toxic comments?
- **What I tried**: Sublinear term-frequency TF-IDF with unigram + bigram representations (`max_features=50000`, `ngram_range=(1,2)`) coupled with L-BFGS Logistic Regression ($C=2.0$).
- **Result (with number)**: Achieved **0.9196 AUROC** and **0.6439 PR-AUC** on 20,000 real Civil Comments test samples with a model size of ~1.8 MB.
- **Decision**: Adopt TF-IDF + LogisticRegression as the high-throughput, low-latency baseline for CPU serving and local development.
- **What failed**: Default $C=1.0$ unigram-only models yielded lower PR-AUC (sub-0.60) on longer comment texts due to missing contextual bigrams (e.g., "not toxic" vs "toxic").

### 2. Demographic Subgroup Unintended Bias
- **Question**: How does the toxicity baseline behave on demographic identity terms according to Borkan et al. (2019)?
- **What I tried**: Tagged 9 demographic groups (`male`, `female`, `lgbtq`, `christian`, `muslim`, `jewish`, `black`, `white`, `psychiatric_or_mental_illness`) and computed Subgroup AUC, BPSN AUC, and BNSP AUC alongside the Generalized Mean ($p=-5$).
- **Result (with number)**: Overall test AUROC is **0.9196**, but BPSN AUC drops severely to **0.7285 for `white`**, **0.7521 for `black`**, and **0.7820 for `lgbtq`**, pulling the final Borkan weighted bias metric down to **0.8681**.
- **Decision**: Preserve subgroup and BPSN/BNSP tracking in all downstream evaluations to identify false-positive penalties against marginalized demographic terms.
- **What failed**: Standard global AUROC completely obscured subgroup degradation, scoring 0.9196 despite a ~19-point drop in BPSN AUC for specific groups.

### 3. Out-of-Domain Generalization (Domain Shift)
- **Question**: How well does a model trained on civil news forum comments transfer to social microblog content?
- **What I tried**: Evaluated the baseline model zero-shot on 1,924 test examples from HateXplain (Twitter/Gab data mapped to binary toxicity).
- **Result (with number)**: AUROC collapsed from **0.9196 (Civil Comments)** to **0.5666 (HateXplain)**, and ECE rose sharply from **0.0228** to **0.3745**.
- **Decision**: Out-of-domain transfer without domain adaptation or transformer fine-tuning produces near-random discrimination on social media syntax, establishing an empirical need for contextual models and active monitoring.
- **What failed**: Assuming news comment moderation weights could directly filter social media hate speech; word distribution mismatch between civil forums and Twitter/Gab destroyed model calibration.

### 4. Dataset Loading Script Incompatibility
- **Question**: How can HateXplain be loaded reproducibly in modern Python environments?
- **What I tried**: Loading `hatexplain` directly via HuggingFace `datasets>=3.0.0` with `trust_remote_code=True`.
- **Result (with number)**: Failed with `RuntimeError: Dataset scripts are no longer supported, but found hatexplain.py` because HF `datasets>=3.0.0` permanently removed remote python builder scripts.
- **Decision**: Downgraded and pinned `datasets<3.0.0` (`datasets==2.21.0`), and used canonical repo ID `Hate-speech-CNERG/hatexplain` with cached Parquet persistence in `data/`.
- **What failed**: Direct `load_dataset('hatexplain')` without version pinning crashed immediately on HuggingFace Datasets 5.x.
