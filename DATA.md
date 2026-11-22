# Dataset Documentation

## Overview
This service utilizes two real-world datasets for training, in-domain validation/testing, subgroup fairness auditing, and out-of-domain shift testing. No synthetic data is used.

---

## 1. Civil Comments (In-Domain & Subgroups)
- **Source**: Hugging Face `google/civil_comments` (Jigsaw / Conversation AI)
- **License**: CC0-1.0 (Public Domain Dedication, verified from dataset card)
- **Total Dataset Size**: 1,804,874 train / 97,320 validation / 97,320 test
- **Task**: Binary toxicity detection (`toxicity >= 0.5`)
- **Sampling Strategy**: Stratified subsampling with fixed seed (`seed=42`) using `StratifiedShuffleSplit`:
  - **Train**: 150,000 samples | 11,995 positive (8.00% toxicity rate)
  - **Validation**: 20,000 samples | 1,576 positive (7.88% toxicity rate)
  - **Test**: 20,000 samples | 1,598 positive (7.99% toxicity rate)
- **Identity Demographics**: 9 demographic subgroup indicator columns tagged for Borkan et al. (2019) unintended bias evaluation:
  - `male`, `female`, `lgbtq`, `christian`, `muslim`, `jewish`, `black`, `white`, `psychiatric_or_mental_illness`
- **Saved Index Files**:
  - `data/train_indices.json`
  - `data/val_indices.json`
  - `data/test_indices.json`
- **Saved Data Files**:
  - `data/civil_comments_train.parquet`
  - `data/civil_comments_val.parquet`
  - `data/civil_comments_test.parquet`

---

## 2. HateXplain (Out-of-Domain External Test)
- **Source**: Hugging Face `Hate-speech-CNERG/hatexplain` (Mathew et al., 2021)
- **License**: CC-BY-4.0 (Creative Commons Attribution 4.0 International, verified from dataset script/card)
- **Dataset Size**: 1,924 test samples
- **Label Mapping**:
  - Original annotator votes: `0: hatespeech`, `1: normal`, `2: offensive`
  - Binary target mapping: majority vote of `hateful` or `offensive` $\rightarrow 1$, `normal` $\rightarrow 0$
  - Positive Class Count: 1,142 positive (59.36% prevalence)
- **Purpose**: Measure cross-domain performance degradation (domain shift from news comments to Twitter/Gab microblog posts).
- **Saved Data File**:
  - `data/hatexplain_test.parquet`
