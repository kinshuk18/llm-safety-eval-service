"""Data preparation module: loads google/civil_comments and HateXplain,
performs stratified subsampling (~150k train, 20k val, 20k test),
tags identity subgroup mentions, and persists splits and index files."""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
IDENTITY_TERMS = {
    "male": [r"\bmale\b", r"\bman\b", r"\bmen\b", r"\bboy\b", r"\bboys\b"],
    "female": [r"\bfemale\b", r"\bwoman\b", r"\bwomen\b", r"\bgirl\b", r"\bgirls\b"],
    "lgbtq": [r"\bgay\b", r"\blesbian\b", r"\bhomosexual\b", r"\bbisexual\b", r"\bqueer\b", r"\blgbt\b", r"\blgbtq\b", r"\btransgender\b"],
    "christian": [r"\bchristian\b", r"\bchristians\b", r"\bchristianity\b", r"\bjesus\b", r"\bchurch\b"],
    "muslim": [r"\bmuslim\b", r"\bmuslims\b", r"\bislam\b", r"\bislamic\b", r"\bquran\b", r"\ballah\b"],
    "jewish": [r"\bjew\b", r"\bjews\b", r"\bjewish\b", r"\bjudaism\b"],
    "black": [r"\bblack\b", r"\bblacks\b", r"\bafrican american\b"],
    "white": [r"\bwhite\b", r"\bwhites\b", r"\bcaucasian\b"],
    "psychiatric_or_mental_illness": [r"\bmental illness\b", r"\bdepression\b", r"\bbipolar\b", r"\bschizophrenia\b", r"\bpsychiatric\b"],
}

# Precompile regexes
IDENTITY_PATTERNS = {
    group: re.compile("|".join(patterns), re.IGNORECASE)
    for group, patterns in IDENTITY_TERMS.items()
}

def tag_identities(text: str) -> Dict[str, int]:
    if not isinstance(text, str):
        return {group: 0 for group in IDENTITY_PATTERNS}
    return {group: int(bool(pat.search(text))) for group, pat in IDENTITY_PATTERNS.items()}

def prepare_civil_comments(
    train_size: int = 150000,
    val_size: int = 20000,
    test_size: int = 20000,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Loads google/civil_comments and creates stratified subsamples."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading google/civil_comments from Hugging Face...")
    ds = load_dataset("google/civil_comments")

    def process_split(hf_split, target_n: int, split_name: str) -> Tuple[pd.DataFrame, List[int]]:
        logger.info(f"Processing split {split_name} (original size: {len(hf_split)})...")
        # Extract toxicity and binary label
        toxicities = np.array(hf_split["toxicity"], dtype=np.float32)
        labels = (toxicities >= 0.5).astype(int)

        # Stratified subsampling
        n_total = len(labels)
        if target_n >= n_total:
            chosen_indices = np.arange(n_total)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=target_n, random_state=seed)
            chosen_indices, _ = next(sss.split(np.zeros(n_total), labels))

        sub_ds = hf_split.select(chosen_indices)
        texts = sub_ds["text"]
        sub_toxicities = toxicities[chosen_indices]
        sub_labels = labels[chosen_indices]
        sub_identity_attack = np.array(sub_ds["identity_attack"], dtype=np.float32)

        df = pd.DataFrame({
            "text": texts,
            "toxicity": sub_toxicities,
            "label": sub_labels,
            "identity_attack": sub_identity_attack,
        })

        # Tag identity terms
        logger.info(f"Tagging identity groups for {split_name}...")
        id_tags = [tag_identities(t) for t in texts]
        id_df = pd.DataFrame(id_tags)
        for col in id_df.columns:
            df[col] = id_df[col].values

        return df, chosen_indices.tolist()

    train_df, train_idx = process_split(ds["train"], train_size, "train")
    val_df, val_idx = process_split(ds["validation"], val_size, "val")
    test_df, test_idx = process_split(ds["test"], test_size, "test")

    # Save index files
    logger.info("Saving index files...")
    with open(DATA_DIR / "train_indices.json", "w") as f:
        json.dump(train_idx, f)
    with open(DATA_DIR / "val_indices.json", "w") as f:
        json.dump(val_idx, f)
    with open(DATA_DIR / "test_indices.json", "w") as f:
        json.dump(test_idx, f)

    # Save parquets
    train_df.to_parquet(DATA_DIR / "civil_comments_train.parquet", index=False)
    val_df.to_parquet(DATA_DIR / "civil_comments_val.parquet", index=False)
    test_df.to_parquet(DATA_DIR / "civil_comments_test.parquet", index=False)
    logger.info("Civil Comments splits saved successfully.")

    return {"train": train_df, "val": val_df, "test": test_df}

def prepare_hatexplain() -> pd.DataFrame:
    """Loads HateXplain external test set, maps to binary labels under two mappings:
    M1: hateful/offensive -> 1, normal -> 0
    M2: hateful -> 1, normal -> 0, offensive -> -1 (excluded)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading HateXplain test split...")
    ds = load_dataset("Hate-speech-CNERG/hatexplain", trust_remote_code=True)
    test_split = ds["test"]

    texts = []
    labels = []
    labels_m1 = []
    labels_m2 = []
    maj_classes = []
    ids = []
    hatespeech_votes = []
    offensive_votes = []
    normal_votes = []

    for item in test_split:
        text = " ".join(item["post_tokens"])
        ann_labels = item["annotators"]["label"]
        # names=["hatespeech", "normal", "offensive"] => 0: hatespeech, 1: normal, 2: offensive
        v_hate = sum(1 for l in ann_labels if l == 0)
        v_norm = sum(1 for l in ann_labels if l == 1)
        v_off = sum(1 for l in ann_labels if l == 2)

        # M1: hateful/offensive -> 1, normal -> 0
        m1 = 1 if (v_hate + v_off) > v_norm else 0

        # Majority 3-class (each post has 3 annotators, no ties in test set)
        if v_hate >= 2:
            maj = 0
            m2 = 1
        elif v_norm >= 2:
            maj = 1
            m2 = 0
        else:
            maj = 2
            m2 = -1  # Excluded under M2

        texts.append(text)
        labels.append(m1)
        labels_m1.append(m1)
        labels_m2.append(m2)
        maj_classes.append(maj)
        ids.append(item["id"])
        hatespeech_votes.append(v_hate)
        offensive_votes.append(v_off)
        normal_votes.append(v_norm)

    df = pd.DataFrame({
        "id": ids,
        "text": texts,
        "label": labels,
        "label_m1": labels_m1,
        "label_m2": labels_m2,
        "majority_label_3class": maj_classes,
        "hatespeech_votes": hatespeech_votes,
        "offensive_votes": offensive_votes,
        "normal_votes": normal_votes,
        "hateful_offensive_votes": [h + o for h, o in zip(hatespeech_votes, offensive_votes)],
    })

    # Add identity tags for HateXplain
    id_tags = [tag_identities(t) for t in texts]
    id_df = pd.DataFrame(id_tags)
    for col in id_df.columns:
        df[col] = id_df[col].values

    df.to_parquet(DATA_DIR / "hatexplain_test.parquet", index=False)
    logger.info(f"HateXplain test saved ({len(df)} samples, M1 positive rate={df['label_m1'].mean():.4f}, M2 valid={sum(df['label_m2'] >= 0)}).")
    return df

def main():
    splits = prepare_civil_comments(train_size=150000, val_size=20000, test_size=20000, seed=42)
    hate_df = prepare_hatexplain()
    print("Data preparation complete.")
    for split_name, df in splits.items():
        print(f"Civil Comments {split_name}: {len(df)} rows, toxic={df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    print(f"HateXplain test: {len(hate_df)} rows, toxic={hate_df['label'].sum()} ({hate_df['label'].mean()*100:.2f}%)")

if __name__ == "__main__":
    main()
