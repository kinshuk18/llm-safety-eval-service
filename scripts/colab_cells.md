# Google Colab Execution Guide for DistilBERT Fine-Tuning

This guide provides the exact cells to run in Google Colab (with a free T4 GPU runtime) to fine-tune `distilbert-base-uncased` on the Civil Comments dataset, calibrate predictions, and evaluate on Civil Comments and HateXplain.

---

### Step 0: What to Prepare and Upload

Before running the Colab notebook, package the prepared parquet files locally:
```bash
zip -j data.zip data/civil_comments_train.parquet data/civil_comments_val.parquet data/civil_comments_test.parquet data/hatexplain_test.parquet
```

Files to upload when prompted by Cell 3:
- **`data.zip`**: Contains the 4 prepared splits (~85 MB compressed).

---

### Colab Cells

#### Cell 1: Check GPU Runtime
```python
# Verify that a GPU (e.g. NVIDIA T4 / V100 / A100) is allocated
!nvidia-smi
```

#### Cell 2: Clone Repo and Install Dependencies
```python
# Clone repository and switch into project root
!git clone https://github.com/<your-username>/llm-safety-eval-service.git
%cd llm-safety-eval-service

# Install runtime dependencies
!pip install -q torch transformers datasets accelerate scikit-learn pandas pyarrow pyyaml scipy joblib
```

#### Cell 3: Upload `data.zip` and Unpack
```python
import os
from google.colab import files
import zipfile

# Create data directory
os.makedirs("data", exist_ok=True)

# Upload data.zip
print("Please upload data.zip created with:")
print("zip -j data.zip data/civil_comments_train.parquet data/civil_comments_val.parquet data/civil_comments_test.parquet data/hatexplain_test.parquet")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall("data")
        print(f"Extracted {filename} into data/")

# Verify parquet files are extracted
!ls -lh data/*.parquet
```

#### Cell 4: Run DistilBERT Fine-Tuning and Evaluation
```python
# Run training, calibration, and evaluation pipeline
# (Hyperparameters: lr=2e-5, batch_size=32, 1 epoch, max_len=128, fp16=True)
!python -m src.safety_eval.train_transformer

# Regenerate RESULTS.md comparison table
!python scripts/make_results.py
```

#### Cell 5: Package and Download Artifacts
```python
from google.colab import files
import shutil

# Archive trained transformer model
shutil.make_archive("distilbert_safety_model", "zip", "models/distilbert_safety")

# Download model archive and updated metrics.json
files.download("distilbert_safety_model.zip")
files.download("metrics.json")
files.download("RESULTS.md")
print("Downloaded model weights and updated metrics!")
```

---

### Step 6: Bring Artifacts Back to Local Repo
After downloading the artifacts from Colab:
1. Move `metrics.json` and `RESULTS.md` into the project root.
2. Unzip `distilbert_safety_model.zip` into `models/distilbert_safety/`:
   ```bash
   unzip -d models/distilbert_safety distilbert_safety_model.zip
   ```
