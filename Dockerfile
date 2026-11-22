FROM python:3.12-slim

WORKDIR /app

# Install serving-only dependencies (no torch/transformers/datasets)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed for serving
COPY src/ src/
COPY configs/ configs/
COPY models/baseline_model.joblib models/baseline_model.joblib
COPY models/calibrator.joblib models/calibrator.joblib

# Non-root user for security; /app must be writable because the review queue creates review_queue.db there
RUN adduser --disabled-password --gecos "" appuser \
    && chown appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.safety_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]
