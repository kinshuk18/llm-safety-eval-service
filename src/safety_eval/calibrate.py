"""Calibration and threshold optimization module using temperature scaling."""
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Any
import yaml

class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration (Guo et al. 2017)."""
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(temperature, 1e-3)

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        def nll_loss(T: float) -> float:
            scaled_logits = logits / max(T, 1e-3)
            # Binary cross entropy
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            bce = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
            return float(bce)

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    n_thresholds: int = 200,
) -> Tuple[float, float]:
    """Find the threshold minimizing expected cost = FN * fn_cost + FP * fp_cost."""
    best_thresh = 0.5
    min_cost = float("inf")
    candidate_thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    for thresh in candidate_thresholds:
        preds = (y_prob >= thresh).astype(int)
        fn = np.sum((y_true == 1) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        cost = fn * fn_cost + fp * fp_cost
        if cost < min_cost:
            min_cost = cost
            best_thresh = float(thresh)

    return best_thresh, float(min_cost)
