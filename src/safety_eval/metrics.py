"""Metrics module: AUROC, PR-AUC, F1, ECE, subgroup AUC, BPSN, BNSP, Borkan et al. bias metric."""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from typing import Dict, List, Optional, Tuple, Any

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))

def compute_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob <= bin_upper) if i == n_bins - 1 else (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)

def compute_subgroup_auc(y_true: np.ndarray, y_score: np.ndarray, subgroup_mask: np.ndarray) -> float:
    sub_y_true = y_true[subgroup_mask]
    sub_y_score = y_score[subgroup_mask]
    return compute_auroc(sub_y_true, sub_y_score)

def compute_bpsn_auc(y_true: np.ndarray, y_score: np.ndarray, subgroup_mask: np.ndarray) -> float:
    # Background Positive (y=1, non-subgroup) + Subgroup Negative (y=0, subgroup)
    bpsn_mask = ((y_true == 1) & (~subgroup_mask)) | ((y_true == 0) & subgroup_mask)
    if np.sum(bpsn_mask) == 0:
        return float("nan")
    return compute_auroc(y_true[bpsn_mask], y_score[bpsn_mask])

def compute_bnsp_auc(y_true: np.ndarray, y_score: np.ndarray, subgroup_mask: np.ndarray) -> float:
    # Background Negative (y=0, non-subgroup) + Subgroup Positive (y=1, subgroup)
    bnsp_mask = ((y_true == 0) & (~subgroup_mask)) | ((y_true == 1) & subgroup_mask)
    if np.sum(bnsp_mask) == 0:
        return float("nan")
    return compute_auroc(y_true[bnsp_mask], y_score[bnsp_mask])

def generalized_mean(values: List[float], p: float = -5.0) -> float:
    clean_vals = [v for v in values if not np.isnan(v)]
    if not clean_vals:
        return float("nan")
    arr = np.array(clean_vals, dtype=float)
    if p == 0:
        return float(np.exp(np.mean(np.log(arr))))
    return float(np.power(np.mean(np.power(arr, p)), 1.0 / p))

def compute_borkan_bias_metric(
    overall_auc: float,
    subgroup_aucs: Dict[str, float],
    bpsn_aucs: Dict[str, float],
    bnsp_aucs: Dict[str, float],
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    p: float = -5.0,
) -> float:
    g_sub = generalized_mean(list(subgroup_aucs.values()), p=p)
    g_bpsn = generalized_mean(list(bpsn_aucs.values()), p=p)
    g_bnsp = generalized_mean(list(bnsp_aucs.values()), p=p)
    w0, w1, w2, w3 = weights
    score = w0 * overall_auc + w1 * g_sub + w2 * g_bpsn + w3 * g_bnsp
    return float(score)
