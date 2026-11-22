import pytest
import numpy as np
from src.safety_eval.metrics import (
    compute_auroc,
    compute_pr_auc,
    compute_f1,
    compute_ece,
    compute_subgroup_auc,
    compute_bpsn_auc,
    compute_bnsp_auc,
    compute_borkan_bias_metric,
    generalized_mean,
)

def test_compute_auroc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auroc(y_true, y_score) == 1.0

def test_compute_pr_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    pr = compute_pr_auc(y_true, y_score)
    assert 0.9 <= pr <= 1.0

def test_compute_f1():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.6, 0.8, 0.9])
    assert compute_f1(y_true, y_score, threshold=0.5) == pytest.approx(0.8, 0.01)

def test_compute_ece():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.1, 0.1])
    ece = compute_ece(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 0.2

def test_subgroup_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    subgroup_mask = np.array([False, False, True, True, False, False])
    sub_auc = compute_subgroup_auc(y_true, y_score, subgroup_mask)
    # y_true on subgroup is all 1s so auroc returns nan
    assert np.isnan(sub_auc)

def test_borkan_bias_metric():
    sub_aucs = {"group_a": 0.85, "group_b": 0.90}
    bpsn_aucs = {"group_a": 0.80, "group_b": 0.85}
    bnsp_aucs = {"group_a": 0.88, "group_b": 0.92}
    bias_score = compute_borkan_bias_metric(
        overall_auc=0.91,
        subgroup_aucs=sub_aucs,
        bpsn_aucs=bpsn_aucs,
        bnsp_aucs=bnsp_aucs,
    )
    assert 0.8 <= bias_score <= 0.95
