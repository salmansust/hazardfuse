import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, jaccard_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from hazardfuse.metrics import METRICS, block_bootstrap, compute_metrics, metrics_from_counts


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=4000) < 0.74).astype(int)
    p = np.clip(0.55 * y + rng.normal(0.25, 0.25, size=y.size), 0, 1)
    return y, p


def test_against_sklearn(data):
    y, p = data
    m, pred = compute_metrics(y, p), (p >= 0.5).astype(int)
    ref = {"Accuracy": accuracy_score(y, pred), "Balanced_Acc": balanced_accuracy_score(y, pred),
           "Precision_0": precision_score(y, pred, pos_label=0), "Recall_0": recall_score(y, pred, pos_label=0),
           "F1_0": f1_score(y, pred, pos_label=0), "Precision_1": precision_score(y, pred),
           "Recall_1": recall_score(y, pred), "F1_1": f1_score(y, pred), "Macro_F1": f1_score(y, pred, average="macro"),
           "Macro_Precision": precision_score(y, pred, average="macro"),
           "Macro_Recall": recall_score(y, pred, average="macro"), "Kappa": cohen_kappa_score(y, pred),
           "MCC": matthews_corrcoef(y, pred), "ROC_AUC": roc_auc_score(y, p), "IoU_0": jaccard_score(y, pred, pos_label=0),
           "IoU_1": jaccard_score(y, pred), "PR_AUC_1": average_precision_score(y, p),
           "PR_AUC_0": average_precision_score(1 - y, 1 - p)}
    for k, v in ref.items():
        assert m[k] == pytest.approx(v, abs=1e-12), k
    assert m["mIoU"] == pytest.approx((ref["IoU_0"] + ref["IoU_1"]) / 2)
    assert list(m)[:len(METRICS)] == METRICS


def test_counts_vectorised():
    c = np.array([[50, 5, 7, 100], [10, 0, 0, 20]], dtype=float)
    v = metrics_from_counts(*c.T)
    for i in range(2):
        s = metrics_from_counts(*c[i])
        assert all(np.isclose(v[k][i], s[k]) for k in s)


def test_one_class_and_no_predictions():
    m = compute_metrics(np.ones(10), np.zeros(10))
    assert np.isnan(m["ROC_AUC"]) and m["MCC"] == 0 and m["Precision_1"] == 0


def test_block_bootstrap(data):
    y, p = data
    blocks = np.repeat(np.arange(40), 100)
    bs = block_bootstrap(y, p, blocks, n_boot=300, n_boot_auc=30, seed=0)
    assert bs["n_blocks"] == 40 and set(bs["ci"]) == set(METRICS)
    for k, (v, lo, hi) in bs["ci"].items():
        assert lo <= hi, k
    assert bs["ci"]["MCC"][0] == pytest.approx(compute_metrics(y, p)["MCC"])
    again = block_bootstrap(y, p, blocks, n_boot=300, n_boot_auc=30, seed=0)
    assert again["ci"] == bs["ci"]                      # reproducible
