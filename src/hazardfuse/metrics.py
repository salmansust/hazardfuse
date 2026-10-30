"""Classification metrics (both classes, macro averages, agreement, ranking) and block-bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# every table reports these, in this order
METRICS = ["Accuracy", "Balanced_Acc", "Precision_0", "Recall_0", "F1_0", "Precision_1", "Recall_1", "F1_1",
           "Macro_Precision", "Macro_Recall", "Macro_F1", "Kappa", "MCC", "ROC_AUC", "IoU_0", "IoU_1", "mIoU",
           "PR_AUC_0", "PR_AUC_1"]
RANK_METRICS = ("ROC_AUC", "PR_AUC_0", "PR_AUC_1")          # threshold-free
COUNT_METRICS = [m for m in METRICS if m not in RANK_METRICS]


def metrics_from_counts(tn, fp, fn, tp) -> dict:
    """Threshold metrics from confusion counts; vectorised (arrays of counts give arrays of metrics)."""
    tn, fp, fn, tp = (np.asarray(v, dtype=float) for v in (tn, fp, fn, tp))
    n = tn + fp + fn + tp
    with np.errstate(divide="ignore", invalid="ignore"):
        rec1, rec0 = tp / (tp + fn), tn / (tn + fp)
        pre1, pre0 = tp / (tp + fp), tn / (tn + fn)
        f1_1, f1_0 = 2 * tp / (2 * tp + fp + fn), 2 * tn / (2 * tn + fn + fp)
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        iou1, iou0 = tp / (tp + fp + fn), tn / (tn + fn + fp)
        kappa = (po - pe) / (1 - pe)
    z = lambda a: np.nan_to_num(a, nan=0.0)                     # undefined precision/F1 (no predictions) -> 0
    return {"Accuracy": po, "Balanced_Acc": (rec0 + rec1) / 2,
            "Precision_0": z(pre0), "Recall_0": rec0, "F1_0": z(f1_0),
            "Precision_1": z(pre1), "Recall_1": rec1, "F1_1": z(f1_1),
            "Macro_Precision": (z(pre0) + z(pre1)) / 2, "Macro_Recall": (rec0 + rec1) / 2,
            "Macro_F1": (z(f1_0) + z(f1_1)) / 2, "Kappa": kappa, "MCC": z(mcc),
            "IoU_0": iou0, "IoU_1": iou1, "mIoU": (iou0 + iou1) / 2}


def kappa_se(tn, fp, fn, tp) -> float:
    """Large-sample standard error of Cohen's kappa (Fleiss, Cohen and Everitt, 1969)."""
    p = np.array([[tn, fp], [fn, tp]], dtype=float)
    n = p.sum()
    if n == 0:
        return np.nan
    p /= n
    row, col = p.sum(1), p.sum(0)
    po, pe = np.trace(p), float(row @ col)
    if np.isclose(pe, 1.0):
        return np.nan
    k = (po - pe) / (1 - pe)
    off = p * (1 - np.eye(2))
    var = (np.sum(np.diag(p) * (1 - (row + col) * (1 - k)) ** 2) + (1 - k) ** 2 * np.sum(off * (col[None] + row[:, None]) ** 2)
           - (k - pe * (1 - k)) ** 2) / (n * (1 - pe) ** 2)
    return float(np.sqrt(var)) if var > 0 else 0.0


def rank_metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {k: np.nan for k in RANK_METRICS}
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC_0": average_precision_score(1 - y, 1 - p),
            "PR_AUC_1": average_precision_score(y, p)}


def compute_metrics(y, prob, threshold=0.5) -> dict:
    """Every metric in METRICS, plus Kappa_SE, Brier score, log loss and the confusion counts."""
    y, p = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    pred = p >= threshold
    tn, fp, fn, tp = (int(((y == a) & (pred == b)).sum()) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    m = {k: float(v) for k, v in metrics_from_counts(tn, fp, fn, tp).items()}
    m.update(rank_metrics(y, p))
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    extra = {"Kappa_SE": kappa_se(tn, fp, fn, tp), "Brier": float(np.mean((np.clip(p, 0, 1) - y) ** 2)),
             "Log_Loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
             "TN": tn, "FP": fp, "FN": fn, "TP": tp, "N": len(y)}
    return {**{k: m[k] for k in METRICS}, **extra}


def block_counts(y, pred, blocks):
    """Confusion counts (TN, FP, FN, TP) per block: (unique blocks, (blocks, 4))."""
    ub, inv = np.unique(blocks, return_inverse=True)
    out = np.zeros((len(ub), 4))
    for j, cond in enumerate([(y == 0) & ~pred, (y == 0) & pred, (y == 1) & ~pred, (y == 1) & pred]):
        out[:, j] = np.bincount(inv, weights=cond.astype(float), minlength=len(ub))
    return ub, out


def block_bootstrap(y, prob, blocks, threshold=0.5, n_boot=2000, n_boot_auc=200, seed=42) -> dict:
    """
    Block bootstrap: blocks are resampled with replacement and the metrics recomputed on the pooled pixels of the drawn
    blocks. Count metrics use all `n_boot` resamples (from block confusion counts); ROC/PR-AUC use the first
    `n_boot_auc` resamples (recomputed from pixels). Returns {metric: (value, 2.5th, 97.5th percentile)} and the draws.
    """
    y, prob, blocks = np.asarray(y).astype(int), np.asarray(prob, dtype=float), np.asarray(blocks)
    ub, cnt = block_counts(y, prob >= threshold, blocks)
    draws = np.random.default_rng(seed).integers(0, len(ub), size=(n_boot, len(ub)))
    boot = metrics_from_counts(*cnt[draws].sum(axis=1).T)
    full = metrics_from_counts(*cnt.sum(axis=0))
    inv = np.unique(blocks, return_inverse=True)[1]
    pix = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    aa = [rank_metrics(y[ix], prob[ix]) for ix in (np.concatenate([pix[j] for j in d]) for d in draws[:n_boot_auc])]
    boot.update({k: np.array([a[k] for a in aa]) for k in RANK_METRICS})
    full.update(rank_metrics(y, prob))
    out = {k: (float(full[k]), *map(float, np.nanpercentile(boot[k], [2.5, 97.5]))) for k in METRICS}
    return {"ci": out, "boot": boot, "n_blocks": len(ub)}
