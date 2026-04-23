"""Binary metrics helper."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix,
)


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)
    return metrics
