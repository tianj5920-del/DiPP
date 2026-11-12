import numpy as np
from scipy.stats import binom
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)
from . import CLASSES, PAIRS


def normalize(p):
    p = np.asarray(p, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError("Expected finite nonnegative N x 3 probabilities")
    p = np.maximum(p, 1e-12)
    return p / p.sum(1, keepdims=True)


def complete_metrics(y, probability):
    y = np.asarray(y)
    if set(y) != {0, 1, 2}:
        raise ValueError("Three-class evaluation needs every class")
    p = normalize(probability)
    if len(y) != len(p):
        raise ValueError("Label/prediction row count mismatch")
    predicted = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, predicted, labels=[0, 1, 2])
    hot = np.eye(3)[y]
    auc = np.array([roc_auc_score(hot[:, c], p[:, c]) for c in range(3)])
    ap = np.array([average_precision_score(hot[:, c], p[:, c]) for c in range(3)])
    accuracy = float(accuracy_score(y, predicted))
    result = dict(macro_precision=float(precision.mean()), macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                  macro_auc_ovr=float(auc.mean()), macro_auc_ovo=float(roc_auc_score(y, p, labels=[0, 1, 2], average="macro", multi_class="ovo")),
                  accuracy=accuracy, balanced_accuracy=float(recall.mean()), micro_precision=accuracy, micro_recall=accuracy,
                  macro_ap=float(ap.mean()), multiclass_brier=float(np.square(p - hot).sum(1).mean()), log_loss=float(log_loss(y, p, labels=[0, 1, 2])))
    for c, name in enumerate(CLASSES):
        tn = len(y) - cm[c].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        result.update({f"{name}_{key}": float(value) for key, value in
                       dict(precision=precision[c], recall=recall[c], f1=f1[c], auc_ovr=auc[c], ap=ap[c], specificity=tn / (tn + fp)).items()})
    for a, b in PAIRS:
        keep = (y == a) | (y == b)
        score = np.log(p[keep, a]) - np.log(p[keep, b])
        result[f"pair_{CLASSES[a]}_vs_{CLASSES[b]}_auc_logratio"] = float(roc_auc_score(y[keep] == a, score))
    return result


def random_reference(y):
    """Analytical expectation: constant probabilities, independently uniform labels."""
    y = np.asarray(y)
    n = len(y)
    if set(y) != {0, 1, 2}:
        raise ValueError("Random reference requires all classes")
    m = np.arange(n + 1)
    mass = binom.pmf(m, n, 1 / 3)
    result = dict(accuracy=1/3, balanced_accuracy=1/3, micro_precision=1/3, micro_recall=1/3,
                  macro_auc_ovr=.5, macro_auc_ovo=.5, macro_ap=1/3, multiclass_brier=2/3, log_loss=float(np.log(3)))
    for c, name in enumerate(CLASSES):
        nc = int((y == c).sum())
        values = dict(precision=nc/n*(1-mass[0]), recall=1/3, f1=float(np.sum(mass*2*m*nc/n/(nc+m))),
                      auc_ovr=.5, ap=nc/n, specificity=2/3)
        result.update({f"{name}_{k}": float(v) for k, v in values.items()})
    for key in ("precision", "recall", "f1"):
        result["macro_"+key] = float(np.mean([result[f"{name}_{key}"] for name in CLASSES]))
    for a, b in PAIRS:
        result[f"pair_{CLASSES[a]}_vs_{CLASSES[b]}_auc_logratio"] = .5
    return result
