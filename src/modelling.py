import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate_classifier(y_true, y_proba, label=''):
    """
    Standard credit-scoring evaluation triple: AUC-ROC, Gini, and KS statistic.

    Gini = 2*AUC - 1 (the standard credit-scoring transform of AUC onto [0, 1]
    for a random classifier baseline of 0, rather than AUC's 0.5).

    KS statistic = max separation between the cumulative TPR and FPR curves
    across all thresholds -- the other standard credit-scoring metric alongside
    AUC/Gini, measuring how well the model's score separates the two classes
    at its single best cutoff.
    """
    auc = roc_auc_score(y_true, y_proba)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    if label:
        print(f"{label} -> AUC-ROC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks:.4f}")
    return {'auc': auc, 'gini': gini, 'ks': ks}
