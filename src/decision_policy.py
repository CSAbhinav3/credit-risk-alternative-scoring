"""
Example-dependent cost-sensitive decision threshold, closing the "tying the
threshold choice to an actual lending-decision rule" gap left open at the end
of notebooks/05_fairness_audit.ipynb (every fairness pass up to now used an
arbitrary 0.5 cutoff).

Follows the Bahnsen, Aouada & Ottersten (2014) "Example-Dependent Cost-Sensitive
Credit Scoring" framework: cost scales with each applicant's loan amount
(AMT_CREDIT), not a flat cost per error -- misclassifying a ₹5,00,000 loan
costs more than misclassifying a ₹50,000 one.

Cost assumptions (illustrative, not fitted to real recovery/margin data --
same "calibrated, not learned" caveat already used for Track B's UPI amounts
in feature_engineering.py):
- LGD (loss given default) = 0.55 of the loan's principal -- mid-range of the
  45-65% typically cited for unsecured retail/microfinance lending in India.
- Profit margin on a performing loan = 0.15 of principal -- mid-range of the
  10-20% net interest margins typical of Indian NBFC/microfinance lenders.

Convention: approve = model predicts NOT high-risk (y_pred == 0); reject =
y_pred == 1. Correctly rejecting an actual defaulter (TP) and correctly
approving an actual good applicant (TN) are both treated as zero marginal
cost -- the baseline. Only the two error types carry a cost:
  - False negative (approved a defaulter): cost = LGD * AMT_CREDIT
  - False positive (rejected a good applicant): cost = margin * AMT_CREDIT
"""
import numpy as np
import pandas as pd

DEFAULT_LGD = 0.55
DEFAULT_MARGIN = 0.15


def expected_cost_by_threshold(y_true, y_proba, amt_credit, thresholds=None,
                                lgd=DEFAULT_LGD, margin=DEFAULT_MARGIN):
    """
    Total expected cost (same currency unit as amt_credit) at each threshold,
    plus approval rate and default-rate-among-approved -- the operational
    numbers a lender would actually look at alongside the cost figure.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    amt_credit = np.asarray(amt_credit)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn_mask = (y_pred == 0) & (y_true == 1)   # approved a defaulter
        fp_mask = (y_pred == 1) & (y_true == 0)   # rejected a good applicant
        fn_cost = float((lgd * amt_credit[fn_mask]).sum())
        fp_cost = float((margin * amt_credit[fp_mask]).sum())
        approved_mask = (y_pred == 0)
        rows.append({
            'threshold': float(t),
            'fn_cost': fn_cost, 'fp_cost': fp_cost, 'total_cost': fn_cost + fp_cost,
            'n_fn': int(fn_mask.sum()), 'n_fp': int(fp_mask.sum()),
            'approval_rate': float(approved_mask.mean()),
            'approved_default_rate': float(y_true[approved_mask].mean()) if approved_mask.any() else float('nan'),
        })
    return pd.DataFrame(rows)


def find_optimal_threshold(y_true, y_proba, amt_credit, thresholds=None,
                            lgd=DEFAULT_LGD, margin=DEFAULT_MARGIN):
    """Threshold minimizing total expected cost -- the calibrated operating point."""
    sweep = expected_cost_by_threshold(y_true, y_proba, amt_credit, thresholds, lgd, margin)
    best_row = sweep.loc[sweep['total_cost'].idxmin()]
    return float(best_row['threshold']), sweep
