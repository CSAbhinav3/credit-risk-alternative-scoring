import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing


def compute_fairness_metrics(y_true, y_proba, sensitive_feature, threshold=0.5, label=''):
    """
    Group-disaggregated fairness metrics for a binary risk score, sliced by a
    sensitive feature (e.g. CODE_GENDER_F).

    `selection_rate` here means "flagged as high-risk" (y_proba >= threshold),
    not "approved" -- the two are inverses in a default-risk model, so a positive
    demographic_parity_difference means one group is flagged as high-risk *more*
    often than the other, not approved more often.

    Demographic parity difference = max group selection rate - min group selection
    rate, the standard Fairlearn measure of whether groups are flagged at similar
    rates regardless of true outcome.

    Equalized odds difference = the larger of the max group gaps in TPR and FPR,
    the standard Fairlearn measure of whether groups are flagged at similar rates
    *conditional on* their true outcome (i.e. errors are evenly distributed).

    Both are computed on hard predictions at `threshold`, not on raw probabilities
    -- Fairlearn's difference metrics operate on the decision, not the score.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    # Two separate MetricFrames, not one dict-of-metrics MetricFrame: selection_rate
    # needs the thresholded y_pred, AUC needs the raw y_proba, and MetricFrame does
    # not support a different y_pred array per metric within a single instance.
    sr_frame = MetricFrame(
        metrics=selection_rate, y_true=y_true, y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )
    auc_frame = MetricFrame(
        metrics=roc_auc_score, y_true=y_true, y_pred=y_proba,
        sensitive_features=sensitive_feature,
    )
    group_selection_rate = sr_frame.by_group
    group_auc = auc_frame.by_group

    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)

    if label:
        print(f"{label} -> demographic parity diff: {dpd:.4f} | equalized odds diff: {eod:.4f}")
        for group in group_selection_rate.index:
            print(f"    group={group}: selection_rate={group_selection_rate[group]:.4f} | "
                  f"AUC={group_auc[group]:.4f}")

    return {
        'demographic_parity_difference': dpd,
        'equalized_odds_difference': eod,
        'group_selection_rate': group_selection_rate.to_dict(),
        'group_auc': group_auc.to_dict(),
    }


def sweep_fairness_thresholds(y_true, y_proba, sensitive_feature, thresholds=None, label=''):
    """
    Re-run compute_fairness_metrics() across a range of decision thresholds, to check
    whether a disparity found at one threshold (typically 0.5) holds across the ROC
    curve or is an artifact of that one cutoff.

    y_true class counts per group don't depend on `threshold` -- only which predictions
    count as positive does -- so equalized-odds TPR/FPR denominators stay well-defined
    at every threshold in (0, 1); no divide-by-zero risk from extreme cutoffs.

    Returns a DataFrame (one row per threshold) rather than a dict-of-dicts, since the
    sweep's whole purpose is to be plotted/scanned as a trend, not looked up by key.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)

    rows = []
    for t in thresholds:
        result = compute_fairness_metrics(y_true, y_proba, sensitive_feature, threshold=t)
        rows.append({
            'threshold': t,
            'demographic_parity_difference': result['demographic_parity_difference'],
            'equalized_odds_difference': result['equalized_odds_difference'],
        })
    sweep_df = pd.DataFrame(rows)

    if label:
        print(f"{label} -- threshold sweep:")
        print(sweep_df.to_string(index=False))

    return sweep_df


def apply_eq_odds_postprocessing(y_true, y_pred, sensitive_feature, privileged_value,
                                  unprivileged_value, seed=42):
    """
    AIF360 EqOddsPostprocessing: adjusts hard predictions (via randomized flipping,
    not retraining) so TPR and FPR match between exactly two groups.

    AIF360's postprocessing algorithms are binary-group only -- `sensitive_feature`
    must be restricted to rows belonging to `privileged_value` or `unprivileged_value`
    before calling this (e.g. REGION_RATING_CLIENT's Tier 1 vs Tier 3, dropping Tier 2).
    Not a limitation of this wrapper; a property of the algorithm itself.

    favorable_label=0 / unfavorable_label=1: "favorable" here means "not flagged
    high-risk" -- consistent with compute_fairness_metrics' selection_rate framing
    (flagged-high-risk = the outcome being disparately distributed), not "approved"
    in some separate loan-decision sense.

    Note: equalizing TPR/FPR does not always mean lowering the worse-off group's
    rate -- the algorithm's LP can equalize by raising the better-off group's rate
    to match instead, whichever satisfies the odds constraint. Confirmed via a
    synthetic scratch check before use here; always inspect the actual before/after
    per-group rates rather than assuming the "fix" moved rates in the expected
    direction.

    Returns the mitigated hard-prediction array (same length/order as the input,
    restricted to the two-group subset the caller passed in).
    """
    sensitive_feature = np.asarray(sensitive_feature)
    mask = np.isin(sensitive_feature, [privileged_value, unprivileged_value])
    if not mask.all():
        raise ValueError(
            "sensitive_feature contains values other than privileged_value/"
            "unprivileged_value -- restrict to the two-group subset before calling "
            "(AIF360 postprocessing is binary-group only)."
        )

    is_privileged = (sensitive_feature == privileged_value).astype(int)
    df_true = pd.DataFrame({'label': np.asarray(y_true), 'group': is_privileged})
    df_pred = pd.DataFrame({'label': np.asarray(y_pred), 'group': is_privileged})

    bld_true = BinaryLabelDataset(
        df=df_true, label_names=['label'], protected_attribute_names=['group'],
        favorable_label=0, unfavorable_label=1,
    )
    bld_pred = BinaryLabelDataset(
        df=df_pred, label_names=['label'], protected_attribute_names=['group'],
        favorable_label=0, unfavorable_label=1,
    )

    eq_odds = EqOddsPostprocessing(
        unprivileged_groups=[{'group': 0}], privileged_groups=[{'group': 1}], seed=seed,
    )
    eq_odds.fit(bld_true, bld_pred)
    bld_mitigated = eq_odds.predict(bld_pred)

    return bld_mitigated.labels.ravel().astype(int)
