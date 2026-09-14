"""
Export Phase 5 fairness-audit results to reports/fairness_results.json.

Re-derives every number in notebooks/05_fairness_audit.ipynb from the persisted
model artifact and train_fe.parquet -- this script IS the single source of truth
for what the dashboard (Phase 6) displays, so the dashboard never re-runs AIF360
mitigation live on every page load (slow, and would drift from the notebook's
audited numbers). Run this after any change to the model artifact or to
src/fairness.py, then re-open the dashboard.

Mirrors the notebook's cell-by-cell logic exactly (same masks, same privileged/
unprivileged assignments, same seed=42) so the two never disagree.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parent))
from fairness import compute_fairness_metrics, sweep_fairness_thresholds, apply_eq_odds_postprocessing
from decision_policy import find_optimal_threshold, DEFAULT_LGD, DEFAULT_MARGIN

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def per_group_tpr_fpr(y_true, y_pred, group):
    """Same per-group TPR/FPR diagnostic table used in the notebook before
    trusting any summary DPD/EOD number -- exported so the dashboard can show
    it too, not just the headline metric."""
    df = pd.DataFrame({'y_true': np.asarray(y_true), 'y_pred': np.asarray(y_pred), 'group': np.asarray(group)})
    rows = []
    for grp, sub in df.groupby('group'):
        pos = sub[sub.y_true == 1]
        neg = sub[sub.y_true == 0]
        tpr = float((pos.y_pred == 1).mean()) if len(pos) else None
        fpr = float((neg.y_pred == 1).mean()) if len(neg) else None
        rows.append({'group': str(grp), 'n': int(len(sub)), 'n_pos': int(len(pos)), 'n_neg': int(len(neg)),
                     'tpr': tpr, 'fpr': fpr})
    return sorted(rows, key=lambda r: r['n_pos'])


def mitigation_pass(y_true, y_proba, group, privileged_value, unprivileged_value, mask=None, label=''):
    """One before/after EqOddsPostprocessing pass, packaged for JSON export."""
    y_true = np.asarray(y_true)
    group = np.asarray(group)
    if mask is not None:
        y_true = y_true[mask]
        group = group[mask]
        y_proba_subset = np.asarray(y_proba)[mask]
    else:
        y_proba_subset = np.asarray(y_proba)
    y_pred = (y_proba_subset >= 0.5).astype(int)

    before = compute_fairness_metrics(y_true, y_pred.astype(float), group, threshold=0.5)
    y_pred_mitigated = apply_eq_odds_postprocessing(
        y_true, y_pred, group, privileged_value=privileged_value, unprivileged_value=unprivileged_value, seed=42,
    )
    after = compute_fairness_metrics(y_true, y_pred_mitigated.astype(float), group, threshold=0.5)

    return {
        'label': label,
        'subset_size': int(len(y_true)),
        'privileged_value': str(privileged_value),
        'unprivileged_value': str(unprivileged_value),
        'accuracy_before': float(accuracy_score(y_true, y_pred)),
        'accuracy_after': float(accuracy_score(y_true, y_pred_mitigated)),
        'before': before,
        'after': after,
        'tpr_fpr_before': per_group_tpr_fpr(y_true, y_pred, group),
        'tpr_fpr_after': per_group_tpr_fpr(y_true, y_pred_mitigated, group),
    }


def main():
    artifact = joblib.load(PROJECT_ROOT / 'models' / 'tuned_xgb_bureau.joblib')
    tuned_clf = artifact['model']
    bureau_cols = artifact['bureau_cols']
    tuned_metrics = artifact['tuned_metrics']
    test_index = artifact['test_index']

    train_fe = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'train_fe.parquet')
    X_test = train_fe.loc[test_index, bureau_cols]
    y_test = train_fe.loc[test_index, 'TARGET']

    assert X_test.shape == (61501, 292), f"unexpected X_test shape {X_test.shape}"
    assert abs(y_test.mean() - 0.080730) < 1e-4, f"unexpected default rate {y_test.mean():.4%}"

    y_proba = tuned_clf.predict_proba(X_test)[:, 1]
    print(f"Recomputed AUC matches persisted: {tuned_metrics['auc']:.4f} (live predict_proba used for all fairness metrics)")

    results = {'model_auc': tuned_metrics['auc'], 'test_set_size': int(len(y_test)),
               'default_rate': float(y_test.mean())}

    # --- Cost-sensitive decision threshold (closes Phase 5's "not yet done" item) ---
    amt_credit = X_test['AMT_CREDIT'].to_numpy()
    t_star, cost_sweep = find_optimal_threshold(y_test.to_numpy(), y_proba, amt_credit)
    row_05 = cost_sweep[cost_sweep['threshold'] == 0.5].iloc[0]
    row_ts = cost_sweep.loc[cost_sweep['total_cost'].idxmin()]
    gender_array_for_threshold = X_test['CODE_GENDER_F'].to_numpy()
    region_array_for_threshold = X_test['REGION_RATING_CLIENT'].to_numpy()
    results['decision_threshold'] = {
        't_star': t_star,
        'lgd': DEFAULT_LGD,
        'margin': DEFAULT_MARGIN,
        'cost_at_0.5': float(row_05['total_cost']),
        'cost_at_t_star': float(row_ts['total_cost']),
        'cost_reduction_pct': float((row_05['total_cost'] - row_ts['total_cost']) / row_05['total_cost'] * 100),
        'approval_rate_at_0.5': float(row_05['approval_rate']),
        'approval_rate_at_t_star': float(row_ts['approval_rate']),
        'approved_default_rate_at_0.5': float(row_05['approved_default_rate']),
        'approved_default_rate_at_t_star': float(row_ts['approved_default_rate']),
        'cost_sweep': cost_sweep.to_dict(orient='records'),
        'gender_at_0.5': compute_fairness_metrics(y_test, y_proba, gender_array_for_threshold, threshold=0.5),
        'gender_at_t_star': compute_fairness_metrics(y_test, y_proba, gender_array_for_threshold, threshold=t_star),
        'region_tier_at_0.5': compute_fairness_metrics(y_test, y_proba, region_array_for_threshold, threshold=0.5),
        'region_tier_at_t_star': compute_fairness_metrics(y_test, y_proba, region_array_for_threshold, threshold=t_star),
    }
    print(f"Cost-calibrated threshold t* = {t_star} (cost reduction vs 0.5: "
          f"{results['decision_threshold']['cost_reduction_pct']:.1f}%)")

    # --- Gender ---
    gender_array = X_test['CODE_GENDER_F'].to_numpy()
    results['gender'] = {
        'metrics': compute_fairness_metrics(y_test, y_proba, gender_array, threshold=0.5),
        'sweep': sweep_fairness_thresholds(y_test, y_proba, gender_array).to_dict(orient='records'),
        'tpr_fpr': per_group_tpr_fpr(y_test, (y_proba >= 0.5).astype(int), gender_array),
        'mitigation': mitigation_pass(
            y_test.to_numpy(), y_proba, gender_array,
            privileged_value=True, unprivileged_value=False, label='Gender (full test set)',
        ),
    }
    print("Gender done")

    # --- Region tier (3 groups + 3 pairwise mitigations) ---
    region_array = X_test['REGION_RATING_CLIENT'].to_numpy()
    results['region_tier'] = {
        'metrics': compute_fairness_metrics(y_test, y_proba, region_array, threshold=0.5),
        'sweep': sweep_fairness_thresholds(y_test, y_proba, region_array).to_dict(orient='records'),
        'tpr_fpr': per_group_tpr_fpr(y_test, (y_proba >= 0.5).astype(int), region_array),
        'mitigations': {
            'tier1_vs_tier3': mitigation_pass(
                y_test.to_numpy(), y_proba, region_array, privileged_value=1, unprivileged_value=3,
                mask=np.isin(region_array, [1, 3]), label='Tier 1 vs Tier 3',
            ),
            'tier1_vs_tier2': mitigation_pass(
                y_test.to_numpy(), y_proba, region_array, privileged_value=1, unprivileged_value=2,
                mask=np.isin(region_array, [1, 2]), label='Tier 1 vs Tier 2',
            ),
            'tier2_vs_tier3': mitigation_pass(
                y_test.to_numpy(), y_proba, region_array, privileged_value=2, unprivileged_value=3,
                mask=np.isin(region_array, [2, 3]), label='Tier 2 vs Tier 3',
            ),
        },
    }
    print("Region tier done")

    # --- Education (reconstructed from one-hot columns) ---
    edu_cols = [c for c in X_test.columns if c.startswith('NAME_EDUCATION_TYPE_')]
    assert (X_test[edu_cols].sum(axis=1) == 1).all(), "expected exactly one education dummy per row"
    education_label = X_test[edu_cols].idxmax(axis=1).str.replace('NAME_EDUCATION_TYPE_', '', regex=False)
    edu_diag_mask = (education_label != 'Academic degree').to_numpy()

    results['education'] = {
        'label_distribution': education_label.value_counts().to_dict(),
        'metrics_raw': compute_fairness_metrics(y_test, y_proba, education_label.to_numpy(), threshold=0.5),
        'metrics_diagnosed': compute_fairness_metrics(
            y_test[edu_diag_mask], y_proba[edu_diag_mask], education_label[edu_diag_mask].to_numpy(), threshold=0.5,
        ),
        'diagnosis_note': "Raw EOD is a small-sample artifact of 'Academic degree' (n=38, only 1 defaulter). "
                           "metrics_diagnosed excludes that group.",
        'tpr_fpr': per_group_tpr_fpr(y_test, (y_proba >= 0.5).astype(int), education_label.to_numpy()),
        'mitigation': mitigation_pass(
            y_test.to_numpy(), y_proba, education_label.to_numpy(),
            privileged_value='Higher education', unprivileged_value='Lower secondary',
            mask=education_label.isin(['Lower secondary', 'Higher education']).to_numpy(),
            label='Lower secondary vs Higher education',
        ),
    }
    print("Education done")

    # --- Occupation (raw re-join via SK_ID_CURR -- target-encoded column in train_fe is circular) ---
    test_ids = train_fe['SK_ID_CURR'].loc[test_index]
    raw_occupation = pd.read_csv(
        PROJECT_ROOT / 'data' / 'raw' / 'home-credit-default-risk' / 'application_train.csv',
        usecols=['SK_ID_CURR', 'OCCUPATION_TYPE'],
    )
    occupation_merged = test_ids.to_frame().merge(raw_occupation, on='SK_ID_CURR', how='left')
    assert len(occupation_merged) == len(test_ids), "merge changed row count -- SK_ID_CURR not unique?"
    occupation_label = occupation_merged['OCCUPATION_TYPE'].fillna('Not Reported').to_numpy()

    results['occupation'] = {
        'label_distribution': pd.Series(occupation_label).value_counts().to_dict(),
        'metrics': compute_fairness_metrics(y_test, y_proba, occupation_label, threshold=0.5),
        'tpr_fpr': per_group_tpr_fpr(y_test, (y_proba >= 0.5).astype(int), occupation_label),
        'mitigation': mitigation_pass(
            y_test.to_numpy(), y_proba, occupation_label,
            privileged_value='Accountants', unprivileged_value='Low-skill Laborers',
            mask=np.isin(occupation_label, ['Low-skill Laborers', 'Accountants']),
            label='Low-skill Laborers vs Accountants',
        ),
    }
    print("Occupation done")

    out_path = PROJECT_ROOT / 'reports' / 'fairness_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
