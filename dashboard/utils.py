"""
Shared loaders for the Phase 6 dashboard pages. Every page imports from here
instead of re-deriving the test set / model / fairness results itself, so all
three pages stay consistent with each other and with the audited notebooks.

Cached with Streamlit's own cache decorators (not manual memoization) so the
model/data only load once per server process, not once per page view.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))


@st.cache_resource
def load_model_artifact():
    """The full persisted dict: model, best_params, bureau_cols, upi_cols,
    feature_cols, tuned_metrics, test_index. cache_resource (not cache_data)
    because the XGBClassifier inside isn't reliably picklable/hashable."""
    return joblib.load(PROJECT_ROOT / 'models' / 'tuned_xgb_bureau.joblib')


@st.cache_data
def load_test_set():
    """Reconstructs X_test/y_test exactly as every Phase 3-5 notebook does:
    train_fe.loc[test_index, bureau_cols]. Also returns SK_ID_CURR for the
    same rows so the Applicant Explorer page can look applicants up by ID."""
    artifact = load_model_artifact()
    train_fe = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'train_fe.parquet')
    test_index = artifact['test_index']
    X_test = train_fe.loc[test_index, artifact['bureau_cols']]
    y_test = train_fe.loc[test_index, 'TARGET']
    sk_ids = train_fe.loc[test_index, 'SK_ID_CURR']
    return X_test, y_test, sk_ids


@st.cache_data
def load_predictions():
    """y_proba for the full test set -- same call every notebook uses
    (tuned_clf.predict_proba(X_test)[:, 1]), cached so every page that needs
    risk scores doesn't re-run inference."""
    artifact = load_model_artifact()
    X_test, _, _ = load_test_set()
    return artifact['model'].predict_proba(X_test)[:, 1]


@st.cache_data
def load_fairness_results():
    """Precomputed by src/export_fairness_results.py -- the dashboard never
    re-runs AIF360 mitigation live (slow, ~19 groups x several passes)."""
    path = PROJECT_ROOT / 'reports' / 'fairness_results.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def get_shap_explainer():
    """Same shap.TreeExplainer(tuned_clf) call as notebooks/04_shap_explainability.ipynb
    Cell 3 -- cheap to construct, so built once and reused for every single-applicant
    explanation instead of the notebook's one-off 5000-row sample."""
    import shap
    artifact = load_model_artifact()
    return shap.TreeExplainer(artifact['model'])
