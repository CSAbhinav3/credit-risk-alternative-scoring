"""
Phase 6 dashboard entry point. Run with:
    streamlit run dashboard/Home.py
from the project root (venv activated).

This page: project overview + headline model performance. Individual applicant
explanations live on the "Applicant Risk Explorer" page; fairness audit results
live on the "Fairness Audit" page (both auto-discovered by Streamlit from
dashboard/pages/).
"""
import sys
from pathlib import Path

import streamlit as st

# Streamlit's multi-page-app runner doesn't always add the main script's own
# directory to sys.path (confirmed via streamlit.testing.v1.AppTest -- it raised
# ModuleNotFoundError: No module named 'utils' without this), unlike a plain
# single-script `streamlit run`. Match what the pages/ files already do.
sys.path.append(str(Path(__file__).resolve().parent))
from utils import load_model_artifact, load_test_set, load_fairness_results

st.set_page_config(page_title="Alternative Credit Scoring", page_icon="\U0001F4CA", layout="wide")

st.title("Alternative Credit Scoring for India's Underbanked Population")
st.caption("Final-year B.Tech project — Christ University, Bengaluru. XGBoost + SHAP + Fairlearn/AIF360.")

st.markdown(
    """
    Traditional bureau scoring (CIBIL) excludes India's credit-invisible-but-active
    population. This project combines Home Credit bureau/application data with
    synthetic UPI-style behavioral data (calibrated to RBI DPSS statistics) to train
    a default-risk model, then audits that model's performance, explainability, and
    fairness jointly rather than in isolation.
    """
)

artifact = load_model_artifact()
X_test, y_test, _ = load_test_set()
fairness_results = load_fairness_results()

st.subheader("Model performance (held-out test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC", f"{artifact['tuned_metrics']['auc']:.4f}")
col2.metric("Gini", f"{artifact['tuned_metrics']['gini']:.4f}")
col3.metric("KS statistic", f"{artifact['tuned_metrics']['ks']:.4f}")
col4.metric("Test set size", f"{len(X_test):,}")

st.subheader("Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Applicants (test set)", f"{len(X_test):,}")
col2.metric("Default rate", f"{y_test.mean():.2%}")
col3.metric("Features (bureau + application)", f"{X_test.shape[1]}")

st.subheader("What's audited here")
st.markdown(
    """
    - **Applicant Risk Explorer** — pick any test-set applicant, see their predicted
      default-risk score and the SHAP factors that drove it.
    - **Fairness Audit** — demographic parity / equalized odds gaps across gender,
      region tier, education, and occupation, with AIF360 `EqOddsPostprocessing`
      before/after mitigation.
    """
)

if fairness_results is None:
    st.warning(
        "reports/fairness_results.json not found — run "
        "`python src/export_fairness_results.py` before opening the Fairness Audit page."
    )
else:
    st.caption(
        f"Fairness results last exported for model AUC {fairness_results['model_auc']:.4f} "
        f"on {fairness_results['test_set_size']:,} test rows."
    )
