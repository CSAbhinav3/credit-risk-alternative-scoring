"""
Pick a test-set applicant, see their predicted default-risk score and the SHAP
factors that drove it. Live per-applicant SHAP (shap.TreeExplainer on a single
row is fast, unlike the notebook's one-off 5000-row summary sample) -- this is
the one page that computes something on the fly rather than reading a
precomputed artifact, since it must respond to whichever applicant is picked.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_model_artifact, load_test_set, load_predictions, get_shap_explainer, load_fairness_results

st.set_page_config(page_title="Applicant Risk Explorer", page_icon="\U0001F50D", layout="wide")
st.title("Applicant Risk Explorer")

artifact = load_model_artifact()
X_test, y_test, sk_ids = load_test_set()
y_proba = load_predictions()

col1, col2 = st.columns([1, 2])
with col1:
    pick_mode = st.radio("Pick applicant by", ["Row position", "SK_ID_CURR"], horizontal=True)
    if pick_mode == "Row position":
        row_pos = st.number_input("Row position in test set", min_value=0, max_value=len(X_test) - 1, value=0, step=1)
    else:
        target_id = st.number_input("SK_ID_CURR", min_value=int(sk_ids.min()), max_value=int(sk_ids.max()),
                                     value=int(sk_ids.iloc[0]), step=1)
        matches = np.where(sk_ids.to_numpy() == target_id)[0]
        if len(matches) == 0:
            st.error(f"SK_ID_CURR {target_id} is not in the test set (61,501 applicants). Try a different ID.")
            st.stop()
        row_pos = int(matches[0])

applicant_row = X_test.iloc[[row_pos]]
applicant_id = int(sk_ids.iloc[row_pos])
true_label = int(y_test.iloc[row_pos])
risk_score = float(y_proba[row_pos])

with col2:
    st.metric("SK_ID_CURR", applicant_id)
    m1, m2 = st.columns(2)
    m1.metric("Predicted default risk", f"{risk_score:.1%}")
    m2.metric("Actual outcome (held-out label)", "Defaulted" if true_label == 1 else "Repaid")

st.divider()

fairness_results = load_fairness_results()
default_threshold = 0.5
threshold_note = "arbitrary 0.5"
if fairness_results and 'decision_threshold' in fairness_results:
    default_threshold = fairness_results['decision_threshold']['t_star']
    threshold_note = f"cost-calibrated t* = {default_threshold} (see Fairness Audit page)"

threshold = st.slider("Decision threshold (flag as high-risk if score >= this)", 0.0, 1.0, default_threshold, 0.01)
st.caption(f"Defaults to this project's {threshold_note}, not an arbitrary cutoff -- drag to explore other thresholds.")
flagged = risk_score >= threshold
st.markdown(f"At this threshold, this applicant would be **{'flagged as high-risk' if flagged else 'not flagged'}**.")

st.subheader("Why the model scored this applicant this way")
with st.spinner("Computing SHAP values..."):
    explainer = get_shap_explainer()
    explanation = explainer(applicant_row)

shap_vals = explanation.values[0]
base_value = explanation.base_values[0]
contrib = pd.DataFrame({
    'feature': applicant_row.columns,
    # applicant_row spans mixed dtypes (bool one-hot flags + float/int amounts), so
    # .to_numpy() on a single row yields an object array PyArrow can't serialize for
    # st.dataframe (confirmed via AppTest: "Could not convert np.True_ ... to double").
    # Stringify for display only -- the numeric 'shap_value' column stays a real float.
    'value': applicant_row.iloc[0].to_numpy().astype(str),
    'shap_value': shap_vals,
}).sort_values('shap_value', key=np.abs, ascending=False).head(15)
contrib['direction'] = np.where(contrib['shap_value'] >= 0, 'Pushes risk up', 'Pushes risk down')

st.caption(f"Base rate (average model output before this applicant's features): {base_value:.3f}")

import plotly.express as px
fig = px.bar(
    contrib.sort_values('shap_value'),
    x='shap_value', y='feature', color='direction', orientation='h',
    color_discrete_map={'Pushes risk up': '#d62728', 'Pushes risk down': '#2ca02c'},
    labels={'shap_value': 'SHAP value (impact on predicted risk)', 'feature': ''},
    title='Top 15 features driving this applicant\'s risk score',
)
fig.update_layout(showlegend=True, height=500)
st.plotly_chart(fig, width='stretch')

with st.expander("Raw feature values for the top contributors"):
    st.dataframe(contrib[['feature', 'value', 'shap_value']].reset_index(drop=True), width='stretch')

st.divider()
st.subheader("Export this explanation")
st.caption("For a report appendix or offline review -- one CSV per applicant, not a bulk export.")

import io

export_buffer = io.StringIO()
export_buffer.write(f"# Applicant risk explanation -- SK_ID_CURR {applicant_id}\n")
export_buffer.write(f"# Predicted default risk: {risk_score:.4f}\n")
export_buffer.write(f"# Actual outcome (held-out label): {'Defaulted' if true_label == 1 else 'Repaid'}\n")
export_buffer.write(f"# Decision threshold used: {threshold:.2f} -> {'flagged high-risk' if flagged else 'not flagged'}\n")
export_buffer.write(f"# Base rate (average model output before this applicant's features): {base_value:.4f}\n")
export_buffer.write("#\n")
contrib[['feature', 'value', 'shap_value', 'direction']].reset_index(drop=True).to_csv(export_buffer, index=False)

st.download_button(
    "Download explanation as CSV",
    data=export_buffer.getvalue(),
    file_name=f"applicant_{applicant_id}_explanation.csv",
    mime="text/csv",
)
