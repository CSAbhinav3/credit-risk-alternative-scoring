"""
Global SHAP explainability -- the model-wide view, complementing the
Applicant Risk Explorer's per-applicant view. Displays the pre-rendered PNGs
from notebooks/04_shap_explainability.ipynb (global importance, beeswarm, and
10 waterfall examples spanning the predicted-risk range) rather than
recomputing SHAP on a fresh 5000-row sample -- that computation exists to
produce these specific audited figures, not to be redone live on every page
load.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import PROJECT_ROOT

st.set_page_config(page_title="Model Explainability", page_icon="\U0001F4C8", layout="wide")
st.title("Model Explainability")
st.caption(
    "From notebooks/04_shap_explainability.ipynb -- SHAP computed on a 5,000-row "
    "stratified sample of the test set (same default rate as the full test set)."
)

figures_dir = PROJECT_ROOT / 'figures'

st.subheader("Global feature importance")
st.markdown("Mean |SHAP value| across the sample -- which features move the model's "
            "predictions the most, on average, regardless of direction.")
st.image(str(figures_dir / 'shap_global_importance.png'), width='stretch')

st.subheader("Feature impact distribution (beeswarm)")
st.markdown("Same top features, but showing the direction and spread of each feature's "
            "impact per applicant, not just its average magnitude.")
st.image(str(figures_dir / 'shap_beeswarm.png'), width='stretch')

st.divider()

st.subheader("Individual applicant waterfalls (10 examples spanning the risk range)")
st.markdown(
    "These 10 applicants were deliberately chosen to span the predicted-risk range of "
    "the sample: 3 lowest-risk, 4 borderline (around the sample median), 3 highest-risk. "
    "For any *other* applicant, use the **Applicant Risk Explorer** page instead -- it "
    "computes the same kind of explanation live for whichever applicant you pick."
)

waterfall_groups = {
    "Low risk (1-3)": [1, 2, 3],
    "Borderline (4-7)": [4, 5, 6, 7],
    "High risk (8-10)": [8, 9, 10],
}
group_label = st.radio("Group", list(waterfall_groups.keys()), horizontal=True)
selected_num = st.select_slider("Example", options=waterfall_groups[group_label])

waterfall_path = figures_dir / f'shap_waterfall_{selected_num}.png'
if waterfall_path.exists():
    st.image(str(waterfall_path), width='stretch')
else:
    st.warning(f"{waterfall_path.name} not found in figures/.")
