"""
Displays the precomputed Phase 5 fairness-audit results (reports/fairness_results.json,
built by src/export_fairness_results.py). Never re-runs AIF360 mitigation live --
that JSON is the single source of truth, kept in sync with notebooks/05_fairness_audit.ipynb.

Follows the same "diagnose before trust" discipline as the notebook: every headline
DPD/EOD number is shown alongside its per-group n/n_pos/n_neg/TPR/FPR table, not on
its own.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_fairness_results

st.set_page_config(page_title="Fairness Audit", page_icon="⚖️", layout="wide")
st.title("Fairness Audit")

results = load_fairness_results()
if results is None:
    st.error(
        "reports/fairness_results.json not found. Run "
        "`./venv/Scripts/python.exe src/export_fairness_results.py` from the project root, then reload this page."
    )
    st.stop()


def render_tpr_fpr_table(rows):
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({'tpr': '{:.4f}', 'fpr': '{:.4f}'}, na_rep='—'), width='stretch')


def render_headline(metrics, label):
    c1, c2 = st.columns(2)
    c1.metric(f"{label} — demographic parity diff", f"{metrics['demographic_parity_difference']:.4f}")
    c2.metric(f"{label} — equalized odds diff", f"{metrics['equalized_odds_difference']:.4f}")


def render_mitigation(m):
    st.markdown(f"**{m['label']}** — subset size {m['subset_size']:,} "
                f"(privileged: {m['privileged_value']}, unprivileged: {m['unprivileged_value']})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Equalized odds diff", f"{m['before']['equalized_odds_difference']:.4f}",
              delta=f"{m['after']['equalized_odds_difference'] - m['before']['equalized_odds_difference']:.4f}",
              delta_color="inverse")
    c2.metric("Demographic parity diff", f"{m['before']['demographic_parity_difference']:.4f}",
              delta=f"{m['after']['demographic_parity_difference'] - m['before']['demographic_parity_difference']:.4f}",
              delta_color="inverse")
    c3.metric("Accuracy on subset", f"{m['accuracy_before']:.4f}",
              delta=f"{m['accuracy_after'] - m['accuracy_before']:.4f}")
    st.caption("Before → after EqOddsPostprocessing. Equalized-odds/parity deltas negative = improved; "
               "accuracy delta reflects the cost of flipping predictions without retraining.")
    with st.expander("Per-group TPR/FPR, before vs. after"):
        st.write("Before:")
        render_tpr_fpr_table(m['tpr_fpr_before'])
        st.write("After:")
        render_tpr_fpr_table(m['tpr_fpr_after'])


if 'decision_threshold' in results:
    dt = results['decision_threshold']
    st.subheader("Cost-calibrated decision threshold")
    st.caption(
        f"Every metric below is computed at threshold=0.5 by default. This project's actual "
        f"cost-minimizing threshold (Bahnsen et al. 2014 example-dependent cost-sensitive "
        f"framework, LGD={dt['lgd']}, profit margin={dt['margin']} of loan principal -- "
        f"illustrative assumptions, not fitted to real recovery data) is **t\\* = {dt['t_star']}**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total expected cost", f"₹{dt['cost_at_t_star']/1e7:,.1f} Cr",
              delta=f"-{dt['cost_reduction_pct']:.1f}% vs 0.5", delta_color="inverse")
    c2.metric("Approval rate", f"{dt['approval_rate_at_t_star']:.1%}",
              delta=f"{(dt['approval_rate_at_t_star'] - dt['approval_rate_at_0.5']) * 100:+.1f} pp")
    c3.metric("Default rate among approved", f"{dt['approved_default_rate_at_t_star']:.2%}",
              delta=f"{(dt['approved_default_rate_at_t_star'] - dt['approved_default_rate_at_0.5']) * 100:+.2f} pp",
              delta_color="inverse")
    with st.expander("Does the cost-optimal threshold change the fairness conclusions?"):
        st.markdown(
            "**Not uniformly.** Demographic parity improves for both gender and region tier at "
            "t\\*, and gender equalized odds improves slightly -- but **region-tier equalized "
            "odds gets worse** (the binding gap shifts from FPR to a wider TPR gap: Tier 1's "
            "true-positive rate drops much more steeply than Tier 3's as the threshold rises). "
            "The cost-optimal operating point and the fairest operating point are not the same "
            "threshold -- see `notebooks/05_fairness_audit.ipynb`'s threshold-calibration section "
            "for the full diagnosis."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Gender**")
            render_headline(dt['gender_at_0.5'], "@ 0.5")
            render_headline(dt['gender_at_t_star'], "@ t*")
        with c2:
            st.markdown("**Region tier**")
            render_headline(dt['region_tier_at_0.5'], "@ 0.5")
            render_headline(dt['region_tier_at_t_star'], "@ t*")
    st.divider()

attribute = st.selectbox("Sensitive attribute", ["Gender", "Region tier", "Education", "Occupation"])

if attribute == "Gender":
    g = results['gender']
    render_headline(g['metrics'], "Gender")
    with st.expander("Per-group TPR/FPR (diagnose before trusting the headline numbers)"):
        render_tpr_fpr_table(g['tpr_fpr'])

    sweep_df = pd.DataFrame(g['sweep'])
    fig = px.line(sweep_df, x='threshold', y=['demographic_parity_difference', 'equalized_odds_difference'],
                   markers=True, title='Fairness gap vs. decision threshold — Gender')
    st.plotly_chart(fig, width='stretch')

    st.subheader("AIF360 EqOddsPostprocessing mitigation")
    render_mitigation(g['mitigation'])

elif attribute == "Region tier":
    r = results['region_tier']
    render_headline(r['metrics'], "Region tier (1=best .. 3=worst)")
    with st.expander("Per-group TPR/FPR (diagnose before trusting the headline numbers)"):
        render_tpr_fpr_table(r['tpr_fpr'])

    sweep_df = pd.DataFrame(r['sweep'])
    fig = px.line(sweep_df, x='threshold', y=['demographic_parity_difference', 'equalized_odds_difference'],
                   markers=True, title='Fairness gap vs. decision threshold — Region tier')
    st.plotly_chart(fig, width='stretch')

    st.subheader("AIF360 EqOddsPostprocessing mitigation (pairwise — AIF360 is binary-group only)")
    pair = st.radio("Pair", ["Tier 1 vs Tier 3", "Tier 1 vs Tier 2", "Tier 2 vs Tier 3"], horizontal=True)
    key = {'Tier 1 vs Tier 3': 'tier1_vs_tier3', 'Tier 1 vs Tier 2': 'tier1_vs_tier2',
           'Tier 2 vs Tier 3': 'tier2_vs_tier3'}[pair]
    render_mitigation(r['mitigations'][key])

elif attribute == "Education":
    e = results['education']
    st.info(e['diagnosis_note'])
    st.markdown("**Raw** (all 5 groups, including the unreliable n=38 'Academic degree' group):")
    render_headline(e['metrics_raw'], "Education (raw)")
    st.markdown("**Diagnosed** ('Academic degree' excluded):")
    render_headline(e['metrics_diagnosed'], "Education (diagnosed)")
    with st.expander("Per-group TPR/FPR (diagnose before trusting the headline numbers)"):
        render_tpr_fpr_table(e['tpr_fpr'])
    st.caption(f"Label distribution: {e['label_distribution']}")

    st.subheader("AIF360 EqOddsPostprocessing mitigation — Lower secondary vs Higher education")
    render_mitigation(e['mitigation'])

else:  # Occupation
    o = results['occupation']
    render_headline(o['metrics'], "Occupation (19 groups)")
    with st.expander("Per-group TPR/FPR (diagnose before trusting the headline numbers — occupation groups vary a lot in size)"):
        render_tpr_fpr_table(o['tpr_fpr'])
    st.caption(f"Label distribution: {o['label_distribution']}")

    st.subheader("AIF360 EqOddsPostprocessing mitigation — Low-skill Laborers vs Accountants")
    st.caption("This is the smallest subset mitigated with the largest starting gap — mitigation here only "
               "partially closes the equalized-odds difference (more finite-sample noise in the randomized-flip "
               "realization), reported as-is rather than smoothed over.")
    render_mitigation(o['mitigation'])
