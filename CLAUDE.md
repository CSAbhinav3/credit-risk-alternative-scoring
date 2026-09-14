# Project: Alternative Credit Scoring for India's Underbanked Population

Final year B.Tech CSE research project. Christ University, Bengaluru. Submission target: Sept 20, 2026.

## Who's working on this
- Abhinav, Register No. 2360352, Section 7BTCS B, batch 2023–27
- Guide: Dr. Karthikeyan H
- Course code: CSE 784 / CSEIOT 784

## Working style — read before doing anything
- **One cell at a time.** Build/verify a single cell with real output confirmed before moving to the next. Do not hand over a complete notebook or large code block unless explicitly asked.
- **Pre-checks before any new aggregation/merge function**: ID overlap, cardinality, missingness, sort order. Always propose this check before writing the function.
- **Scratch cell → validate → commit**: new functions/fixes get written and verified in a scratch cell first, then wired into `feature_engineering.py` / Cell 2.
- **After any change to `feature_engineering.py` or Cell 2: Restart Kernel → Run All Cells.** Never re-run an individual cell in isolation — this is the exact cause of three separate `_x`/`_y` duplicate-merge bugs already hit in this project (stale cell re-merging onto an already-merged `train_fe`).
- **Cell 2 is the single source of truth** for the entire feature engineering pipeline. No aggregation/merge logic lives anywhere else.
- When debugging: ask for the actual error output first. Check for the duplicate/stale-merge-cell pattern before assuming a logic bug.
- Be direct and technically precise. Correct shape/number/cell-reference mismatches plainly, don't soften it.
- `fillna(0)` and similar decisions must be verified against the data (missingness pattern, structural-zero check), never assumed.

## Environment
- Windows, Python 3.13.5, VS Code + Jupyter notebooks
- Version control: **GitHub Desktop only** — no command-line git
- Venv: `venv\Scripts\activate` (activate every session)
- Working directory: `D:\Semester 7\PROJECT\credit-risk-alternative-scoring`
- GitHub repo: https://github.com/CSAbhinav3/credit-risk-alternative-scoring (public)
- Key libraries (no version pins): numpy, pandas, scikit-learn, xgboost, lightgbm, optuna, shap, fairlearn, aif360, imbalanced-learn, sdv, streamlit, matplotlib, seaborn, missingno, plotly, joblib, scipy, jupyter, notebook, ipykernel
- AIF360 + Fairlearn confirmed conflict-free in this environment

## Repo structure
```
credit-risk-alternative-scoring/
├── data/
│   ├── raw/home-credit-default-risk/   ← all 9 Home Credit CSVs (gitignored)
│   ├── processed/
│   │   └── train_fe.parquet             ← final engineered features (307505, 307), 120.8MB
│   └── synthetic/
│       └── upi_transactions.parquet     ← 18.46M synthetic UPI txn rows (Track B, see below)
├── notebooks/
│   ├── 01_eda.ipynb                     ← Phase 1: complete
│   ├── 02_feature_engineering.ipynb     ← Phase 2: Track A + Track B merge complete
│   ├── 03_synthetic_upi.ipynb           ← Phase 2: Track B generation (amount fits, count-mix, row gen, income corr.)
│   ├── 03_modelling.ipynb               ← Phase 3: complete (baselines, ablation, tuning, ensemble)
│   └── 04_shap_explainability.ipynb     ← Phase 4: complete (global/individual SHAP, H4 diagnosed)
├── models/
│   └── tuned_xgb_bureau.joblib          ← persisted Phase 3 final model + metadata, for Phase 4/5
├── src/
│   ├── feature_engineering.py           ← Track A pipeline functions
│   ├── synthetic_upi.py                 ← Track B synthetic UPI generators
│   ├── modelling.py                     ← Phase 3: evaluate_classifier (AUC/Gini/KS)
│   ├── fairness.py                      ← Phase 5: fairness metrics + AIF360 mitigation wrapper
│   ├── decision_policy.py               ← Phase 5: cost-sensitive threshold (t*=0.71)
│   └── export_fairness_results.py       ← Phase 6: exports reports/fairness_results.json for the dashboard
├── rbi_dpss_upi_stats.md                ← RBI DPSS/NPCI aggregate stats, Track B calibration source
├── dashboard/                            ← Phase 6: streamlit run dashboard/Home.py
│   ├── Home.py                           ← overview + model performance
│   ├── utils.py                          ← shared cached loaders
│   └── pages/
│       ├── 1_Applicant_Risk_Explorer.py  ← per-applicant risk score + live SHAP + CSV export
│       ├── 2_Fairness_Audit.py           ← per-attribute DPD/EOD + mitigation, reads fairness_results.json
│       └── 3_Model_Explainability.py     ← global SHAP (importance/beeswarm/10 waterfalls), reads figures/
├── figures/                              ← shap_global_importance.png, shap_beeswarm.png, shap_waterfall_{1..10}.png, cost_sensitive_threshold.png
├── reports/                               ← fairness_results.json (Phase 6 dashboard data)
├── .gitignore
├── requirements.txt
└── README.md
```

## Project thesis
Traditional bureau scoring (CIBIL) excludes India's credit-invisible-but-active population. The gap in existing literature (per Bahlool et al. 2026, JRFM) is that performance, fairness, and explainability are studied in isolation. This project's differentiator: combine all three — XGBoost + SHAP + Fairlearn/AIF360 — jointly, using Home Credit Default Risk data merged with synthetic UPI-style behavioral data calibrated to RBI DPSS statistics.

Full literature review, hypotheses, and framing live in the research proposal doc — not duplicated here.

## Pipeline status

| Phase | Status |
|---|---|
| 0 — Setup | ✅ Complete |
| 1 — EDA | ✅ Complete |
| 2 — Feature Engineering | ✅ Track A + Track B complete (UPI features merged into `train_fe`) |
| 3 — Modelling | ✅ Complete — baselines, ablation (H2 not supported), tuning, ensemble (H1 not met, best AUC 0.7914) |
| 4 — SHAP Explainability | ✅ Complete — global importance, 10 waterfall plots, plain-language template, H4 diagnosed & supported |
| 5 — Fairness Audit | ✅ Complete — 4 attributes audited + mitigated (AIF360); occupation mitigation incomplete (EOD only partially closes, diagnosed as small-sample/large-gap, reported as-is); cost-sensitive threshold calibrated (t\*=0.71, −22.2% cost vs 0.5) |
| 6 — Streamlit Dashboard | ✅ Built — 4 pages (overview, applicant explorer, fairness audit + cost-calibrated threshold, model explainability), verified via `streamlit.testing.v1.AppTest` |
| 7 — Thesis + Paper | Not started |

### ✅ Track A — fully closed
Row-drop fix, stale-cell cleanup, and feature selection are all resolved and verified end-to-end.

**Row-drop fix**: Six rows (`CODE_GENDER == 'XNA'`, 4 rows; `NAME_FAMILY_STATUS == 'Unknown'`, 2 rows) dropped live in Cell 4, immediately after `pd.read_csv` for `application_train.csv`, before `fix_days_employed()`. Two stale cells that were silently corrupting `train_fe` on re-run were also deleted (former Cell 11 — redundant `add_ext_source_missing_flags` call; former Cell 14 — redundant `transform_amounts` call that was overwriting `AMT_INCOME_TOTAL_RAW` with the already-capped value on second call). Notebook went 69 → 65 cells.

**Feature selection**: `drop_zero_variance_features(df)` added as the pipeline's 21st function, wired in as the last step in Cell 2, right before the final-shape print. Drops 10 columns total:
- 9 exact-zero-variance `_nan` dummy columns (`nunique(dropna=False) == 1`): `CODE_GENDER_nan`, `FLAG_OWN_CAR_nan`, `FLAG_OWN_REALTY_nan`, `NAME_CONTRACT_TYPE_nan`, `NAME_EDUCATION_TYPE_nan`, `NAME_FAMILY_STATUS_nan`, `NAME_HOUSING_TYPE_nan`, `NAME_INCOME_TYPE_nan`, `WEEKDAY_APPR_PROCESS_START_nan` — all bool dtype, sole value `False` across all 307,505 rows (their source columns have zero missingness, so `dummy_na=True` produced a dead indicator).
- `FLAG_MOBIL` — **near-constant, not exact-zero-variance** (1 applicant out of 307,505 has `FLAG_MOBIL == 0`, rest are 1; 99.999675% dominant). Included in the drop on the strength of 0.0 permutation importance from a baseline RF, not on the zero-variance criterion. (An earlier version of this file incorrectly grouped `FLAG_MOBIL` with the exact-zero-variance columns — corrected here.)

`FLAG_DOCUMENT_*` columns were explicitly and deliberately **not** touched — confirmed untouched in the scratch-cell validation before the drop was committed.

**Final verified `train_fe.shape`: `(307505, 294)`** — confirmed via Restart Kernel → Run All Cells, sequential execution 1→47, zero errors, zero leftover diagnostic cells. (This is Track A's own contribution shape, before Track B's UPI merge — see below for the current cumulative shape.)

### ✅ Track B — feature generation & merge complete
RBI stats gathered, both amount distributions fitted (with a real bug found and fixed), the
transaction count/split mix fitted and income-anchored, per-transaction row generation done and
saved, transaction count correlated with income (region tier deliberately excluded — see below),
and the resulting features merged onto `train_fe`. Not yet done: documenting the
calibrated-not-learned limitation in the thesis methodology (a writing task, tracked under Phase 7,
not a pipeline gap).

**RBI stats**: `rbi_dpss_upi_stats.md` — P2M ATS=₹659, 86th percentile=₹500; P2P ATS=₹2,812, capped
at ₹1,00,000/txn (confirmed unchanged); P2M/P2P volume split ~63–64%/36–37% (current era); P2M
general ceiling ambiguous (₹1–5 lakh depending on category, or ₹10 lakh/**day** for select verified
merchants — a daily aggregate, not comparable to a per-transaction limit).

**P2M amount fit**: exact closed-form solve, both constraints real (ATS + p86) →
`mu=3.630972, sigma=2.391548` untruncated. Implied median ~₹38 — a direct, unavoidable consequence
of the two-constraint fit, not a separate assumption.

**P2M cap bug (found and fixed)**: original `cap=None` design reasoned only in percentile terms
(p99.9≈₹61,170, under every ceiling) — but at production scale (11.7M draws) the realized max hit
₹1.24 CRORE, since a log-normal's tail is unbounded. Naive resample-above-cap (unchanged mu/sigma)
still drags the mean down 18% at just 0.05% resampled. A mu-only re-solve (mirroring the P2P fix
below) breaks the p86 constraint instead (drifts to ₹628 at cap=₹1L) — P2M has two real constraints,
not one, so a single free parameter can't fix truncated-mean without corrupting truncated-p86. Fix:
jointly re-solve **both** mu and sigma so the truncated distribution hits both targets at once
(`fit_p2m_params(cap=...)`, `truncated_lognormal_percentile`). Locked **cap=₹3,00,000** (midpoint of
the ambiguous ₹1–5 lakh range — flagged assumption, documented in the docstring) →
`mu=3.530206, sigma=2.485975`, resample-above-cap (not clip, to avoid a point-mass spike).

**P2P amount fit**: no independent RBI shape evidence exists (only ATS is real). Cap-as-percentile
was tested at p99/99.5/99.9/99.99 and rejected (infeasible or absurd near-zero medians). Shape
borrowed from P2M's sigma instead (flagged assumption). First version matched the *untruncated*
mean to 2812 — wrong, since the real ATS is already a post-cap observed figure; clipping then left
the post-clip mean ~26% short. Fixed the same way as P2M: re-solved mu so the **truncated** mean
(conditional on X≤cap) hits 2812 exactly → `mu=5.753896, sigma=2.391548` (sigma unchanged from the
shape-borrow). Resample-above-cap at ₹1,00,000. Validated at N=307,505: mean 2807.97 vs target 2812.

**Count/split mix**: total transaction count per applicant ~ Negative Binomial (overdispersed, not
Poisson), λ=**20/month** over **n_months=3** (explicit visible default). No RBI figure gives
individual transaction frequency at all — λ is **income-anchored**, not independently sourced:
blended ATS at the 0.635/0.365 mix = ₹1,444.84/txn, and λ=20 implies median simulated monthly UPI
turnover ≈2.0x `train_fe`'s median monthly income (`AMT_INCOME_TOTAL`/12 = ₹12,262.50) — read as
"money recycles through UPI about twice a month," plausible without being extreme. Confirmed the
mean-based estimate (2.36x) and median-based estimate (1.98x) diverge substantially due to the
per-transaction amounts' heavy right tail even after summing ~60 transactions — always compare
median-to-median or mean-to-mean, never mix the two. P2M/P2P split via independent per-transaction
Bernoulli(p=0.635), not a fixed ratio per applicant (0.635 is a national volume-share aggregate, not
a documented individual trait). Both count dispersion and split are deliberately independent of
income/region tier for now — correlation is deferred to the merge step, not skipped.

**Per-transaction rows**: `generate_upi_transactions()` composes the above directly — **not SDV**,
despite that being the original plan here. No real UPI microdata exists to fit an SDV synthesizer
*to*; every distribution above is calibrated or income-anchored, not learned from rows, so SDV would
only add approximation error on top of generators that are already exact. Revisit if a real
correlation-structure reference dataset ever surfaces for the deferred income/region-tier merge.
**Income correlation (count only, region tier deliberately excluded)**: transaction count is now
correlated with `AMT_INCOME_TOTAL` via `income_percentile_multiplier(income, lo=0.7, hi=1.3)` —
linear in income percentile rank (continuous, not decile bins), applied to `lam_month` before
drawing each applicant's count. Amounts (ticket-size distributions) stay unconditional on income —
RBI's ATS/percentile figures are national aggregates with no income-conditional breakdown, so
conditioning amounts would be a wholly new unranked assumption with zero anchor; frequency at least
extends the existing `lam_month` income-anchoring precedent. The multiplier is exactly 1.0 at the
median by construction, so the already-validated λ=20 / ~1.98x-median-ratio finding is **preserved**,
not re-anchored — confirmed post-injection: overall median ratio 2.01x. By-decile validation surfaced
a real, honestly-reported finding: the **bottom income decile's ratio is 3.05x** (vs. **1.09x** at the
top decile) — not a bug, but a direct consequence of income varying far more across deciles (~5.3x,
₹5,625→₹30,000 median monthly) than the deliberately modest ±30% count multiplier does. Reads as
poorer applicants cycling proportionately more of their income through UPI (lower savings buffer →
higher money-recycling velocity relative to income) — economically plausible, but flag the 3.05x
figure explicitly in the thesis methodology as an emergent property of the two assumptions stacked
together, not a separately-calibrated target. Confirmed the income↔count correlation survives full
row-level generation: Pearson r=0.234 (income vs. per-applicant txn count, small-to-moderate, as
expected given NB dispersion dilutes it).

**`REGION_RATING_CLIENT` deliberately NOT correlated**: RBI publishes no district/region-tier
transaction granularity at all (no calibration precedent to extend, unlike income), and
`REGION_RATING_CLIENT` is a likely audited attribute in this project's Phase 5 fairness audit —
injecting a synthetic region correlation into the generator now would predetermine that audit's
findings rather than letting them emerge from whatever the model actually learns from real features.
Revisit only if a real reference dataset surfaces tying region to payment behavior.

Generated and saved: **18,457,353 rows** (all 307,505 `train_fe` applicants, ~60 txns/applicant avg,
income-correlated) → `data/synthetic/upi_transactions.parquet` (92.4 MB, overwrote the prior
income-independent version). Columns: `SK_ID_CURR`, `TXN_TYPE` (P2M/P2P), `AMOUNT`, `MONTH_INDEX`
(1..3, uniform-random within window).

**Merge onto `train_fe`**: `aggregate_upi_transactions(upi_df, n_months=3)` collapses the
18.46M-row transaction table to one row per `SK_ID_CURR` (13 features — counts/turnover overall and
by P2M/P2P, `UPI_P2M_SHARE`, avg/median/max ticket size, monthly turnover std/CV, active-months
count), following the exact pre-check convention used for every Track A table (ID overlap,
cardinality, missingness, sort order all confirmed before writing the function). `merge_upi_features`
then left-joins it onto `train_fe`, same pattern as `merge_bureau_features` etc. Unlike every real
Home Credit auxiliary table, this merge introduces **zero new missingness** at the locked generation
defaults — confirmed empirically (0 applicants missing UPI features) — because a 60-txn/applicant
average makes a zero-transaction draw vanishingly unlikely; `fillna(0)` on count/turnover/active-months
columns is still applied defensively (never assumed) in case a future reseed/parameter change
produces one, with ticket-size stats and `UPI_P2M_SHARE` deliberately left `NaN` rather than 0 in
that case (a nonexistent ticket size isn't meaningfully zero).

**3-month window: adequate for the research question, but weak for volatility features.** `n_months=3`
matches real-world alternative/cash-flow underwriting convention (e.g. RBI's Account Aggregator
framework, common fintech practice of using 3–6 months of statement history) — a defensible choice,
not an arbitrary one, and ~60 txns/applicant is plenty to stabilize the average-based features
(ticket size, turnover, P2M/P2P share). But `UPI_MONTHLY_TURNOVER_STD`/`UPI_MONTHLY_TURNOVER_CV` are
each computed from only **3 monthly observations per applicant** — a genuinely noisy std/CV estimate
at n=3, not a stable volatility signal — and `UPI_ACTIVE_MONTHS` is coarse by construction (only
4 possible values: 0–3). The window also structurally cannot capture salary-cycle seasonality, slow
income trends, or one-off shocks (job loss, medical expense), regardless of estimate quality within
it. Flag both points explicitly in the thesis methodology, next to the income-correlation caveat —
same spirit: surface the limitation rather than let the feature look more solid than it is.

**`UPI_TURNOVER`-to-income ratio deliberately excluded** from the merged features: turnover is
itself generated as a function of `AMT_INCOME_TOTAL` (via `income_percentile_multiplier`), so a
precomputed ratio feature would be largely circular — letting the model "discover" a relationship
built into the generator by construction. Raw components (`UPI_TXN_COUNT_TOTAL`,
`UPI_TURNOVER_TOTAL`, etc.) are exposed instead, and whatever relationship exists is left for the
model/SHAP to find on its own. Flag this as a methodology caveat regardless — even the raw turnover
carries some of that income-derived signal.

Wired into Cell 2 as the last table-merge step (right before `drop_zero_variance_features`).
**Current cumulative `train_fe.shape`: `(307505, 307)`** — 294 (Track A) + 13 (Track B UPI features),
confirmed via Restart Kernel → Run All Cells, zero errors.

Full derivation, diagnosis, and validation for all of the above: `notebooks/03_synthetic_upi.ipynb`
(generation) and `notebooks/02_feature_engineering.ipynb` Cell 2 (aggregation + merge).

## Cell 2 pipeline (current order)
```python
from feature_engineering import (
    fix_days_employed, consolidate_building_stats, add_ext_source_missing_flags,
    transform_amounts, fit_target_encoding, apply_target_encoding, one_hot_encode_remaining,
    add_ratio_features,
    aggregate_bureau, merge_bureau_features,
    aggregate_previous_application, merge_previous_application_features,
    aggregate_installments, merge_installments_features,
    aggregate_pos_cash, merge_pos_cash_features,
    aggregate_bureau_balance, merge_bureau_balance_features,
    aggregate_credit_card_balance, merge_credit_card_balance_features,
    aggregate_upi_transactions, merge_upi_features,
    drop_zero_variance_features
)

train_raw = pd.read_csv('../data/raw/home-credit-default-risk/application_train.csv')
before_rows = len(train_raw)
train_raw = train_raw[train_raw['CODE_GENDER'] != 'XNA']
train_raw = train_raw[train_raw['NAME_FAMILY_STATUS'] != 'Unknown']
print(f"Dropped {before_rows - len(train_raw)} rows: {before_rows} -> {len(train_raw)}")

train_fe = fix_days_employed(train_raw)
train_fe = consolidate_building_stats(train_fe)
train_fe = add_ext_source_missing_flags(train_fe)
train_fe = transform_amounts(train_fe)

encoding_maps = fit_target_encoding(train_fe)
train_fe = apply_target_encoding(train_fe, encoding_maps)
train_fe = one_hot_encode_remaining(train_fe)

train_fe = add_ratio_features(train_fe)

bureau_raw = pd.read_csv('../data/raw/home-credit-default-risk/bureau.csv')
bureau_agg = aggregate_bureau(bureau_raw)
train_fe = merge_bureau_features(train_fe, bureau_agg)

prev_raw = pd.read_csv('../data/raw/home-credit-default-risk/previous_application.csv')
prev_agg = aggregate_previous_application(prev_raw)
train_fe = merge_previous_application_features(train_fe, prev_agg)

instal_raw = pd.read_csv('../data/raw/home-credit-default-risk/installments_payments.csv')
instal_agg = aggregate_installments(instal_raw)
train_fe = merge_installments_features(train_fe, instal_agg)

pos_cash_raw = pd.read_csv('../data/raw/home-credit-default-risk/POS_CASH_balance.csv')
pos_agg = aggregate_pos_cash(pos_cash_raw)
train_fe = merge_pos_cash_features(train_fe, pos_agg)

bureau_balance_raw = pd.read_csv('../data/raw/home-credit-default-risk/bureau_balance.csv')
bb_agg = aggregate_bureau_balance(bureau_balance_raw, bureau_raw)
train_fe = merge_bureau_balance_features(train_fe, bb_agg)

cc_raw = pd.read_csv('../data/raw/home-credit-default-risk/credit_card_balance.csv')
cc_agg = aggregate_credit_card_balance(cc_raw)
train_fe = merge_credit_card_balance_features(train_fe, cc_agg)

upi_raw = pd.read_parquet('../data/synthetic/upi_transactions.parquet')
upi_agg = aggregate_upi_transactions(upi_raw)
train_fe = merge_upi_features(train_fe, upi_agg)

train_fe = drop_zero_variance_features(train_fe)

print(f"\nFinal shape: {train_fe.shape}")

processed_path = '../data/processed/train_fe.parquet'
train_fe.to_parquet(processed_path, index=False, engine='pyarrow')
```
**Current verified shape: `(307505, 307)`** — row-drop fix, feature selection, and the Track B UPI merge all applied and confirmed end-to-end (see Track A and Track B sections above).

**Persisted to `data/processed/train_fe.parquet`** (120.8MB, Parquet not CSV — preserves the int8/int32/float32/category dtypes exactly; a CSV round-trip would silently widen them back to int64/float64/object). Round-trip verified (shape + dtypes match on reload). This is now the canonical engineered-features artifact — Phase 3 modelling should load this file directly rather than re-running the full notebook.

## Phase 3 — Modelling (complete)

**Notebook**: `notebooks/03_modelling.ipynb`, built up cell-by-cell (same convention as Phase 2).

**Data & split**: loads `data/processed/train_fe.parquet` directly (307505, 307) — no
`application_test.csv` involvement, since the Kaggle test set has no `TARGET` column and can't
be used for evaluation. 80/20 stratified split on `TARGET` (`random_state=42`): `X_train`
(246004, 305), `X_test` (61501, 305), both preserving the 8.0730% default rate exactly. Test
set is reserved for final evaluation only; CV/tuning happens within the train split.

**Evaluation function**: `evaluate_classifier(y_true, y_proba, label='')` in `src/modelling.py`
— AUC-ROC, Gini (`2*AUC-1`), and KS statistic (max TPR−FPR separation), the three standard
credit-scoring metrics per the proposal's evaluation-metrics section. Scratch → validate →
commit, same pattern as Track A/B functions.

**Baselines** (untuned, `class_weight='balanced'`/`scale_pos_weight` for the 11.39:1 imbalance):

| Model | AUC-ROC | Gini | KS | Notes |
|---|---|---|---|---|
| Logistic Regression | 0.7750 | 0.5500 | 0.4169 | median-imputed + scaled |
| Random Forest | 0.7594 | 0.5187 | 0.3930 | median-imputed, untuned (300 trees, depth 10) — trails LR |
| XGBoost | 0.7863 | 0.5725 | 0.4328 | native NaN handling, no imputation — best of the three, as expected |

None has yet crossed H1's AUC-ROC > 0.80 target; XGBoost is closest and is the tuning target.
Random Forest trailing Logistic Regression is plausible at these untuned depth/tree-count
settings on 305 already-engineered features — not a bug, just an artifact of no tuning yet.

**Ablation study (H2) — negative result, diagnosed and reported honestly, not a bug:**

| Feature set | AUC-ROC | Gini | KS |
|---|---|---|---|
| Bureau-only (292 features) | 0.7874 | 0.5748 | 0.4327 |
| UPI-only (13 features) | 0.5023 | 0.0045 | 0.0125 |
| Combined (305 features) | 0.7863 | 0.5725 | 0.4328 |

H2's threshold (ΔAUC ≥ 0.02, combined vs. bureau-only) is **not met** — the sign is even
slightly negative. UPI-only is statistically indistinguishable from a random classifier.

**Root cause, traced not assumed**: the UPI generator's only per-applicant real-data anchor is
`income_percentile_multiplier(AMT_INCOME_TOTAL)` (see Track B above). Checked `AMT_INCOME_TOTAL`
alone against `TARGET` directly: **AUC 0.4809, correlation -0.0199** — raw income is essentially
non-predictive of default in this dataset (`EXT_SOURCE_*` and credit-history features carry the
real signal, not income). Every UPI feature inherits that same near-zero signal through its one
connection to real applicant data, diluted further by injected randomness (NB-distributed counts,
log-normal ticket sizes, per-transaction Bernoulli P2M/P2P splits) — confirmed per-feature: all 13
UPI columns individually sit at AUC 0.486–0.503 against `TARGET`, no exceptions.

**Not a modelling bug — a direct consequence of a documented design choice**: the generator was
calibrated to RBI aggregate *distributions* and to income, deliberately never given a relationship
to `TARGET` (no real UPI microdata exists to calibrate one against — same reasoning already
documented for excluding `REGION_RATING_CLIENT`). Calibrating synthetic behavioral data to
plausible real-world distributions does not automatically make it predictive of an unrelated
outcome variable unless that relationship is deliberately injected — which this project's
methodology explicitly chose not to do, for lack of real evidence to justify it. **Report this as
a genuine, diagnosed negative finding in the thesis**, consistent with the project's existing
limitations-transparency stance (income-correlation caveat, 3-month-window caveat). All subsequent
tuning/ensemble work proceeds on the **bureau-only** feature set (UPI columns dropped) — they add
no signal and slightly dilute tree-split capacity in the combined set at fixed hyperparameters.

Full derivation: `notebooks/03_modelling.ipynb`, ablation + diagnosis cells.

**Hyperparameter tuning (Optuna, bureau-only)**: first attempt (40 trials, full 197K-row
validation split, `n_estimators` cap 1000) timed out past 30 minutes — too heavy. Reduced:
search runs on a 60K-row stratified subsample (48K/12K train/valid), `n_estimators` cap 400,
20 trials, `TPESampler(seed=42)`. Completed in 139.0s, best subsample validation AUC 0.7645.
Best params: `max_depth=4, learning_rate=0.0421, subsample=0.761, colsample_bytree=0.634,
min_child_weight=10, gamma=1.769, reg_alpha=8.37, reg_lambda=0.0393`. Refit on the **full**
246K-row train split (`n_estimators=800`, early-stopped at iteration 797) and evaluated once
on the untouched test set: **AUC-ROC 0.7913, Gini 0.5825, KS 0.4404** — a genuine improvement
over untuned bureau-only (0.7874), still short of H1's 0.80 target.

**Stacked ensemble**: LR + RF + tuned-XGBoost (fixed `n_estimators=798`, no early stopping —
`StackingClassifier`'s internal CV can't cleanly support per-fold eval sets) base learners, all
on bureau-only features, LR meta-learner on out-of-fold predictions (`cv=3`). Result: **AUC-ROC
0.7914** — ties the tuned XGBoost alone (ΔAUC +0.0001, negligible). Stacking added no real value
here — unsurprising, since the ensemble's weaker members (untuned LR/RF) don't offer much
complementary error structure against the much stronger XGBoost base learner for a linear
meta-learner to exploit.

**Full Phase 3 model comparison** (all on the untouched test set):

| Model | AUC-ROC | Gini | KS |
|---|---|---|---|
| Stacked Ensemble (bureau-only) | 0.7914 | 0.5827 | 0.4426 |
| XGBoost (bureau-only, tuned) | 0.7913 | 0.5825 | 0.4404 |
| XGBoost (bureau-only, untuned) | 0.7874 | 0.5748 | 0.4327 |
| XGBoost (baseline, combined) | 0.7863 | 0.5725 | 0.4328 |
| Logistic Regression (baseline, combined) | 0.7750 | 0.5500 | 0.4169 |
| Random Forest (baseline, combined) | 0.7594 | 0.5187 | 0.3930 |

**H1's AUC-ROC > 0.80 target is not met by any model.** This is reported as a real, if modest,
shortfall (best result 0.7914), consistent with — not contradicted by — the H2 diagnosis above:
there was no untapped alternative-data signal available for tuning or ensembling to recover,
since the synthetic UPI features carry none for `TARGET`. 0.79 AUC on the bureau/traditional
feature set alone is a strong, literature-typical result for Home Credit; the shortfall against
0.80 reflects the absent alternative-data signal in this synthetic-data construction, not a
modelling-technique weakness. **Report H1 and H2 together in the thesis as one coherent finding**:
cash-flow-style alternative data can, in principle, help credit scoring (per Berg et al. and this
project's own literature survey), but only if it's actually correlated with the outcome being
predicted — which requires either real behavioral data or a deliberately-injected outcome
relationship in any synthetic substitute, neither of which this project's methodology provided.

**Subgroup ablation — does UPI data help the *actually* credit-invisible population?** The
population-wide ablation tests all 307,505 applicants, most of whom have *some* bureau history —
raising the objection that any alternative-data lift could simply be swamped by a strong bureau
signal most people already have. Sharper test: restrict to `HAS_BUREAU_RECORD == 0` (44,019
applicants, 14.31% — the dataset's truest credit-invisible cohort, per `merge_bureau_features`'s
own docstring), same train/test split (just filtered, not re-split), same untuned XGBoost
architecture:

| Feature set | AUC-ROC | Gini | KS |
|---|---|---|---|
| Bureau-only (no-bureau subgroup) | 0.7565 | 0.5130 | 0.4017 |
| UPI-only (no-bureau subgroup) | 0.4984 | -0.0032 | 0.0166 |
| Combined (no-bureau subgroup) | 0.7549 | 0.5098 | 0.3873 |

**Result holds, and is now more convincing, not less.** UPI-only is still statistically random
(AUC 0.4984) and combined is again marginally below bureau-only (ΔAUC -0.0016) — same sign as
the population-wide result. This rules out the natural objection that bureau data was just
swamping a real signal: restricted to applicants with *no* bureau signal to be swamped by, UPI
still contributes nothing. Consistent with the root cause (income's own near-zero predictive
power for `TARGET`), since that has nothing to do with bureau-record status. Also notable:
bureau-only AUC is meaningfully lower here (0.7565 vs. 0.7874 population-wide, since all 44
`bureau.csv`-derived columns are constant/absent for this cohort by construction) — this
subgroup is inherently harder to score, which is exactly why it's the population a working
alternative-data source would matter most for, and exactly why this null result is the sharper
one to report: **the population this project's motivation centers on is the one this
synthetic-data construction was least able to help.**

**Final model selected for Phase 4/5 (SHAP + fairness audit): the tuned XGBoost (`tuned_clf`,
bureau-only), not the stacked ensemble** — SHAP's `TreeExplainer` applies cleanly to a single
XGBoost model, and the ensemble's negligible AUC gain doesn't justify the added explainability
complexity of a 3-model stack for the phases that are this project's actual differentiator.

**Model persistence**: final cell saves `tuned_clf` + `best_params` + `bureau_cols`/`upi_cols`/
`feature_cols` + `tuned_metrics` + the exact `test_index` to `models/tuned_xgb_bureau.joblib`
(via `joblib.dump`, `models/` gitignored-worthy but currently tracked) — so Phase 4/5 load the
trained model directly instead of re-running this notebook's 139s Optuna search + 409s stacking
fit. Same pattern as this notebook loading `train_fe.parquet` instead of re-running Phase 2.

**Phase 3 is complete.** Notebook: `notebooks/03_modelling.ipynb` (42 cells).

## Phase 4 — SHAP Explainability (complete)

**Notebook**: `notebooks/04_shap_explainability.ipynb` (23 cells), built cell-by-cell. Loads
`models/tuned_xgb_bureau.joblib` directly (not re-running Phase 3) and reconstructs the exact
same test set via the saved `test_index`.

**Tests H4** (Explainability consistency) — Null: SHAP explanations for structurally similar
applicants (feature cosine similarity > 0.95) are inconsistent (SHAP rank correlation < 0.70).
Alternative: explanations are consistent (rank correlation > 0.70).

**SHAP setup**: `TreeExplainer` on `tuned_clf`, computed on a 5,000-row sample of the test set
stratified by `TARGET` (compute budget for the O(n²) pairwise-similarity test below; SHAP
computation itself took 1.8s — TreeExplainer is exact, not sampled).

**Global feature importance**: `EXT_SOURCE_1/2/3` dominate, followed by credit-ratio/repayment
features (`CREDIT_GOODS_RATIO`, `INSTAL_LATE_PAYMENT_RATE`, `ANNUITY_CREDIT_RATIO`) and
demographics (`CODE_GENDER_F`, `OWN_CAR_AGE`) — matches domain expectations and confirms Phase
3's diagnosis that raw income carries no real signal (it's nowhere near the top). Plots saved to
`figures/shap_global_importance.png` (bar) and `figures/shap_beeswarm.png` (direction + magnitude).

**10 individual applicant explanations**: selected across the full predicted-risk range (3
lowest, 4 near-median, 3 highest of the 5,000-row sample; range 1.2%–98.2% predicted default
probability) — not 10 random/similar rows. Waterfall plots saved to
`figures/shap_waterfall_{1..10}.png`. Low-risk applicants are explained by strong `EXT_SOURCE_*`
values; high-risk applicants by weak `EXT_SOURCE_2` plus specific red flags (credit-card
utilization, overdue bureau amounts, late-payment history).

**Plain-language explanation template** (`explain_applicant()` in the notebook): converts each
applicant's top-5 SHAP contributors into a reason string suitable for a non-technical loan
officer — the deliverable the RBI Digital Lending Guidelines require (algorithmic basis
explainable to applicants on request, per the research proposal's Section 10.1).

**H4 test — diagnosed, same pattern as Phase 3's H2 diagnosis, not taken at face value:**

1. *Naive attempt* (cosine similarity > 0.95 across all 292 standardized, median-imputed
   features): **zero of 12,497,500 pairs qualify** — max similarity found was 0.94, median ≈ 0
   (near-orthogonal). Diagnosed as curse-of-dimensionality, not a failed test: two applicants
   would need to coincide almost exactly across 292 largely-independent dimensions, including
   dozens of sparse low-signal aggregates, to clear 0.95. The threshold is untestable as literally
   stated in this feature space.
2. *Fix*: restrict "structural similarity" to the top 10 features by mean |SHAP| (the features
   that actually drive predictions — also the more substantively meaningful notion of similarity
   for this hypothesis's purpose). Yields **2,750 qualifying pairs** (0.022% of all pairs).
3. *Naive rank correlation* (full 292-dim SHAP vector, on those 2,750 pairs): mean **0.337** —
   diluted by tie-breaking noise among ~280 features with near-zero SHAP for any given applicant.
4. *Diagnosed rank correlation* (restricted to the same top-10 SHAP dimensions — the reasons an
   applicant would actually be given): mean **0.778**, median 0.818, 73.4% of pairs individually
   clear 0.70.

**H4 is supported under this diagnosed, documented formalization** (mean rank correlation 0.778
> 0.70) — applicants who look alike on the factors that actually matter get explanations that
agree on those same factors. The letter of the original 0.95-on-292-dims threshold cannot be
evaluated as written, but the substantive claim H4 exists to test — explanation consistency for
regulatory-compliance purposes — holds.

**Caveat for the thesis**: this phase explains the *bureau-only* model (per Phase 3's
model-selection rationale) — since H2 found no UPI signal, no alternative-data feature appears
among the top drivers here. Direct, expected consequence of the Phase 3 finding, not a new
result; worth revisiting if the UPI generator is ever reworked to carry real signal.

Full derivation: `notebooks/04_shap_explainability.ipynb`.

## Phase 5 — Fairness Audit (in progress)

**Notebook**: `notebooks/05_fairness_audit.ipynb`, built cell-by-cell, same convention as
Phases 3/4. Loads `models/tuned_xgb_bureau.joblib` directly and reconstructs the exact same
test set via the saved `test_index` (confirmed `X_test.shape == (61501, 292)`, default rate
8.0730%, matching Phases 3/4 exactly).

**New module**: `compute_fairness_metrics()` and `sweep_fairness_thresholds()` in
`src/fairness.py`, mirroring `evaluate_classifier()`'s style in `src/modelling.py`. Both were
scratch-validated against hand-computed values on synthetic data before running on real data —
same discipline as every `aggregate_*`/Track B function. That validation caught a real bug:
`fairlearn.metrics.MetricFrame` does not accept a per-metric `y_pred` dict the way first
assumed (selection_rate needs the thresholded prediction, AUC needs the raw probability) —
fixed by using two separate `MetricFrame` instances instead of one.

**Gender audit** (`CODE_GENDER_F`, threshold=0.5): demographic parity difference **0.1385**,
equalized odds difference **0.1316**. Men flagged high-risk 37.59% of the time vs. 23.74% for
women — directionally consistent with the already-known raw default-rate gap (M 10.14% vs.
F 6.99–7.00%, see Key Numbers below), so the model is reflecting a real label difference, not
manufacturing a new one. Per-group AUC (0.7904 M / 0.7851 F) stays close to the overall test
AUC (0.7913) — the model isn't degrading for either group.

**Region-tier audit** (`REGION_RATING_CLIENT`, threshold=0.5, 3 ordinal groups): demographic
parity difference **0.2769**, equalized odds difference **0.2519** — roughly double the gender
gap. Tier 1 (best) flagged 14.37% of the time vs. Tier 3 (worst) 42.06%. Directionally
consistent with the known region-tier default-rate range (4.82% Tier 1 → 11.10% Tier 3).
**Particularly meaningful here**: Track B's synthetic UPI generator deliberately withheld any
region-tier correlation specifically so this audit's findings would emerge from real Home
Credit features only, not be predetermined by this project's own synthetic-data construction —
so this gap is not an artifact introduced upstream. Per-group AUC (0.7992 / 0.7847 / 0.7891)
again stays close to the overall AUC across all three tiers.

**Threshold sweep**: re-ran both audits across thresholds 0.1–0.9 (`sweep_fairness_thresholds`,
itself re-validated on a synthetic 3-group case before use) to check whether the 0.5-threshold
gaps are an artifact of that one cutoff. They are not — both demographic-parity curves stay
well above 0 across nearly the entire range, only collapsing near the extremes (0.1, 0.9),
which is expected (near-universal selection/rejection mechanically erases any group gap).
Region rating's gap dominates gender's at **every single threshold tested**, peaking at 0.323
(region) vs. 0.161 (gender), both at threshold=0.3. Equalized odds difference is noisier and
less monotonic — likely small-numerator noise at high thresholds where few applicants are
flagged at all — reported as-is rather than smoothed over, same honest-reporting stance as
H2/H4. Plot saved to `figures/fairness_threshold_sweep.png`.

**AIF360 mitigation** (`EqOddsPostprocessing`, region rating, Tier 1 vs. Tier 3): AIF360's
postprocessing algorithms are binary-group only, so this covers the two extreme tiers exclusively
(15,961 of 61,501 test rows; Tier 2 and the gender gap remain unmitigated in this pass).
`apply_eq_odds_postprocessing()` (`src/fairness.py`) was validated on a synthetic deliberately-
biased example first (equalized odds diff 0.2165 → 0.0002) before running on real data. Result:
**near-complete equalization** — equalized odds diff **0.2519 → 0.0051**, demographic parity diff
**0.2769 → 0.0215**. The algorithm found a genuine compromise point between the two original
selection rates (Tier 1: 14.37% → 32.53%, Tier 3: 42.06% → 34.68%), not a fixed "match the better
group" or "match the worse group" rule — confirmed empirically, not assumed, since the earlier
synthetic check happened to move the privileged group's rate all the way to the unprivileged
group's instead. **Real, honestly-reported cost**: overall accuracy on the Tier1+Tier3 subset
fell **0.7295 → 0.6892** (~4.0pp) — the standard fairness-accuracy tradeoff of post-processing
mitigation, which adjusts decisions rather than retraining the model. Whether that cost is
acceptable for near-eliminating a 0.25-magnitude gap is a normative call for the thesis's
discussion section, not resolved here.

**AIF360 mitigation, gender** (full test set, no subsetting needed — `CODE_GENDER_F` is already
binary, unlike region's 3 tiers): reuses the same, already-validated `apply_eq_odds_postprocessing`
code path. Result: equalized odds diff **0.1316 → 0.0065**, demographic parity diff
**0.1385 → 0.0117** — near-complete equalization again, selection rates converging to 33.04% (men)
/ 31.87% (women). **Notable, non-obvious pattern**: the accuracy cost (**0.7452 → 0.7050, ~4.02pp**)
is almost identical in magnitude to region's (~4.0pp), despite gender's original gap being roughly
half region's size — mitigation cost does not appear to scale proportionally with gap size here,
plausibly driven more by how many predictions must flip to reach *any* equalized-odds-satisfying
point than by the headline gap itself. Reported as an observed pattern across two runs, not a
general claim. **Both audited sensitive features are now mitigated** (gender: full test set;
region: Tier 1/Tier 3 subset only, per AIF360's binary-group constraint).

**AIF360 mitigation, Tier 2 coverage**: two more pairwise passes close the Tier 2 gap —
Tier 1 vs. Tier 2 (DPD 0.1324 → 0.0121, EOD 0.1219 → 0.0008) and Tier 2 vs. Tier 3
(DPD 0.1445 → 0.0121, EOD 0.1331 → 0.0087). Every tier now has at least one mitigation pass.
**Notable**: Tier 1 vs. Tier 2 is the only pass across the whole notebook where accuracy
*improved* (0.7643 → 0.7703) rather than fell — mechanically possible since
EqOddsPostprocessing's randomized flipping optimizes for the equalized-odds constraint, not
accuracy, reported as the exception not the rule. Tier 2 vs. Tier 3 costs the most of any
pass (0.7318 → 0.6619, ~7.0pp) despite a smaller starting gap than Tier 1 vs. Tier 3 —
further evidence mitigation cost doesn't track gap size simply, more likely driven by
subset-specific structure (45,540 vs. 9,579 rows here).

**Interaction check**: does mitigating one attribute affect the other? Both directions show
a mild **positive spillover**, not conflict — region's gap shrinks modestly after gender
mitigation (DPD 0.2769 → 0.2563, EOD 0.2519 → 0.2359), and gender's gap on the Tier1+Tier3
subset shrinks modestly after region mitigation (DPD 0.1323 → 0.1177, EOD 0.1538 → 0.1342).
Neither incidental effect is large enough to substitute for the attribute's own targeted
pass, and this finding is specific to this dataset/these two attributes, not assumed to
generalize to further audited attributes.

**Third sensitive attribute: education** (`NAME_EDUCATION_TYPE`, reconstructed from 5 one-hot
columns via `idxmax` — confirmed exactly one dummy=1 per row, zero missingness). `OCCUPATION_TYPE`
was deliberately skipped: it was target-encoded during Track A, so only a continuous,
TARGET-derived numeric column survives — using that as a fairness *grouping* variable would be
circular, so a proper occupation audit needs a raw re-join from `application_train.csv` first —
done below as the fourth attribute. Raw result: DPD **0.2184**, EOD **0.7171** — but the EOD was diagnosed, not
trusted at face value: `Academic degree` (n=38) has only **1 actual defaulter** in the test set,
and that single case wasn't flagged (TPR=0.0000) — a one-person estimate, not a stable rate.
Excluding that group: DPD unchanged (**0.2184**, driven by two large reliable groups — Lower
secondary 37.94% selection rate, n=759, vs. Higher education 16.10%, n=14,936), but EOD drops
from the artifact-inflated 0.7171 to a real, comparable **0.1990** (between gender's 0.1316 and
region's 0.2519). Selection rates track years of schooling in the expected direction.

**AIF360 mitigation, education** (Lower secondary vs. Higher education, the two extremes):
near-complete equalization (DPD 0.2184 → 0.0190, EOD 0.1990 → 0.0133) but at the **largest
accuracy cost of any pass in this notebook** — 0.8333 → 0.6995 (~13.4pp), more than triple the
usual ~4pp. Diagnosed, not left unexplained: `Higher education` (14,936 of 15,695 rows, 95% of
the subset) needed its selection rate moved from 16.10% to 31.04% to reach parity — roughly
2,231 of its predictions flipped, versus ~38 in `Lower secondary`. This refines the earlier
"cost doesn't track gap size alone" pattern with a mechanism: cost scales with how many rows in
the *dominant* group must flip, which depends on both the required rate shift and that group's
share of the subset — this was the most imbalanced pairwise subset yet (19.7:1), producing by
far the largest absolute flip count.

**Fourth sensitive attribute: occupation** (`OCCUPATION_TYPE`, raw re-join via `SK_ID_CURR`
from `application_train.csv` — the feature-engineered version is target-encoded, unusable as a
grouping variable since it's derived from the outcome itself. `SK_ID_CURR` survives in
`train_fe` as an ordinary column even though excluded from `bureau_cols`. 31.35% missing,
kept as an explicit `Not Reported` group rather than dropped — 100% merge coverage). Result:
**the largest disparity found across this entire fairness audit** — DPD **0.4783**, EOD
**0.4505**, nearly double region tier's and more than triple gender's. Driven by
`Low-skill Laborers` (63.98% selection rate) vs. `Accountants` (16.14%) for DPD, and the same
two groups' FPR gap (59.06% vs. 14.02%) for EOD. Checked, not assumed, to be real rather than
a small-sample artifact (the exact mistake the raw education EOD made): both extreme groups
are healthily sized (Low-skill Laborers n=372/52 positives; Accountants n=1,914/109
positives), and per-group AUC stays in a consistent 0.71–0.86 range across all 19 groups — a
genuine disparity, not a broken submodel for one group. `Not Reported` sits mid-pack (21.20%),
not itself an extreme. First pairwise mitigation pass: Low-skill Laborers vs. Accountants.

**AIF360 mitigation, occupation** — **different from every other pass, reported as such
rather than smoothed over**: demographic parity reaches near-parity (0.4783 → 0.0213), but
equalized odds only *partially* closes (0.4505 → 0.0919 — an order of magnitude larger than
every other pass's post-mitigation EOD, which all landed between 0.0008 and 0.0133), and the
accuracy cost is the largest of any pass yet (0.7822 → 0.5831, ~19.9pp). Plausible
explanation: this is by far the smallest subset mitigated (2,286 rows, ~7x smaller than the
next-smallest, education's 15,695) — EqOddsPostprocessing's equalized-odds guarantee is
realized via per-applicant randomized flipping at LP-fitted probabilities, and on a small
sample the realized empirical rates can deviate meaningfully from the theoretical target from
that randomization alone, compounded by starting from the largest gap of any attribute
audited. Flagged explicitly for the thesis discussion as a case where post-processing
mitigation alone may not fully suffice — not reported alongside the other four passes as if
all were equally clean.

### ✅ Threshold calibration — closed

Every pass above used an arbitrary 0.5 cutoff; this closes that gap. `src/decision_policy.py`
implements an example-dependent cost-sensitive threshold following Bahnsen, Aouada &
Ottersten (2014) — cost scales with each applicant's `AMT_CREDIT`, not a flat cost per
error. Cost assumptions (illustrative, not fitted to real recovery/margin data, same
"calibrated not learned" caveat as Track B): LGD = 0.55 of principal (mid-range of the
45–65% typically cited for unsecured retail/microfinance in India), profit margin = 0.15
of principal (mid-range of 10–20% NBFC/microfinance net interest margins). Scratch-validated
on 3 hand-computed cases (interior threshold, all-approve, all-reject) before use on real
data.

**Result: t\* = 0.71** (swept 0.01–0.99 in steps of 0.01, minimizing total expected cost on
the test set). vs. the arbitrary 0.5:
- Total expected cost: ₹167.7 Cr → ₹130.4 Cr (**−22.2%**)
- Approval rate: 71.52% → 90.28% (+18.8 pp)
- Default rate among approved: 3.54% → 5.53% (+1.99 pp — a deliberate, cost-optimal trade,
  not an oversight: LGD > margin per rupee, but ~11x more good applicants than defaulters at
  the ~8% base rate means low thresholds reject far more good applicants than they catch
  extra defaulters, so raising the threshold nets a lower *total* cost)

**Does the calibrated threshold change the fairness conclusions? Not uniformly** — diagnosed
with the same per-group TPR/FPR discipline used throughout this notebook (all three region
tiers have solid sample sizes, n_pos 303–3,594, so this is a real effect, not a small-sample
artifact):
- Demographic parity **improves** for both gender (DPD 0.1385→0.0657) and region tier (DPD
  0.2769→0.1369) — fewer people flagged high-risk overall shrinks the gap in flagging rates.
- Gender equalized odds **improves slightly** (EOD 0.1316→0.1209).
- Region-tier equalized odds **gets worse** (EOD 0.2519→0.3283). At 0.5 the binding gap was
  FPR (Tier 3 vs Tier 1, 0.375 vs 0.124); at t\*=0.71 TPR drops far more steeply for Tier 1
  (0.548→0.178) than Tier 3 (0.781→0.507), so the TPR gap becomes the new, larger binding
  constraint.

**Reported as-is**: the cost-optimal operating point and the fairest operating point are not
the same threshold — a real trade-off for the thesis discussion, not smoothed over the way
every other pass in this notebook was clean. This mirrors the occupation-mitigation finding
above in spirit: honest reporting of a genuine limitation rather than picking whichever
threshold makes the numbers look best.

Wired into the dashboard: `src/export_fairness_results.py` exports `t_star` and the
gender/region metrics at both thresholds; the Fairness Audit page shows this section, and
the Applicant Risk Explorer's threshold slider now **defaults to t\* = 0.71** (still fully
adjustable) instead of an arbitrary 0.5.

Full derivation: `notebooks/05_fairness_audit.ipynb` (cells 21–25, appended and re-executed
via Restart Kernel → Run All, zero errors, gapless execution counts 1–24).

## Phase 6 — Streamlit Dashboard (built)

Multi-page Streamlit app under `dashboard/`, run with `streamlit run dashboard/Home.py`
from the project root (venv activated). Four pages:

- **`dashboard/Home.py`** — project overview + headline model performance (AUC/Gini/KS)
  and dataset stats, read from the persisted model artifact.
- **`dashboard/pages/1_Applicant_Risk_Explorer.py`** — pick any test-set applicant (by row
  position or `SK_ID_CURR`), see their predicted default-risk score, a decision-threshold
  slider, and the top-15 SHAP-driven features for that specific applicant (live
  `shap.TreeExplainer` call on the single row — cheap, unlike the notebook's 5000-row
  summary sample; no precomputed artifact needed here since it must respond to whichever
  applicant is picked). Includes a per-applicant CSV export (`st.download_button`) of the
  full explanation (risk score, threshold decision, top-15 contributions) for a report
  appendix — one applicant at a time, not a bulk export.
- **`dashboard/pages/2_Fairness_Audit.py`** — headline DPD/EOD per sensitive attribute
  (gender, region tier, education, occupation), always paired with the per-group
  n/n_pos/n_neg/TPR/FPR table before the headline number (same "diagnose before trust"
  discipline as the notebook), threshold-sweep charts, and before/after AIF360
  `EqOddsPostprocessing` mitigation comparisons (region tier exposes all 3 pairwise
  passes via a radio selector).
- **`dashboard/pages/3_Model_Explainability.py`** — the global (model-wide) SHAP view,
  complementing the Applicant Explorer's per-applicant view: displays the pre-rendered
  `figures/shap_global_importance.png` and `shap_beeswarm.png`, plus the 10 pre-rendered
  waterfall examples from Phase 4 (grouped low/borderline/high risk via a radio + slider)
  — reads existing PNGs rather than recomputing SHAP on a fresh sample.
- **`dashboard/utils.py`** — shared `st.cache_resource`/`st.cache_data` loaders (model
  artifact, reconstructed test set, predictions, fairness results, SHAP explainer) so
  all pages load the model/data exactly once per server process and stay consistent
  with each other.

**`src/export_fairness_results.py`** — re-derives every number from
`notebooks/05_fairness_audit.ipynb` (same masks, same privileged/unprivileged
assignments, same `seed=42`) and writes `reports/fairness_results.json`. This JSON is
the fairness page's single source of truth — the dashboard never re-runs AIF360
mitigation live on page load (slow: ~19 groups × several mitigation passes), and this
keeps the dashboard from silently drifting out of sync with the audited notebook. Run
it again after any change to the model artifact or `src/fairness.py`, before reloading
the dashboard. Output spot-checked against the notebook's audited numbers on export
(gender DPD/EOD, region DPD/EOD, education raw/diagnosed EOD 0.7171/0.1990, occupation
EOD 0.4505→0.0919 post-mitigation) — all matched exactly.

**Verification**: syntax-checked (`py_compile`), then run through
`streamlit.testing.v1.AppTest` (Streamlit's headless script-execution harness) for every
page and every interactive branch — all 4 fairness-attribute selections, all 3
region-tier mitigation pairs, both applicant-picker modes, a non-default row and
threshold, all 3 explainability groups. Also verified live in a real browser session
(chrome-devtools MCP, screenshots + accessibility snapshots) for Home, Applicant
Explorer, Fairness Audit, and Model Explainability — confirmed rendered numbers match
the model artifact and `reports/fairness_results.json` exactly (e.g. gender DPD 0.1385 /
EOD 0.1316, mitigation deltas -0.1268/-0.1251), and that the CSV export button and the
waterfall-example selector actually render. Caught two real bugs this way before ever
opening a browser:
`dashboard/Home.py`'s `from utils import ...` raised `ModuleNotFoundError` under
Streamlit's multi-page-app runner (fixed: explicit `sys.path.append` of the script's
own directory, matching what the `pages/` files already did); the Applicant Explorer's
raw-feature-value table raised a PyArrow serialization error because a single row's
mixed bool/float/int values become one object-dtype numpy array (fixed: stringify that
column for display only). Also replaced the deprecated `use_container_width=True`
argument with `width='stretch'` throughout (flagged by Streamlit 1.58.0 as removed
after 2025-12-31). Final AppTest pass: zero exceptions across every page and branch.

**Not yet done**: no deployment (local `streamlit run` only); no authentication/access
control (not needed for a local academic-project demo); Phase 5's outstanding
threshold-to-lending-decision-rule work is unchanged by this phase — the dashboard's
threshold slider is illustrative, not a calibrated policy.

## Functions implemented (`src/feature_engineering.py`)
Column counts below are the current, post-row-drop-fix trace (307,505 base rows), confirmed by direct execution — not the pre-fix trace from earlier project states.

1. `fix_days_employed(df)` — `IS_NOT_EMPLOYED` flag (18.01%), 365243 placeholder → NaN. 122→123
2. `consolidate_building_stats(df)` — 14 triplets → `_AVG` only + `BUILDING_INFO_AVAILABLE` via `.any(axis=1)`. 123→96
3. `add_ext_source_missing_flags(df)` — `EXT_SOURCE_1_MISSING` (56.38%), `EXT_SOURCE_3_MISSING` (19.83%). 96→98
4. `transform_amounts(df)` — cap `AMT_INCOME_TOTAL` @ 10M, log-transform, preserve `_RAW`. 98→102
5. `fit_target_encoding(df, ...)` — fills NaN→'Missing' before grouping (bug fix, see below)
6. `apply_target_encoding(df, encoding_maps)`
7. `one_hot_encode_remaining(df)` — 14 object cols, `dummy_na=True`. Post-fix, `CODE_GENDER_XNA` and `NAME_FAMILY_STATUS_Unknown` are never created (those categories no longer exist in the data), which is the source of the 2-column difference vs. pre-fix totals downstream.
8. `add_ratio_features(df)` — 8 ratio features. →172 (pre-merge total; was 174 pre-fix)
9. `aggregate_bureau(bureau_df)` — bureau.csv (1.72M rows) → 45 cols
10. `merge_bureau_features(df, bureau_agg)` — 44,019 applicants (14.31%) with no bureau record post-fix. 172→217
11. `aggregate_previous_application(prev_df)` — 1.67M rows → 31 cols, incl. `PREV_REFUSAL_RATE`
12. `merge_previous_application_features(df, prev_agg)` — 16,452 (5.35%) no prior app post-fix. 217→248
13. `aggregate_installments(installments_df)` — 13.6M rows (largest table). `DAYS_LATE`, `AMT_SHORTFALL`, `MISSED_PAYMENT`, vectorized `INSTAL_LATE_PAYMENT_RATE`. 16 cols
14. `merge_installments_features(df, instal_agg)` — 15,866 applicants (5.16%) no installment history post-fix, confirmed. 248→264
15. `aggregate_pos_cash(pos_df)` — 10M rows. `NAME_CONTRACT_STATUS` → Active/Completed/Other. DPD via mean/max. `CNT_INSTALMENT_FUTURE` mean + `_LAST` (explicit sort by MONTHS_BALANCE required — raw file unsorted). 16 cols
16. `merge_pos_cash_features(df, pos_agg)` — 18,065 (5.87%) no POS_CASH record post-fix, confirmed. 264→280
17. `aggregate_bureau_balance(balance_df, bureau_df)` — 27.3M rows, two-hop merge (SK_ID_BUREAU → SK_ID_CURR via bureau.csv). STATUS pivoted into 8 categories + max-severity + ever-late-rate, then re-aggregated to 5 applicant-level cols. 6 cols total
18. `merge_bureau_balance_features(df, bb_agg)` — 280→286
19. `aggregate_credit_card_balance(cc_df)` — 3.84M rows → confirmed directly: **103,558 applicants, 19 columns** (includes ID column; print statement added this session — was previously the only `aggregate_*` function with no summary print, now matches the other eight). Contract status → Active/Completed/Other (96.31/3.36/0.34%). Drawings/payment NaNs verified as structural zeros before `fillna(0)`. `CNT_INSTALMENT_MATURE_CUM`/`AMT_INST_MIN_REGULARITY` NaNs verified as pre-first-instalment-cycle (MONTHS_BALANCE ≤ -21). Explicit sort before `.last()`. `UTILIZATION_RATIO`, `CC_COMPLETED_RATE` derived.
20. `merge_credit_card_balance_features(df, cc_agg)` — 220,600 (71.74%) no CC record post-fix, confirmed. 286→304
21. `aggregate_upi_transactions(upi_df, n_months=3)` — Track B merge step. 18.46M synthetic UPI txn rows → 307,505 applicants, 13 features (counts/turnover overall + by P2M/P2P, `UPI_P2M_SHARE`, avg/median/max ticket size, monthly turnover std/CV, active-months). See Track B section above for the full feature list and the deliberately-excluded turnover-to-income ratio.
22. `merge_upi_features(df, upi_agg)` — 0 applicants missing UPI features at current generation defaults (confirmed, not assumed); `fillna(0)` on count/turnover/active-months columns applied defensively regardless. 304→317
23. `drop_zero_variance_features(df)` — drops 9 exact-zero-variance `_nan` dummy columns + `FLAG_MOBIL` (near-constant, not exact-zero-variance — see Track A section above for the distinction). Wired in as the last step of Cell 2. 317→307

## Functions implemented (`src/synthetic_upi.py`)
Track B, all validated in `notebooks/03_synthetic_upi.ipynb` before being wired here (scratch → validate → commit, same as Track A). See the Track B section above for the full reasoning behind each fit/assumption.

1. `fit_lognormal_mean_percentile(target_mean, target_pctile_value, pctile, root)` — generic closed-form 2-constraint log-normal solver.
2. `fit_p2m_params(cap=None)` — P2M amount fit. `cap=None`: exact untruncated solve. `cap=<value>`: joint (mu, sigma) re-solve under truncation via `fsolve` (needed because P2M has two real constraints — a mu-only re-solve breaks p86).
3. `truncated_lognormal_mean(mu, sigma, cap)`, `truncated_lognormal_percentile(mu, sigma, cap, p)` — shared truncated-distribution moment/percentile helpers.
4. `fit_p2p_params(cap=100_000, target_mean=2812, sigma_p2m=None)` — P2P amount fit, shape borrowed from P2M, mu solved via `brentq` so the truncated mean hits the real ATS.
5. `generate_p2m_amounts(n, cap=300_000, random_state=None)` / `generate_p2p_amounts(n, cap=100_000, random_state=None)` — sample + resample-above-cap (not clip, to avoid a point-mass spike at the cap).
6. `fit_count_params(lam_month=20, n_months=3, var_mean_ratio=2.0)` — Negative Binomial parameters for total transaction count; λ is income-anchored (see Track B section), `var_mean_ratio` confirmed low-impact on downstream ratios (checked 1.5–3.0). `lam_month` may be scalar or a per-applicant array.
7. `generate_transaction_counts(n_applicants, lam_month, ...)` — draws per-applicant count; `lam_month` scalar (income/region-independent) or array (income-correlated, see `income_percentile_multiplier`).
8. `split_p2m_p2p_counts(counts, p_p2m=0.635, random_state=None)` — per-transaction Binomial split, not a fixed per-applicant ratio.
9. `income_percentile_multiplier(income, lo=0.7, hi=1.3)` — per-applicant count multiplier, linear in income percentile rank, 1.0 at the median by construction (preserves the λ=20 anchor). Deliberately not extended to region tier — see Track B section above.
10. `generate_applicant_turnover(n_applicants, ..., income=None)` — convenience wrapper (count → split → amounts → sum); used to income-anchor λ and to validate the by-decile turnover-ratio table.
11. `generate_upi_transactions(sk_id_curr, ..., income=None)` — full per-transaction row generator (the production entry point); returns one row per synthetic transaction, sorted by `SK_ID_CURR`. Not SDV — see Track B section above. `income=None` reproduces the original income-independent behavior exactly.

## Functions implemented (`src/fairness.py`)
Phase 5, all three validated on synthetic data (hand-computed 2-group/3-group cases, and a deliberately-biased case for the mitigation function) in scratch scripts before being wired into `notebooks/05_fairness_audit.ipynb` — same scratch → validate → commit convention as Track A/B. See the Phase 5 section above for the full findings.

1. `compute_fairness_metrics(y_true, y_proba, sensitive_feature, threshold=0.5, label='')` — demographic parity difference, equalized odds difference, and per-group selection rate/AUC via `fairlearn.metrics`. Uses two separate `MetricFrame` instances (one for thresholded selection rate, one for raw-probability AUC) — `MetricFrame` does not support a different `y_pred` array per metric within a single instance, a real bug caught during scratch validation.
2. `sweep_fairness_thresholds(y_true, y_proba, sensitive_feature, thresholds=None, label='')` — re-runs `compute_fairness_metrics` across a threshold grid (default 0.1–0.9 step 0.1), returns a DataFrame for trend inspection/plotting. Validated on a synthetic separated-group case (gap should collapse to 0 at extreme thresholds, peak where group score ranges cleanly diverge) before use.
3. `apply_eq_odds_postprocessing(y_true, y_pred, sensitive_feature, privileged_value, unprivileged_value, seed=42)` — AIF360 `EqOddsPostprocessing` wrapper; binary-group only (an AIF360 property, not a limitation of this wrapper), caller must pre-restrict `sensitive_feature` to exactly two values. `favorable_label=0`/`unfavorable_label=1` (favorable = not flagged high-risk, consistent with `compute_fairness_metrics`' framing). Validated on a synthetic deliberately-biased example (equalized odds diff 0.2165 → 0.0002) before use on real data.

## Findings worth citing in the thesis
- **Synthetic UPI features carry ~zero predictive signal for `TARGET` (H2 not supported)**: ablation study shows UPI-only AUC 0.5023 (random), combined (0.7863) doesn't beat bureau-only (0.7874). Traced to root cause: the generator's only real-data anchor, `AMT_INCOME_TOTAL`, itself has AUC 0.4809 against `TARGET` (correlation -0.0199) — income is a weak default predictor in this dataset, so a synthetic construct built only from income inherits that same weak-to-nil signal. Reported as an honest, diagnosed negative result, not a bug — see Phase 3 section above for the full derivation. **Confirmed to hold even in the subgroup the thesis's motivation actually centers on**: restricted to the 44,019 applicants with zero bureau record (`HAS_BUREAU_RECORD == 0`), UPI-only is still random (AUC 0.4984) and combined still doesn't beat bureau-only (ΔAUC -0.0016) — rules out the objection that bureau data was simply swamping a real alternative-data signal.
- **Bimodal bureau_balance coverage**: at the bureau-line level (scoped to bureau.csv's applicant set, unaffected by the row-drop fix), of applicants with ≥1 bureau line, 56.00% (171,269) have 0% of their lines covered by balance history, 43.85% (134,108) have 100% covered, only ~0.15% in between. At the `train_fe` applicant level post-fix (N=307,505), 215,274 (70.01%) have no bureau_balance history at all — combining the 44,019 with no bureau record plus the ~171K with a bureau record but 0% balance coverage. This is an institutional reporting-switch effect, not per-line randomness — verified empirically, not assumed.
- **CC utilization can legitimately exceed [0,1]**: `CC_UTILIZATION_MEAN` ranges -0.085 to 2.14, `_LAST` up to 11.78. Negative = overpayment; >1 = over-limit spending or a post-hoc credit limit reduction. Confirmed not a data error via spot check.
- **H4's literal 0.95-cosine-similarity-on-292-features threshold is untestable (curse of dimensionality), not failed**: zero of 12.5M sampled pairs qualify at the full feature-set dimensionality (max similarity found: 0.94). Restricting "structural similarity" to the top-10 features by mean |SHAP| makes it tractable (2,750 qualifying pairs) and is also the substantively correct notion of similarity for this hypothesis. Naive full-292-dim SHAP rank correlation on those pairs (0.337) is diluted by noise among ~280 near-zero-importance features; restricted to the same top-10 dimensions, mean rank correlation is 0.778 — **H4 supported** under this diagnosed formalization. See Phase 4 section above.

## Bugs already caught (don't reintroduce)
1. Building-stat availability flag using single reference column instead of `.any(axis=1)` across all 14 — undercounted.
2. Target encoding: `groupby()` silently drops NaN groups → missing `OCCUPATION_TYPE` fell to global mean instead of learned encoding. Fix: fill NaN with 'Missing' string before grouping.
3. Bureau coverage: naive estimate was ~1,700 applicants without a bureau record; actual was 44,020 (14.31%, pre-fix population; 44,019 post-fix) because bureau.csv spans train+test. Always verify ID overlap via set ops before trusting merge-count arithmetic.
4. Duplicate-merge bug (hit 3x — bureau, previous_application, installments): stale standalone cells re-merging onto already-merged `train_fe` → `_x`/`_y` suffix collisions. Root cause: re-running individual cells instead of Restart Kernel → Run All. This is why that rule exists.
5. Installments late-rate `.apply()` deprecation → replaced with vectorized `IS_LATE.mean()`.
6. Row-drop fix pasted into Cell 2 before `train_raw` existed (wrong insertion point) → crashed → cell reverted to clean form but the crash traceback was never cleared, and the kernel was never restarted afterward. Every downstream cell's displayed output was therefore stale (leftover from an earlier session), not reflecting the code as it actually stood — even though the *code* itself (once the fix was correctly repositioned) turned out fine. **Lesson: a notebook's stored output is not proof the current code was run. Check `execution_count` sequencing (should be gapless, ascending, no `null`s) before trusting any displayed shape/number, especially after this project's own history of skipped kernel restarts.**
7. Two stale standalone cells outside Cell 2 (`add_ext_source_missing_flags`, `transform_amounts`) were silently re-invoking pipeline functions already called inside Cell 2. The `transform_amounts` one was non-idempotent and corrupted `AMT_INCOME_TOTAL_RAW` on its second call (overwrote true raw income with the already-capped value). Same root cause as bug #4 — leftover pre-consolidation cells still wired to mutate `train_fe` — caught by an audit pass, not by a crash, since neither produced an error.
8. **Zero-variance check methodology trap**: `df.var(numeric_only=True) == 0` (the approach the notebook's own diagnostic cell originally used) reports **zero** constant columns even though 9 genuinely exist. `pd.get_dummies` produces `bool` dtype columns, and `numeric_only=True` / `select_dtypes(include='number')` silently excludes `bool` — so all 76 one-hot columns, including the true constants, never enter the variance calculation at all. Correct check: `df.nunique(dropna=False) == 1`, which is dtype-agnostic. Don't trust a "nothing is constant" result from a `.var()`-based check on a one-hot-encoded frame.
9. **[Track B] Uncapped log-normal tail produces structurally impossible values at scale**: `generate_p2m_amounts`'s original `cap=None` design checked only that p99.9 (~₹61,170) sat under every published ceiling — true, but irrelevant, since a log-normal's tail is unbounded. At production scale (11.7M draws) the realized max was ₹1.24 crore, ~12x the highest cited P2M ceiling. Small-N validation runs (the initial N=20,000 check) didn't surface this — the bug only showed up once generation ran at the real target volume. **Lesson: percentile-level checks (p99, p99.9) don't bound the maximum of an unbounded distribution; a "does this look safe" check must include the actual max at production N, not just a tail percentile at a smaller validation N.**
10. **[Track B] Fixing a truncated-mean constraint with only one free parameter can break a second, unrelated constraint**: re-solving P2M's mu alone (mirroring the P2P cap fix, which only had one real constraint) to hit the truncated mean broke the p86 constraint instead (drifted to ₹628 vs target ₹500 at cap=₹1L), because shifting mu shifts every percentile of the distribution together. Needed a joint (mu, sigma) re-solve instead. Caught by explicitly re-checking the *other* constraint after each candidate fix, not by assuming a fix for one target leaves everything else alone.

## Key numbers
All post row-drop-fix (N=307,505), reprinted directly from the notebook's current stored outputs — nothing estimated. EDA-stage numbers (default rate, gender/region gaps, missingness) are computed on the full 307,511 pre-fix population and unaffected by the 6-row drop.

| Metric | Value |
|---|---|
| Training applicants (raw) | 307,511 → 307,505 after row-drop fix (4 XNA gender + 2 Unknown family status) |
| Default rate | 8.07% (class imbalance ~11.4:1) |
| Region tier default range | 4.82% (Tier 1) → 11.10% (Tier 3) |
| Gender default gap | 6.99–7.00% (F) → 10.14% (M) |
| No bureau record | 44,019 (14.31%) post-fix — confirmed |
| No prior Home Credit app | 16,452 (5.35%) post-fix — confirmed |
| No installment history | 15,866 (5.16%) post-fix — confirmed |
| No POS_CASH record | 18,065 (5.87%) post-fix — confirmed |
| No bureau_balance coverage (train_fe scope) | 215,274 (70.01%) post-fix — confirmed |
| No bureau_balance coverage (bureau-line scope, unaffected by row-drop) | 56.00% of applicants with ≥1 bureau line have 0% coverage, 43.85% have 100% |
| No credit_card_balance record | 220,600 (71.74%) post-fix — confirmed |
| EXT_SOURCE_1 / _3 missingness | 56.38% / 19.83% |
| Baseline RF AUC (diagnostic, feature-selection pass only) | 0.7535 — RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced'), median-imputed, 80/20 split, not a pipeline artifact |
| **Current `train_fe` shape** | **(307505, 307)** — 294 (Track A) + 13 (Track B UPI merge), confirmed via clean Restart Kernel → Run All, zero errors, zero `_x`/`_y` duplicate columns, zero leftover diagnostic cells |

### Track B key numbers
All confirmed via direct execution in `notebooks/03_synthetic_upi.ipynb`, not estimated.

| Metric | Value |
|---|---|
| P2M fit (untruncated) | mu=3.630972, sigma=2.391548 — exact solve, ATS=659/p86=500 |
| P2M fit (cap=₹3,00,000, locked) | mu=3.530206, sigma=2.485975 — joint truncated re-solve |
| P2M cap bug: pre-fix realized max (11.7M draws) | ₹1.24 crore (uncapped) → ₹2,99,918 (post-fix, full N) |
| P2P fit (shape-borrowed, truncated-mean-corrected) | mu=5.753896, sigma=2.391548 (=P2M's untruncated sigma) |
| P2P post-resample mean @ N=307,505 | 2,807.97 (target ATS 2,812) |
| Blended ATS @ 0.635/0.365 mix | ₹1,444.84/txn |
| Median monthly income (`AMT_INCOME_TOTAL`/12, row-drop applied) | ₹12,262.50 |
| λ (income-anchored transaction count) | 20/month, n_months=3 (explicit default) |
| Median monthly turnover ratio to median income @ λ=20 | 1.98x (mean-based estimate: 2.36x — don't mix the two) |
| Income-correlated multiplier range | 0.7x–1.3x (linear in income percentile, 1.0 at median) |
| Median ratio post income-correlation | 2.01x overall; by decile: 3.05x (bottom) → 1.09x (top) |
| Income↔txn-count correlation (full row-level check) | Pearson r = 0.234 |
| Full per-transaction generation | 18,457,353 rows, 307,505/307,505 applicants, ~60 txns/applicant avg, income-correlated |
| Output file | `data/synthetic/upi_transactions.parquet` (92.4 MB) |
| UPI features merged onto `train_fe` | 13 (via `aggregate_upi_transactions`/`merge_upi_features`), 0 applicants missing |
| Final cumulative `train_fe` shape | (307505, 307) |

## What's left in Phase 2
- **Track A: fully closed.** All six auxiliary tables merged, row-drop fix applied, stale cells removed, feature selection done. No outstanding items.
- **Track B: feature generation & merge complete.** See the "✅ Track B" section above for the full write-up (RBI stats, both amount fits incl. the P2M cap bug, count-mix/income-anchoring, per-transaction row generation, income correlation, the merge onto `train_fe`, the SDV decision). Remaining: documenting two limitations in the thesis methodology (Phase 7, a writing task, not a pipeline gap) — (1) the calibrated-not-learned nature of every distribution/correlation in the generator, and (2) the 3-month window's weak volatility estimates (`UPI_MONTHLY_TURNOVER_STD`/`CV` at n=3 observations) and inability to capture seasonality/trends/shocks.
- **Optional/low-priority**: markdown section ordering in `02_feature_engineering.ipynb` is scrambled near the top (section "1" appears before section "0"; the credit_card_balance summary sits near the top instead of the end). Cosmetic only, flagged in an audit pass, not yet fixed — do only if there's spare time before review.

## After this file goes stale
Whenever a table gets merged, a bug gets fixed, or a new phase starts, this file should be regenerated to reflect it — treat it as a static snapshot, not a live log.
