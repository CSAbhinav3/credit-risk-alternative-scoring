# Project: Alternative Credit Scoring for India's Underbanked Population

Final year B.Tech CSE research project. Christ University, Bengaluru. Submission target: ~Oct 25, 2026.

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
│   ├── processed/                       ← engineered features saved here later
│   └── synthetic/
│       └── upi_transactions.parquet     ← 18.45M synthetic UPI txn rows (Track B, see below)
├── notebooks/
│   ├── 01_eda.ipynb                     ← Phase 1: complete
│   ├── 02_feature_engineering.ipynb     ← Phase 2: Track A complete
│   └── 03_synthetic_upi.ipynb           ← Phase 2: Track B in progress (amount fits, count-mix, row gen)
├── src/
│   ├── feature_engineering.py           ← Track A pipeline functions
│   └── synthetic_upi.py                 ← Track B synthetic UPI generators
├── rbi_dpss_upi_stats.md                ← RBI DPSS/NPCI aggregate stats, Track B calibration source
├── dashboard/
├── figures/
├── reports/
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
| 2 — Feature Engineering | ✅ Track A complete (incl. feature selection); Track B in progress |
| 3 — Modelling | Not started |
| 4 — SHAP Explainability | Not started |
| 5 — Fairness Audit | Not started |
| 6 — Streamlit Dashboard | Not started |
| 7 — Thesis + Paper | Not started |

### ✅ Track A — fully closed
Row-drop fix, stale-cell cleanup, and feature selection are all resolved and verified end-to-end.

**Row-drop fix**: Six rows (`CODE_GENDER == 'XNA'`, 4 rows; `NAME_FAMILY_STATUS == 'Unknown'`, 2 rows) dropped live in Cell 4, immediately after `pd.read_csv` for `application_train.csv`, before `fix_days_employed()`. Two stale cells that were silently corrupting `train_fe` on re-run were also deleted (former Cell 11 — redundant `add_ext_source_missing_flags` call; former Cell 14 — redundant `transform_amounts` call that was overwriting `AMT_INCOME_TOTAL_RAW` with the already-capped value on second call). Notebook went 69 → 65 cells.

**Feature selection**: `drop_zero_variance_features(df)` added as the pipeline's 21st function, wired in as the last step in Cell 2, right before the final-shape print. Drops 10 columns total:
- 9 exact-zero-variance `_nan` dummy columns (`nunique(dropna=False) == 1`): `CODE_GENDER_nan`, `FLAG_OWN_CAR_nan`, `FLAG_OWN_REALTY_nan`, `NAME_CONTRACT_TYPE_nan`, `NAME_EDUCATION_TYPE_nan`, `NAME_FAMILY_STATUS_nan`, `NAME_HOUSING_TYPE_nan`, `NAME_INCOME_TYPE_nan`, `WEEKDAY_APPR_PROCESS_START_nan` — all bool dtype, sole value `False` across all 307,505 rows (their source columns have zero missingness, so `dummy_na=True` produced a dead indicator).
- `FLAG_MOBIL` — **near-constant, not exact-zero-variance** (1 applicant out of 307,505 has `FLAG_MOBIL == 0`, rest are 1; 99.999675% dominant). Included in the drop on the strength of 0.0 permutation importance from a baseline RF, not on the zero-variance criterion. (An earlier version of this file incorrectly grouped `FLAG_MOBIL` with the exact-zero-variance columns — corrected here.)

`FLAG_DOCUMENT_*` columns were explicitly and deliberately **not** touched — confirmed untouched in the scratch-cell validation before the drop was committed.

**Final verified `train_fe.shape`: `(307505, 294)`** — confirmed via Restart Kernel → Run All Cells, sequential execution 1→47, zero errors, zero leftover diagnostic cells.

### 🔶 Track B — in progress
RBI stats gathered, both amount distributions fitted (with a real bug found and fixed), the
transaction count/split mix fitted and income-anchored, and per-transaction row generation done and
saved. Not yet done: income/region-tier correlation injection and the merge onto `train_fe`.

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
Generated and saved: **18,451,776 rows** (all 307,505 `train_fe` applicants, ~60 txns/applicant avg)
→ `data/synthetic/upi_transactions.parquet` (92.3 MB). Columns: `SK_ID_CURR`, `TXN_TYPE`
(P2M/P2P), `AMOUNT`, `MONTH_INDEX` (1..3, uniform-random within window).

Full derivation, diagnosis, and validation for all of the above: `notebooks/03_synthetic_upi.ipynb`.

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
    aggregate_credit_card_balance, merge_credit_card_balance_features
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

print(f"\nFinal shape: {train_fe.shape}")
```
**Current verified shape: `(307505, 294)`** — row-drop fix and feature selection both applied and confirmed end-to-end (see Track A section above).

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
21. `drop_zero_variance_features(df)` — drops 9 exact-zero-variance `_nan` dummy columns + `FLAG_MOBIL` (near-constant, not exact-zero-variance — see Track A section above for the distinction). Wired in as the last step of Cell 2. 304→294

## Functions implemented (`src/synthetic_upi.py`)
Track B, all validated in `notebooks/03_synthetic_upi.ipynb` before being wired here (scratch → validate → commit, same as Track A). See the Track B section above for the full reasoning behind each fit/assumption.

1. `fit_lognormal_mean_percentile(target_mean, target_pctile_value, pctile, root)` — generic closed-form 2-constraint log-normal solver.
2. `fit_p2m_params(cap=None)` — P2M amount fit. `cap=None`: exact untruncated solve. `cap=<value>`: joint (mu, sigma) re-solve under truncation via `fsolve` (needed because P2M has two real constraints — a mu-only re-solve breaks p86).
3. `truncated_lognormal_mean(mu, sigma, cap)`, `truncated_lognormal_percentile(mu, sigma, cap, p)` — shared truncated-distribution moment/percentile helpers.
4. `fit_p2p_params(cap=100_000, target_mean=2812, sigma_p2m=None)` — P2P amount fit, shape borrowed from P2M, mu solved via `brentq` so the truncated mean hits the real ATS.
5. `generate_p2m_amounts(n, cap=300_000, random_state=None)` / `generate_p2p_amounts(n, cap=100_000, random_state=None)` — sample + resample-above-cap (not clip, to avoid a point-mass spike at the cap).
6. `fit_count_params(lam_month=20, n_months=3, var_mean_ratio=2.0)` — Negative Binomial parameters for total transaction count; λ is income-anchored (see Track B section), `var_mean_ratio` confirmed low-impact on downstream ratios (checked 1.5–3.0).
7. `generate_transaction_counts(n_applicants, ...)` — draws per-applicant count, deliberately independent of income/region tier.
8. `split_p2m_p2p_counts(counts, p_p2m=0.635, random_state=None)` — per-transaction Binomial split, not a fixed per-applicant ratio.
9. `generate_applicant_turnover(n_applicants, ...)` — convenience wrapper (count → split → amounts → sum); used to income-anchor λ.
10. `generate_upi_transactions(sk_id_curr, ...)` — full per-transaction row generator (the production entry point); returns one row per synthetic transaction, sorted by `SK_ID_CURR`. Not SDV — see Track B section above.

## Findings worth citing in the thesis
- **Bimodal bureau_balance coverage**: at the bureau-line level (scoped to bureau.csv's applicant set, unaffected by the row-drop fix), of applicants with ≥1 bureau line, 56.00% (171,269) have 0% of their lines covered by balance history, 43.85% (134,108) have 100% covered, only ~0.15% in between. At the `train_fe` applicant level post-fix (N=307,505), 215,274 (70.01%) have no bureau_balance history at all — combining the 44,019 with no bureau record plus the ~171K with a bureau record but 0% balance coverage. This is an institutional reporting-switch effect, not per-line randomness — verified empirically, not assumed.
- **CC utilization can legitimately exceed [0,1]**: `CC_UTILIZATION_MEAN` ranges -0.085 to 2.14, `_LAST` up to 11.78. Negative = overpayment; >1 = over-limit spending or a post-hoc credit limit reduction. Confirmed not a data error via spot check.

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
| **Current `train_fe` shape** | **(307505, 294)** — confirmed via clean Restart Kernel → Run All, zero errors, zero `_x`/`_y` duplicate columns, zero leftover diagnostic cells |

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
| Full per-transaction generation | 18,451,776 rows, 307,505/307,505 applicants, ~60 txns/applicant avg |
| Output file | `data/synthetic/upi_transactions.parquet` (92.3 MB) |

## What's left in Phase 2
- **Track A: fully closed.** All six auxiliary tables merged, row-drop fix applied, stale cells removed, feature selection done. No outstanding items.
- **Track B (in progress)** — see the "🔶 Track B — in progress" section above for the full write-up (RBI stats, both amount fits incl. the P2M cap bug, count-mix/income-anchoring, per-transaction row generation, the SDV decision). Remaining: income/region-tier correlation injection, the merge onto `train_fe`, and documenting the calibrated-not-learned limitation in the thesis methodology.
- **Optional/low-priority**: markdown section ordering in `02_feature_engineering.ipynb` is scrambled near the top (section "1" appears before section "0"; the credit_card_balance summary sits near the top instead of the end). Cosmetic only, flagged in an audit pass, not yet fixed — do only if there's spare time before review.

## After this file goes stale
Whenever a table gets merged, a bug gets fixed, or a new phase starts, this file should be regenerated to reflect it — treat it as a static snapshot, not a live log.
