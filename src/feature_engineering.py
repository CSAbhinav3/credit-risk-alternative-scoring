import pandas as pd
import numpy as np

def consolidate_building_stats(df):
    """
    The Home Credit dataset has ~20 building/apartment statistic columns in 
    _AVG / _MODE / _MEDI triplets (e.g. COMMONAREA_AVG, COMMONAREA_MODE, 
    COMMONAREA_MEDI). EDA confirmed these are co-missing (same root cause: 
    absence of the underlying building record) and the three variants within 
    each triplet carry largely redundant information.

    Strategy: for each triplet, keep only the _AVG column (drop _MODE and 
    _MEDI), and add a single binary flag indicating whether building data 
    was available for that applicant at all.
    """
    df = df.copy()

    # Identify all building-stat base names (the part before _AVG/_MODE/_MEDI)
    avg_cols = [c for c in df.columns if c.endswith('_AVG')]
    building_bases = []
    for col in avg_cols:
        base = col[:-4]  # strip "_AVG"
        mode_col = base + '_MODE'
        medi_col = base + '_MEDI'
        if mode_col in df.columns and medi_col in df.columns:
            building_bases.append(base)

    print(f"Found {len(building_bases)} building-stat triplets: {building_bases}")

    # Create ONE availability flag: 1 if ANY triplet's _AVG column is non-null,
    # 0 only if ALL 14 are missing. This is more robust than relying on a single
    # reference column, since individual triplets can have different missingness
    # rates even when the applicant does have some building record.
    if building_bases:
        avg_cols_all = [base + '_AVG' for base in building_bases]
        df['BUILDING_INFO_AVAILABLE'] = df[avg_cols_all].notnull().any(axis=1).astype(int)

    # Drop the _MODE and _MEDI columns for every triplet, keep _AVG
    cols_to_drop = []
    for base in building_bases:
        cols_to_drop.append(base + '_MODE')
        cols_to_drop.append(base + '_MEDI')

    df = df.drop(columns=cols_to_drop)

    print(f"Dropped {len(cols_to_drop)} redundant MODE/MEDI columns")
    print(f"Kept {len(building_bases)} _AVG columns + 1 new BUILDING_INFO_AVAILABLE flag")

    return df

def add_ext_source_missing_flags(df):
    """
    EDA found that EXT_SOURCE_1 (56.4% missing) and EXT_SOURCE_3 (19.8% missing)
    carry predictive signal in their missingness itself, not just their values:
    - EXT_SOURCE_1 missing -> 8.52% default rate vs 7.50% when present
    - EXT_SOURCE_3 missing -> 9.31% default rate vs 7.77% when present

    EXT_SOURCE_2 is excluded here (only 0.21% missing -> too rare to be a 
    useful flag on its own).

    This function creates binary flags BEFORE any imputation happens, so the 
    missingness signal is preserved as a separate feature rather than lost 
    when NaNs get filled in.
    """
    df = df.copy()

    df['EXT_SOURCE_1_MISSING'] = df['EXT_SOURCE_1'].isnull().astype(int)
    df['EXT_SOURCE_3_MISSING'] = df['EXT_SOURCE_3'].isnull().astype(int)

    print(f"EXT_SOURCE_1_MISSING: {df['EXT_SOURCE_1_MISSING'].sum()} flagged ({df['EXT_SOURCE_1_MISSING'].mean()*100:.2f}%)")
    print(f"EXT_SOURCE_3_MISSING: {df['EXT_SOURCE_3_MISSING'].sum()} flagged ({df['EXT_SOURCE_3_MISSING'].mean()*100:.2f}%)")

    return df

def transform_amounts(df):
    """
    EDA found AMT_INCOME_TOTAL and AMT_CREDIT are both heavily right-skewed, 
    with log-transform producing a much more usable distribution for both.

    EDA also found a single genuine outlier in AMT_INCOME_TOTAL (117,000,000 -
    a likely data entry error, isolated and disconnected from the next-highest
    value at 18,000,090). This is capped at 10,000,000 BEFORE log-transforming,
    since log-transform alone does not fix a true data error - it only 
    compresses its effect.

    AMT_CREDIT's maximum (4,050,000, repeated across multiple applicants) was 
    confirmed in EDA to be a legitimate loan product ceiling, not an outlier -
    no capping applied to AMT_CREDIT.

    Original raw columns are preserved with _RAW suffix for reference; the 
    base column name becomes the log-transformed, model-ready version.
    """
    df = df.copy()

    # --- AMT_INCOME_TOTAL ---
    df['AMT_INCOME_TOTAL_RAW'] = df['AMT_INCOME_TOTAL']
    income_cap = 10_000_000
    n_capped = (df['AMT_INCOME_TOTAL'] > income_cap).sum()
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=income_cap)
    df['AMT_INCOME_TOTAL_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])

    print(f"AMT_INCOME_TOTAL: capped {n_capped} rows at {income_cap:,}, log-transformed")

    # --- AMT_CREDIT ---
    df['AMT_CREDIT_RAW'] = df['AMT_CREDIT']
    df['AMT_CREDIT_LOG'] = np.log1p(df['AMT_CREDIT'])

    print(f"AMT_CREDIT: log-transformed (no capping - ceiling confirmed legitimate in EDA)")

    return df

def fit_target_encoding(df, high_card_cols=None, target_col='TARGET'):
    """
    Learns target-encoding lookup tables (category -> mean default rate) from
    a TRAINING dataframe only. Must never be called on validation/test data,
    since that would leak target information.

    Missing values are filled with the string 'Missing' before grouping, so 
    NaN gets its own learned encoding instead of falling back to a generic 
    global mean (relevant for OCCUPATION_TYPE, where ~31% missingness strongly 
    overlaps with the not-employed population - see fix_days_employed).

    Returns a dict: {column_name: {category: mean_target_rate}}
    """
    if high_card_cols is None:
        high_card_cols = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']

    encoding_maps = {}
    for col in high_card_cols:
        if col not in df.columns:
            continue
        col_filled = df[col].fillna('Missing')
        mapping = df[target_col].groupby(col_filled).mean()
        encoding_maps[col] = mapping
        print(f"Learned encoding for {col}: {len(mapping)} categories (including 'Missing' if present)")

    return encoding_maps


def apply_target_encoding(df, encoding_maps):
    """
    Applies pre-learned target-encoding lookup tables to ANY dataframe 
    (training, validation, or test). NaN is filled with 'Missing' to match 
    fit_target_encoding's treatment. Categories truly unseen during fitting
    (new categories appearing only in validation/test) fall back to that 
    column's global mean.
    """
    df = df.copy()

    for col, mapping in encoding_maps.items():
        if col not in df.columns:
            continue
        col_filled = df[col].fillna('Missing')
        global_mean = mapping.mean()
        df[col + '_TARGET_ENC'] = col_filled.map(mapping).fillna(global_mean)
        n_unseen = col_filled.map(mapping).isnull().sum()
        print(f"{col}: encoded, {n_unseen} truly-unseen categories filled with global mean ({global_mean:.4f})")

    df = df.drop(columns=[c for c in encoding_maps.keys() if c in df.columns])
    return df


def one_hot_encode_remaining(df):
    """
    One-hot encodes all remaining object-dtype (categorical) columns.
    EDA found these are all low cardinality (2-8 categories) except the 
    two columns handled separately by target encoding above.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"One-hot encoding {len(cat_cols)} columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    return df

def fix_days_employed(df):
    """
    EDA found DAYS_EMPLOYED has a placeholder value of 365243 (~1000 days) 
    used to encode "not currently employed" - confirmed via NAME_INCOME_TYPE 
    (99.96% of these rows are Pensioner). 

    This must run BEFORE other feature engineering steps, since OCCUPATION_TYPE
    missingness is found to strongly overlap with this not-employed population
    (55,372 of 55,374 not-employed applicants also have missing OCCUPATION_TYPE).

    Creates IS_NOT_EMPLOYED flag, then replaces the placeholder with NaN so it 
    no longer distorts any statistic or transform applied to DAYS_EMPLOYED.
    """
    df = df.copy()

    df['IS_NOT_EMPLOYED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    print(f"IS_NOT_EMPLOYED: {df['IS_NOT_EMPLOYED'].sum()} flagged ({df['IS_NOT_EMPLOYED'].mean()*100:.2f}%)")
    print(f"DAYS_EMPLOYED placeholder replaced with NaN")

    return df

def add_ratio_features(df):
    """
    Creates ratio and interaction features that contextualize raw amounts 
    relative to the applicant's own financial profile - generally more 
    predictive than raw amounts alone in credit scoring.

    Uses AMT_INCOME_TOTAL (capped, pre-log version) and AMT_CREDIT (raw) 
    as the base amounts, since ratios should be computed on actual values, 
    not log-transformed ones.

    DAYS_BIRTH and DAYS_EMPLOYED are negative-days-before-application 
    (confirmed in EDA); converting to positive years here for interpretability.
    """
    df = df.copy()

    # --- Financial ratios ---
    # Credit-to-income: how large is the loan relative to what they earn?
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Annuity-to-income: what fraction of income goes to loan repayment?
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # Credit-to-goods-price: is the loan larger than the goods being financed?
    # (a loan amount well above goods price can signal added fees/risk)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # Annuity-to-credit: implies the effective loan term/structure
    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # --- Age and employment, converted to interpretable years ---
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25  # NaN for not-employed, as intended

    # Employment length relative to age: a higher ratio suggests employed 
    # for a larger fraction of adult life (stability signal)
    df['EMPLOYED_AGE_RATIO'] = df['EMPLOYED_YEARS'] / df['AGE_YEARS']

    # --- Family/household context ---
    # Income per family member: total income may look fine, but a large 
    # family changes what that income actually has to cover
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    new_cols = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_GOODS_RATIO',
                'ANNUITY_CREDIT_RATIO', 'AGE_YEARS', 'EMPLOYED_YEARS', 
                'EMPLOYED_AGE_RATIO', 'INCOME_PER_PERSON']

    print(f"Created {len(new_cols)} ratio/interaction features")
    print(f"Checking for inf/extreme values:")
    for col in new_cols:
        n_inf = np.isinf(df[col]).sum()
        n_null = df[col].isnull().sum()
        if n_inf > 0 or n_null > 0:
            print(f"  {col}: {n_inf} inf, {n_null} null")

    return df

def aggregate_bureau(bureau_df):
    """
    Aggregates bureau.csv (credit bureau history, multiple rows per SK_ID_CURR)
    into one row per applicant. 305,811 of 307,511 applicants (99.4%) have at
    least one bureau record; the ~1,700 without any are the dataset's truest
    "credit-invisible" cases - their absence after merging becomes a NaN flag,
    same pattern as EXT_SOURCE missingness.

    CREDIT_ACTIVE counts are pivoted into separate count columns, since 
    "Active" vs "Closed" vs "Bad debt" carry very different risk meaning - 
    averaging them together would destroy that distinction.
    """
    bureau_df = bureau_df.copy()

    # --- Pivot CREDIT_ACTIVE into count-per-status columns ---
    active_counts = pd.crosstab(bureau_df['SK_ID_CURR'], bureau_df['CREDIT_ACTIVE'])
    active_counts.columns = ['BUREAU_STATUS_' + c.upper().replace(' ', '_') + '_COUNT' for c in active_counts.columns]

    # --- Pivot CREDIT_TYPE into count-per-type columns (keep only common types to avoid sparse noise) ---
    top_credit_types = bureau_df['CREDIT_TYPE'].value_counts().head(5).index
    bureau_df['CREDIT_TYPE_GROUPED'] = bureau_df['CREDIT_TYPE'].where(
        bureau_df['CREDIT_TYPE'].isin(top_credit_types), 'Other'
    )
    type_counts = pd.crosstab(bureau_df['SK_ID_CURR'], bureau_df['CREDIT_TYPE_GROUPED'])
    type_counts.columns = ['BUREAU_TYPE_' + c.upper().replace(' ', '_') + '_COUNT' for c in type_counts.columns]

    # --- Numeric aggregations ---
    agg_funcs = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['max', 'sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
    }
    numeric_agg = bureau_df.groupby('SK_ID_CURR').agg(agg_funcs)
    numeric_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in numeric_agg.columns]

    # --- Record count per applicant (how many bureau lines do they have at all) ---
    record_count = bureau_df.groupby('SK_ID_CURR').size().rename('BUREAU_RECORD_COUNT')

    # --- Combine everything ---
    result = pd.concat([numeric_agg, active_counts, type_counts, record_count], axis=1)
    result = result.reset_index()

    print(f"Aggregated bureau.csv: {bureau_df['SK_ID_CURR'].nunique()} applicants -> {result.shape[0]} rows, {result.shape[1]} columns")

    return result


def merge_bureau_features(df, bureau_agg):
    """
    Left-merges aggregated bureau features onto the main applicant dataframe.
    Applicants with no bureau record (left join, no match) get NaN across all 
    bureau columns - this is preserved deliberately as a signal, not filled, 
    since "no bureau history at all" is itself informative (the dataset's 
    truest credit-invisible population).

    Adds HAS_BUREAU_RECORD flag explicitly for convenience/clarity, since 
    relying on implicit NaN-checking across 30+ columns is error-prone.
    """
    df = df.copy()
    n_before = df.shape[1]

    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df['HAS_BUREAU_RECORD'] = df['BUREAU_RECORD_COUNT'].notnull().astype(int)

    n_no_bureau = (df['HAS_BUREAU_RECORD'] == 0).sum()
    print(f"Merged bureau features: {df.shape[1] - n_before} new columns")
    print(f"Applicants with NO bureau record: {n_no_bureau} ({n_no_bureau/len(df)*100:.2f}%)")

    return df

def aggregate_previous_application(prev_df):
    """
    Aggregates previous_application.csv (prior Home Credit loan applications,
    multiple rows per SK_ID_CURR) into one row per applicant.

    NAME_CONTRACT_STATUS pivoted into count columns (Approved/Refused/Canceled/
    Unused offer carry very different risk meaning - a history of refusals is
    a strong distinct signal, not safely averaged with approvals).

    RATE_INTEREST_PRIMARY and RATE_INTEREST_PRIVILEGED excluded entirely - 
    99.64% missing, essentially no signal to extract.
    """
    prev_df = prev_df.copy()

    # --- Pivot NAME_CONTRACT_STATUS into count-per-status columns ---
    status_counts = pd.crosstab(prev_df['SK_ID_CURR'], prev_df['NAME_CONTRACT_STATUS'])
    status_counts.columns = ['PREV_STATUS_' + c.upper().replace(' ', '_') + '_COUNT' for c in status_counts.columns]

    # --- Numeric aggregations ---
    agg_funcs = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean', 'sum'],
        'AMT_CREDIT': ['max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'max'],
        'DAYS_LAST_DUE': ['min', 'max'],
        'DAYS_TERMINATION': ['min', 'max'],
    }
    numeric_agg = prev_df.groupby('SK_ID_CURR').agg(agg_funcs)
    numeric_agg.columns = ['PREV_' + '_'.join(col).upper() for col in numeric_agg.columns]

    # --- Record count ---
    record_count = prev_df.groupby('SK_ID_CURR').size().rename('PREV_APP_RECORD_COUNT')

    # --- Refusal rate: a directly interpretable derived feature ---
    refusal_rate = (status_counts.filter(like='REFUSED').sum(axis=1) / record_count).rename('PREV_REFUSAL_RATE')

    result = pd.concat([numeric_agg, status_counts, record_count, refusal_rate], axis=1)
    result = result.reset_index()

    print(f"Aggregated previous_application.csv: {prev_df['SK_ID_CURR'].nunique()} applicants -> {result.shape[0]} rows, {result.shape[1]} columns")

    return result


def merge_previous_application_features(df, prev_agg):
    """
    Left-merges aggregated previous_application features onto the main 
    applicant dataframe. Verifies ID overlap directly rather than assuming - 
    this table, like bureau.csv, spans both train and test applicants combined.
    """
    df = df.copy()
    n_before = df.shape[1]

    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    df['HAS_PREV_APPLICATION'] = df['PREV_APP_RECORD_COUNT'].notnull().astype(int)

    n_no_prev = (df['HAS_PREV_APPLICATION'] == 0).sum()
    print(f"Merged previous_application features: {df.shape[1] - n_before} new columns")
    print(f"Applicants with NO previous Home Credit application: {n_no_prev} ({n_no_prev/len(df)*100:.2f}%)")

    return df

def aggregate_installments(installments_df):
    """
    Aggregates installments_payments.csv (13.6M rows - the largest secondary
    table, individual installment-level records) into one row per SK_ID_CURR.

    Derives two key behavioral signals not present as raw columns:
    - DAYS_LATE: DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT (positive = paid late, 
      negative/zero = paid on time or early)
    - AMT_SHORTFALL: AMT_INSTALMENT - AMT_PAYMENT (positive = underpaid)

    2,905 rows (0.02%) have NaN for DAYS_ENTRY_PAYMENT/AMT_PAYMENT - these are 
    installments that were never paid at all, a stronger signal than "paid late".
    Flagged explicitly via MISSED_PAYMENT before computing DAYS_LATE/AMT_SHORTFALL, 
    so this signal isn't silently lost as NaN in the derived features.
    """
    df = installments_df.copy()

    df['MISSED_PAYMENT'] = df['DAYS_ENTRY_PAYMENT'].isnull().astype(int)
    df['DAYS_LATE'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['AMT_SHORTFALL'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']

    agg_funcs = {
        'DAYS_LATE': ['max', 'mean'],
        'AMT_SHORTFALL': ['max', 'mean', 'sum'],
        'MISSED_PAYMENT': ['sum'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['max', 'mean', 'sum'],
        'NUM_INSTALMENT_NUMBER': ['max'],  # proxy for how many installments this loan had
    }
    numeric_agg = df.groupby('SK_ID_CURR').agg(agg_funcs)
    numeric_agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in numeric_agg.columns]

    record_count = df.groupby('SK_ID_CURR').size().rename('INSTAL_RECORD_COUNT')

    # Late-payment rate: a directly interpretable derived feature.
    # Vectorized (no .apply()) - faster on 13.6M rows and avoids the 
    # groupby-apply-on-grouping-column deprecation warning.
    df['IS_LATE'] = (df['DAYS_LATE'] > 0).astype(int)
    late_rate = df.groupby('SK_ID_CURR')['IS_LATE'].mean().rename('INSTAL_LATE_PAYMENT_RATE')

    result = pd.concat([numeric_agg, record_count, late_rate], axis=1)
    result = result.reset_index()

    print(f"Aggregated installments_payments.csv: {df['SK_ID_CURR'].nunique()} applicants -> {result.shape[0]} rows, {result.shape[1]} columns")

    return result


def merge_installments_features(df, instal_agg):
    """
    Left-merges aggregated installment payment features onto the main 
    applicant dataframe.
    """
    df = df.copy()
    n_before = df.shape[1]

    df = df.merge(instal_agg, on='SK_ID_CURR', how='left')
    df['HAS_INSTALLMENT_HISTORY'] = df['INSTAL_RECORD_COUNT'].notnull().astype(int)

    n_no_instal = (df['HAS_INSTALLMENT_HISTORY'] == 0).sum()
    print(f"Merged installments features: {df.shape[1] - n_before} new columns")
    print(f"Applicants with NO installment history: {n_no_instal} ({n_no_instal/len(df)*100:.2f}%)")

    return df

def aggregate_pos_cash(pos_df):
    """
    Aggregates POS_CASH_balance.csv (monthly POS/cash loan balance snapshots)
    into one row per SK_ID_CURR.

    Verified against raw data:
    - 9 NAME_CONTRACT_STATUS categories; Active (91.5%) + Completed (7.4%) dominate,
      remaining 7 categories all <1% each -> grouped into 'Other'.
    - SK_DPD / SK_DPD_DEF have no missing values, no placeholder-value trap.
    - MONTHS_BALANCE must be sorted ascending per loan before taking 'last' snapshot,
      since raw file is not guaranteed pre-sorted.
    """
    df = pos_df.copy()

    # Sort so that within each SK_ID_PREV, rows go from earliest to most recent MONTHS_BALANCE.
    # Required before any .agg('last') call -- confirmed via direct inspection that the
    # raw file order cannot be trusted.
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    # Group rare statuses into 'Other' -- Active/Completed are the only categories >1%
    top_statuses = ['Active', 'Completed']
    df['STATUS_GROUPED'] = df['NAME_CONTRACT_STATUS'].where(
        df['NAME_CONTRACT_STATUS'].isin(top_statuses), 'Other'
    )
    status_counts = pd.crosstab(df['SK_ID_CURR'], df['STATUS_GROUPED'])
    status_counts.columns = [f'POS_STATUS_{c.upper()}_COUNT' for c in status_counts.columns]
    status_counts = status_counts.reset_index()

    # DPD-based flags -- vectorized rate computation, same fix applied to installments late-rate
    df['IS_DPD'] = (df['SK_DPD'] > 0).astype(int)
    df['IS_DPD_DEF'] = (df['SK_DPD_DEF'] > 0).astype(int)

    numeric_agg = df.groupby('SK_ID_CURR').agg(
        POS_MONTHS_BALANCE_COUNT=('MONTHS_BALANCE', 'count'),
        POS_MONTHS_BALANCE_MIN=('MONTHS_BALANCE', 'min'),
        POS_CNT_INSTALMENT_MEAN=('CNT_INSTALMENT', 'mean'),
        POS_CNT_INSTALMENT_FUTURE_MEAN=('CNT_INSTALMENT_FUTURE', 'mean'),
        POS_CNT_INSTALMENT_FUTURE_LAST=('CNT_INSTALMENT_FUTURE', 'last'),  # valid now that sort is enforced above
        POS_SK_DPD_MEAN=('SK_DPD', 'mean'),
        POS_SK_DPD_MAX=('SK_DPD', 'max'),
        POS_SK_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        POS_SK_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
        POS_DPD_RATE=('IS_DPD', 'mean'),
        POS_DPD_DEF_RATE=('IS_DPD_DEF', 'mean'),
        POS_LOAN_COUNT=('SK_ID_PREV', 'nunique'),
    ).reset_index()

    pos_agg = numeric_agg.merge(status_counts, on='SK_ID_CURR', how='left')

    print(f"POS_CASH aggregated: {pos_agg.shape[0]} applicants, {pos_agg.shape[1]} columns")
    return pos_agg


def merge_pos_cash_features(df, pos_agg):
    """Left-merges POS_CASH aggregated features onto train_fe. Adds HAS_POS_CASH_RECORD flag."""
    before_cols = df.shape[1]
    merged = df.merge(pos_agg, on='SK_ID_CURR', how='left')
    merged['HAS_POS_CASH_RECORD'] = merged['POS_LOAN_COUNT'].notna().astype(int)

    n_missing = merged['HAS_POS_CASH_RECORD'].eq(0).sum()
    print(f"Applicants without POS_CASH record: {n_missing} ({n_missing/len(merged)*100:.2f}%)")
    print(f"Shape: {before_cols} -> {merged.shape[1]}")
    return merged

def aggregate_bureau_balance(balance_df, bureau_df):
    """
    Aggregates bureau_balance.csv (27,299,925 monthly status rows, keyed by
    SK_ID_BUREAU only) up to one row per SK_ID_CURR via a two-hop merge
    through bureau.csv's SK_ID_BUREAU -> SK_ID_CURR mapping.

    Verified against raw data:
    - 54.89% of bureau.csv credit lines (942,074 of 1,716,428) have no
      balance history at all -- expected dataset characteristic, not an error.
    - STATUS has 8 categories: '0'-'5' (DPD severity buckets, 0=current,
      5=worst/written off), 'C' (closed), 'X' (status unknown that month).
    - The 43,041 SK_ID_BUREAU in bureau_balance.csv with no match in
      bureau.csv are dropped naturally by the merge direction used here.
    """
    df = balance_df.copy()

    # Pivot STATUS into count-per-category columns
    status_counts = pd.crosstab(df['SK_ID_BUREAU'], df['STATUS'])
    status_counts.columns = [f'BB_STATUS_{c}_COUNT' for c in status_counts.columns]
    status_counts = status_counts.reset_index()

    # Numeric DPD severity - only defined for '0'-'5', NaN for lines with only C/X months
    numeric_mask = df['STATUS'].isin(['0', '1', '2', '3', '4', '5'])
    df.loc[numeric_mask, 'STATUS_NUM'] = df.loc[numeric_mask, 'STATUS'].astype(int)
    max_dpd = df.groupby('SK_ID_BUREAU')['STATUS_NUM'].max().rename('BB_MAX_DPD_STATUS')

    # Ever-late rate (status 1-5) per bureau line, vectorized
    df['IS_LATE'] = df['STATUS'].isin(['1', '2', '3', '4', '5']).astype(int)
    late_rate = df.groupby('SK_ID_BUREAU')['IS_LATE'].mean().rename('BB_LATE_RATE')

    months_agg = df.groupby('SK_ID_BUREAU').agg(
        BB_MONTHS_COUNT=('MONTHS_BALANCE', 'count'),
        BB_MONTHS_MIN=('MONTHS_BALANCE', 'min'),
    ).reset_index()

    # Combine to one row per SK_ID_BUREAU
    balance_agg = months_agg.merge(status_counts, on='SK_ID_BUREAU', how='left')
    balance_agg = balance_agg.merge(max_dpd, on='SK_ID_BUREAU', how='left')
    balance_agg = balance_agg.merge(late_rate, on='SK_ID_BUREAU', how='left')

    # Hop 2: attach SK_ID_CURR via bureau.csv, then roll up to applicant level
    bureau_link = bureau_df[['SK_ID_CURR', 'SK_ID_BUREAU']]
    linked = bureau_link.merge(balance_agg, on='SK_ID_BUREAU', how='left')

    applicant_agg = linked.groupby('SK_ID_CURR').agg(
        BB_MONTHS_COUNT_MEAN=('BB_MONTHS_COUNT', 'mean'),
        BB_MONTHS_COUNT_SUM=('BB_MONTHS_COUNT', 'sum'),
        BB_LATE_RATE_MEAN=('BB_LATE_RATE', 'mean'),
        BB_LATE_RATE_MAX=('BB_LATE_RATE', 'max'),
        BB_MAX_DPD_STATUS_MAX=('BB_MAX_DPD_STATUS', 'max'),
    ).reset_index()

    print(f"bureau_balance aggregated: {applicant_agg.shape[0]} applicants, {applicant_agg.shape[1]} columns")
    return applicant_agg


def merge_bureau_balance_features(df, bb_agg):
    """Left-merges bureau_balance-derived features onto train_fe. Adds HAS_BUREAU_BALANCE flag."""
    before_cols = df.shape[1]
    merged = df.merge(bb_agg, on='SK_ID_CURR', how='left')
    merged['HAS_BUREAU_BALANCE'] = merged['BB_MONTHS_COUNT_MEAN'].notna().astype(int)

    n_missing = merged['HAS_BUREAU_BALANCE'].eq(0).sum()
    print(f"Applicants without bureau_balance history: {n_missing} ({n_missing/len(merged)*100:.2f}%)")
    print(f"Shape: {before_cols} -> {merged.shape[1]}")
    return merged

def aggregate_credit_card_balance(cc_df):
    df = cc_df.copy()

    # Verified structural zeros (checked against AMT_DRAWINGS_CURRENT / AMT_PAYMENT_TOTAL_CURRENT / per-loan null pattern)
    zero_fill_cols = [
        'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
        'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
        'AMT_PAYMENT_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'AMT_INST_MIN_REGULARITY'
    ]
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    # NAME_CONTRACT_STATUS grouping: Active/Completed/Other (same as POS_CASH)
    df['CONTRACT_STATUS_GROUPED'] = df['NAME_CONTRACT_STATUS'].apply(
        lambda x: x if x in ['Active', 'Completed'] else 'Other'
    )

    # Explicit sort before any .last() snapshot — raw file confirmed unsorted
    df = df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])

    # Utilization ratio, guarding against zero/NaN credit limit
    df['UTILIZATION_RATIO'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)

    # --- Aggregate to one row per SK_ID_CURR ---
    agg = df.groupby('SK_ID_CURR').agg(
        CC_MONTHS_COUNT=('MONTHS_BALANCE', 'count'),
        CC_BALANCE_MEAN=('AMT_BALANCE', 'mean'),
        CC_BALANCE_LAST=('AMT_BALANCE', 'last'),
        CC_CREDIT_LIMIT_MEAN=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        CC_CREDIT_LIMIT_LAST=('AMT_CREDIT_LIMIT_ACTUAL', 'last'),
        CC_UTILIZATION_MEAN=('UTILIZATION_RATIO', 'mean'),
        CC_UTILIZATION_LAST=('UTILIZATION_RATIO', 'last'),
        CC_DRAWINGS_ATM_SUM=('AMT_DRAWINGS_ATM_CURRENT', 'sum'),
        CC_DRAWINGS_POS_SUM=('AMT_DRAWINGS_POS_CURRENT', 'sum'),
        CC_DRAWINGS_OTHER_SUM=('AMT_DRAWINGS_OTHER_CURRENT', 'sum'),
        CC_PAYMENT_CURRENT_MEAN=('AMT_PAYMENT_CURRENT', 'mean'),
        CC_RECEIVABLE_PRINCIPAL_LAST=('AMT_RECEIVABLE_PRINCIPAL', 'last'),
        CC_TOTAL_RECEIVABLE_LAST=('AMT_TOTAL_RECEIVABLE', 'last'),
        CC_DPD_MEAN=('SK_DPD', 'mean'),
        CC_DPD_MAX=('SK_DPD', 'max'),
        CC_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        CC_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
        CC_COMPLETED_RATE=('CONTRACT_STATUS_GROUPED', lambda s: (s == 'Completed').mean()),
    ).reset_index()

    print(f"credit_card_balance aggregated: {agg.shape[0]} applicants, {agg.shape[1]} columns")
    return agg

def merge_credit_card_balance_features(df, cc_agg):
    before_shape = df.shape
    merged = df.merge(cc_agg, on='SK_ID_CURR', how='left')
    no_cc_record = merged['CC_MONTHS_COUNT'].isnull().sum()
    print(f"Applicants with no credit_card_balance record: {no_cc_record} ({no_cc_record/len(merged)*100:.2f}%)")
    print(f"Shape: {before_shape} -> {merged.shape}")
    return merged