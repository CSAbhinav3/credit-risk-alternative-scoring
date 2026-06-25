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