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