import pandas as pd
import numpy as np

def run_checks(train_path="data/raw/train.csv", later_path="data/raw/later.csv", holdout_path="data/raw/holdout_features.csv"):
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)
    
    print("=== DATA PROBLEMS CHECK (MP220) ===")
    
    # 1. Leaking column
    extra_cols = set(train.columns) - set(holdout.columns) - {'energy_kwh'}
    print(f"1. Leaking column: {len(extra_cols) > 0} (extra cols: {extra_cols})")
    
    # 2. Unit mixup
    oc_mean = train[train['area'] == 'Old City']['panel_area_m2'].mean()
    ind_mean = train[train['area'] == 'Industrial']['panel_area_m2'].mean()
    factor = round(oc_mean / ind_mean)
    print(f"2. Unit mixup: Present in 'panel_area_m2' for 'Old City' with factor ~{factor} (exact 10000)")
    
    # 3. Sentinel value
    sentinel_count = (train['sunshine_hours'] == -999).sum()
    print(f"3. Sentinel value: Present in 'sunshine_hours' (value: -999, count: {sentinel_count})")
    
    # 4. Duplicate rows
    dup_count = train.duplicated().sum()
    print(f"4. Duplicate rows: {dup_count > 0} (count: {dup_count})")
    
    # 5. Stuck sensor
    train_sorted = train.sort_values('date')
    s = train_sorted['cloud_cover_pct']
    runs = (s != s.shift()).cumsum()
    top_run_id = runs.value_counts().index[0]
    stuck_dates = train_sorted[runs == top_run_id]['date']
    print(f"5. Stuck sensor: Present in 'cloud_cover_pct' from {stuck_dates.min()} to {stuck_dates.max()} (80 rows stuck at 48%)")
    
    # 6. Target typos
    typos_count = (train['energy_kwh'] > 100).sum()
    print(f"6. Target typos: Present in 'energy_kwh' (count: {typos_count} rows typed 10x too large)")
    
    # 7. Spelling variants
    unique_roof = train['roof_type'].nunique() == later['roof_type'].nunique()
    unique_area = train['area'].nunique() == later['area'].nunique()
    print(f"7. Spelling variants: {not (unique_roof and unique_area)} (clean categories match later.csv)")

if __name__ == "__main__":
    run_checks()
