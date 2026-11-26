import pandas as pd


def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str = "energy_kwh") -> dict:
    """Detect columns present in train but missing from holdout (excluding target)."""
    extra_cols = sorted(set(train.columns) - set(holdout.columns) - {target})
    return {"present": len(extra_cols) > 0, "column": extra_cols[0] if extra_cols else None}


def check_unit_mixup(train: pd.DataFrame, group_col: str = "area") -> dict:
    """Detect if any group recorded a numeric feature in a wildly different unit."""
    numeric_cols = [
        "panel_area_m2",
        "sunshine_hours",
        "temp_c",
        "dust_days",
        "panel_age_yr",
        "cloud_cover_pct",
    ]
    for col in numeric_cols:
        if col not in train.columns:
            continue
        means = train.groupby(group_col)[col].mean()
        median_group = means.median()
        if median_group == 0:
            continue
        ratios = means / median_group
        max_group = ratios.idxmax()
        if ratios[max_group] > 5:
            return {
                "present": True,
                "column": col,
                "group_value": max_group,
                "approx_factor": round(float(ratios[max_group])),
            }
    return {"present": False, "column": None, "group_value": None, "approx_factor": None}


def check_sentinel_value(train: pd.DataFrame, sentinels=(-1, -999, 999, 9999)) -> dict:
    """Detect sentinel placeholder values in numeric columns."""
    for col in train.select_dtypes(include="number").columns:
        for val in sentinels:
            count = int((train[col] == val).sum())
            if count > 0:
                return {"present": True, "column": col, "value": val, "count": count}
    return {"present": False, "column": None, "value": None, "count": None}


def check_stuck_sensor(train: pd.DataFrame, min_run: int = 15) -> dict:
    """Detect a sensor column stuck on the exact same value across consecutive dates."""
    train_sorted = train.sort_values("date").reset_index(drop=True)
    for col in ["panel_area_m2", "sunshine_hours", "temp_c", "dust_days", "panel_age_yr", "cloud_cover_pct"]:
        if col not in train_sorted.columns:
            continue
        series = train_sorted[col]
        runs = (series != series.shift()).cumsum()
        counts = runs.value_counts()
        top_run_id = counts.index[0]
        top_count = int(counts.iloc[0])
        if top_count >= min_run:
            stuck_rows = train_sorted[runs == top_run_id]
            return {
                "present": True,
                "column": col,
                "value": stuck_rows[col].iloc[0].item() if hasattr(stuck_rows[col].iloc[0], "item") else stuck_rows[col].iloc[0],
                "run_rows": top_count,
                "start_date": str(stuck_rows["date"].min()),
                "end_date": str(stuck_rows["date"].max()),
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train: pd.DataFrame, target: str = "energy_kwh", threshold: float = 100.0) -> dict:
    """Detect extreme 10x target typos above the normal range."""
    count = int((train[target] > threshold).sum())
    return {"present": count > 0, "count": count if count > 0 else None}


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    """Compare categorical column levels between train and clean later data."""
    for col in ["roof_type", "area"]:
        if col in train.columns and col in later.columns:
            if set(train[col].dropna().unique()) != set(later[col].dropna().unique()):
                return {"present": True, "column": col, "true_levels": int(later[col].nunique())}
    return {"present": False, "column": None, "true_levels": None}


def run_checks(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> dict:
    """Run all seven data quality checks and print a structured report."""
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    dup_count = int(train.duplicated().sum())
    results = {
        "leaking_column": check_leaking_column(train, holdout),
        "unit_mixup": check_unit_mixup(train),
        "sentinel_value": check_sentinel_value(train),
        "duplicate_rows": {"present": dup_count > 0, "count": dup_count if dup_count > 0 else None},
        "stuck_sensor": check_stuck_sensor(train),
        "target_typos": check_target_typos(train),
        "spelling_variants": check_spelling_variants(train, later),
    }

    print("=== DATA PROBLEMS CHECK (MP220) ===")
    for idx, (name, info) in enumerate(results.items(), start=1):
        print(f"{idx}. {name}: {info}")
    return results


if __name__ == "__main__":
    run_checks()
