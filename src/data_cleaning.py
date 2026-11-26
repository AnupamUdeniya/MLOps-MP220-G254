from pathlib import Path
import pandas as pd
import yaml


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the solar energy dataset by fixing the identified issues:
    1. Duplicate rows: drop exact duplicate records if any exist.
    2. Unit mixup: Old City panel_area_m2 was recorded in cm2 (factor 10000).
    3. Sentinel values: sunshine_hours == -999 imputed with valid median.
    4. Target typos: energy_kwh > 100 divided by 10 (extra zero typed).
    """
    cleaned = df.drop_duplicates().copy()

    # 1. Fix unit mixup in panel_area_m2 for Old City
    if "area" in cleaned.columns and "panel_area_m2" in cleaned.columns:
        mask = (cleaned["area"] == "Old City") & (cleaned["panel_area_m2"] > 1000)
        cleaned.loc[mask, "panel_area_m2"] = cleaned.loc[mask, "panel_area_m2"] / 10000.0

    # 2. Fix sentinel values in sunshine_hours (-999)
    if "sunshine_hours" in cleaned.columns:
        valid_median = cleaned.loc[cleaned["sunshine_hours"] != -999, "sunshine_hours"].median()
        cleaned.loc[cleaned["sunshine_hours"] == -999, "sunshine_hours"] = valid_median

    # 3. Fix target typos in energy_kwh (values > 100 are 10x too large)
    if "energy_kwh" in cleaned.columns:
        typo_mask = cleaned["energy_kwh"] > 100
        cleaned.loc[typo_mask, "energy_kwh"] = cleaned.loc[typo_mask, "energy_kwh"] / 10.0

    return cleaned


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """Verify that known data anomalies have been removed after cleaning."""
    if df.duplicated().any():
        raise ValueError("Duplicate rows still present after cleaning.")
    if "sunshine_hours" in df.columns and (df["sunshine_hours"] == -999).any():
        raise ValueError("Sentinel value -999 still present in sunshine_hours.")
    if "area" in df.columns and "panel_area_m2" in df.columns:
        bad_area = (df["area"] == "Old City") & (df["panel_area_m2"] > 1000)
        if bad_area.any():
            raise ValueError("Unconverted panel_area_m2 values remain for Old City.")
    if "energy_kwh" in df.columns and (df["energy_kwh"] > 100).any():
        raise ValueError("Uncorrected target typos (>100 kWh) remain in energy_kwh.")
    return True


def clean_from_params(params_path: str = "params.yaml") -> pd.DataFrame:
    """Load raw train path from params.yaml, clean it, and validate the output."""
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    raw_path = Path(params["data"]["train_path"])
    cleaned = clean_data(pd.read_csv(raw_path))
    validate_cleaned_data(cleaned)
    return cleaned


if __name__ == "__main__":
    cleaned_df = clean_from_params()
    print("Cleaned shape:", cleaned_df.shape)
    print(
        "Old City panel_area range:",
        cleaned_df[cleaned_df["area"] == "Old City"]["panel_area_m2"].min(),
        "to",
        cleaned_df[cleaned_df["area"] == "Old City"]["panel_area_m2"].max(),
    )
    print("Sunshine min:", cleaned_df["sunshine_hours"].min())
    print("Max energy_kwh:", cleaned_df["energy_kwh"].max())
