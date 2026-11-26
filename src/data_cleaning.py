import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the solar energy dataset by fixing the identified issues:
    1. Unit mixup: Old City panel_area_m2 was recorded in cm2 (factor 10000).
    2. Sentinel values: sunshine_hours == -999 imputed with valid median.
    3. Target typos: energy_kwh > 100 divided by 10 (extra zero typed).
    """
    cleaned = df.copy()
    
    # 1. Fix unit mixup in panel_area_m2 for Old City
    if 'area' in cleaned.columns and 'panel_area_m2' in cleaned.columns:
        mask = (cleaned['area'] == 'Old City') & (cleaned['panel_area_m2'] > 1000)
        cleaned.loc[mask, 'panel_area_m2'] = cleaned.loc[mask, 'panel_area_m2'] / 10000.0
        
    # 2. Fix sentinel values in sunshine_hours (-999)
    if 'sunshine_hours' in cleaned.columns:
        valid_median = cleaned.loc[cleaned['sunshine_hours'] != -999, 'sunshine_hours'].median()
        cleaned.loc[cleaned['sunshine_hours'] == -999, 'sunshine_hours'] = valid_median
        
    # 3. Fix target typos in energy_kwh (values > 100 are 10x too large)
    if 'energy_kwh' in cleaned.columns:
        typo_mask = cleaned['energy_kwh'] > 100
        cleaned.loc[typo_mask, 'energy_kwh'] = cleaned.loc[typo_mask, 'energy_kwh'] / 10.0
        
    return cleaned


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """Verify that known data anomalies have been removed after cleaning."""
    if "sunshine_hours" in df.columns and (df["sunshine_hours"] == -999).any():
        raise ValueError("Sentinel value -999 still present in sunshine_hours.")
    if "area" in df.columns and "panel_area_m2" in df.columns:
        bad_area = (df["area"] == "Old City") & (df["panel_area_m2"] > 1000)
        if bad_area.any():
            raise ValueError("Unconverted panel_area_m2 values remain for Old City.")
    if "energy_kwh" in df.columns and (df["energy_kwh"] > 100).any():
        raise ValueError("Uncorrected target typos (>100 kWh) remain in energy_kwh.")
    return True


if __name__ == "__main__":
    df = pd.read_csv("data/raw/train.csv")
    cleaned_df = clean_data(df)
    print("Cleaned shape:", cleaned_df.shape)
    print("Old City panel_area range:", cleaned_df[cleaned_df['area'] == 'Old City']['panel_area_m2'].min(), "to", cleaned_df[cleaned_df['area'] == 'Old City']['panel_area_m2'].max())
    print("Sunshine min:", cleaned_df['sunshine_hours'].min())
    print("Max energy_kwh:", cleaned_df['energy_kwh'].max())
