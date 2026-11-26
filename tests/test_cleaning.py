import pandas as pd
import pytest
from src.data_cleaning import clean_data, validate_cleaned_data
from src.data_checks import (
    check_leaking_column,
    check_sentinel_value,
    check_target_typos,
    check_unit_mixup,
)


@pytest.fixture
def sample_dirty_data():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "panel_area_m2": [348000.0, 45.0, 50.0],
        "sunshine_hours": [-999.0, 8.0, 6.0],
        "temp_c": [35.0, 30.0, 25.0],
        "dust_days": [10, 15, 20],
        "panel_age_yr": [5.0, 2.0, 8.0],
        "cloud_cover_pct": [20, 40, 50],
        "roof_type": ["flat", "tin", "sloped"],
        "area": ["Old City", "Industrial", "New Colony"],
        "energy_kwh": [250.0, 25.0, 30.0],
    })


def test_clean_unit_mixup(sample_dirty_data):
    cleaned = clean_data(sample_dirty_data)
    # 348000.0 should be divided by 10000.0 -> 34.8
    assert cleaned.loc[0, "panel_area_m2"] == pytest.approx(34.8, 0.01)


def test_clean_sentinel_value(sample_dirty_data):
    cleaned = clean_data(sample_dirty_data)
    # -999.0 should be replaced with valid median (7.0 from [8.0, 6.0])
    assert cleaned.loc[0, "sunshine_hours"] > 0
    assert (cleaned["sunshine_hours"] == -999).sum() == 0


def test_clean_target_typo(sample_dirty_data):
    cleaned = clean_data(sample_dirty_data)
    # 250.0 should be divided by 10.0 -> 25.0
    assert cleaned.loc[0, "energy_kwh"] == pytest.approx(25.0, 0.01)


def test_clean_does_not_mutate_input(sample_dirty_data):
    original_area = sample_dirty_data.loc[0, "panel_area_m2"]
    _ = clean_data(sample_dirty_data)
    assert sample_dirty_data.loc[0, "panel_area_m2"] == original_area


def test_validate_cleaned_data_passes(sample_dirty_data):
    cleaned = clean_data(sample_dirty_data)
    assert validate_cleaned_data(cleaned) is True


def test_clean_drops_duplicates(sample_dirty_data):
    with_dup = pd.concat([sample_dirty_data, sample_dirty_data.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(with_dup)
    assert len(cleaned) == 3


def test_validate_cleaned_data_rejects_dirty(sample_dirty_data):
    with pytest.raises(ValueError):
        validate_cleaned_data(sample_dirty_data)


def test_data_checks_detect_anomalies(sample_dirty_data):
    holdout = sample_dirty_data.drop(columns=["energy_kwh"])
    assert check_leaking_column(sample_dirty_data, holdout)["present"] is False
    assert check_unit_mixup(sample_dirty_data)["group_value"] == "Old City"
    assert check_sentinel_value(sample_dirty_data)["value"] == -999
    assert check_target_typos(sample_dirty_data)["count"] == 1
