
import hashlib

import numpy as np
import pandas as pd
import yaml

import src.model_tracking as tracking

from src.model_training import build_pipeline


def test_ridge_pipeline_trains_and_predicts():
    """Verify preprocessing and Ridge regression work together."""

    config = {
        "features": {
            "numeric": [
                "panel_area_m2",
                "sunshine_hours",
                "temp_c",
                "dust_days",
                "panel_age_yr",
                "cloud_cover_pct"
            ],
            "categorical": [
                "roof_type",
                "area"
            ]
        }
    }

    model_config = {
        "name": "ridge_test",
        "type": "ridge",
        "params": {"alpha": 1.0}
    }

    X = pd.DataFrame({
        "panel_area_m2": [10, 20, 30, 40, 50, 60, 70, 80],
        "sunshine_hours": [2, 3, 4, 5, 6, 7, 8, 9],
        "temp_c": [20, 21, 22, 23, 24, 25, 26, 27],
        "dust_days": [1, 2, 3, 4, 5, 6, 7, 8],
        "panel_age_yr": [1, 2, 3, 4, 5, 6, 7, 8],
        "cloud_cover_pct": [10, 20, 30, 40, 50, 60, 70, 80],
        "roof_type": [
            "flat", "sloped", "tin", "flat",
            "sloped", "tin", "flat", "sloped"
        ],
        "area": [
            "Old City", "New Colony", "Industrial",
            "Old City", "New Colony", "Industrial",
            "Old City", "New Colony"
        ]
    })

    y = pd.Series([10, 12, 15, 18, 20, 23, 25, 28])

    pipeline = build_pipeline(config, model_config)

    pipeline.fit(X, y)

    predictions = pipeline.predict(X.head(3))

    assert len(predictions) == 3
    assert np.isfinite(predictions).all()


def test_original_data_sha256(tmp_path):
    """Verify SHA-256 calculation for an input file."""

    data = b"record_id,energy_kwh\nR001,25.5\n"

    data_path = tmp_path / "train.csv"
    data_path.write_bytes(data)

    expected_hash = hashlib.sha256(data).hexdigest()

    actual_hash = tracking.calculate_data_sha256(data_path)

    assert actual_hash == expected_hash


def test_tracking_update_preserves_other_findings(
    tmp_path,
    monkeypatch
):
    """Verify MLflow updates do not overwrite Data or Ops findings."""

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    findings_path = reports_dir / "findings.yaml"

    original_findings = {
        "group_id": "G254",
        "data_issues": {
            "unit_mixup": {"present": True},
            "sentinel_value": {"present": True}
        },
        "tracking": {
            "experiment_name": None,
            "best_run_id": None,
            "registered_model_name": None,
            "champion_version": None
        },
        "monitoring": {
            "psi": {"panel_area_m2": None},
            "decision": None
        }
    }

    findings_path.write_text(
        yaml.safe_dump(original_findings, sort_keys=False),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        tracking,
        "PROJECT_ROOT",
        tmp_path
    )

    tracking.update_tracking_findings(
        experiment_name="mp220_solar_prediction",
        best_run_id="test_run_123",
        registered_model_name="mp220_solar_model",
        champion_version=1
    )

    updated = yaml.safe_load(
        findings_path.read_text(encoding="utf-8")
    )

    assert updated["data_issues"] == original_findings["data_issues"]
    assert updated["monitoring"] == original_findings["monitoring"]

    assert updated["tracking"]["experiment_name"] == (
        "mp220_solar_prediction"
    )

    assert updated["tracking"]["best_run_id"] == "test_run_123"

    assert updated["tracking"]["registered_model_name"] == (
        "mp220_solar_model"
    )

    assert updated["tracking"]["champion_version"] == 1
