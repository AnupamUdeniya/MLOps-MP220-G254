
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from mlflow import MlflowClient

from src.data_cleaning import clean_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """Load the project configuration."""

    config_path = PROJECT_ROOT / "params.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_holdout_predictions():
    """Generate predictions using the registered MLflow champion."""

    config = load_config()

    # Load paths and feature definitions.
    holdout_path = PROJECT_ROOT / config["data"]["holdout_path"]

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    feature_columns = numeric_features + categorical_features

    registered_name = config["mlflow"]["registered_model_name"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    output_path = (
        PROJECT_ROOT / config["output"]["predictions_path"]
    )

    # Initialize MLflow.
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    # Verify that the champion alias exists.
    champion = client.get_model_version_by_alias(
        registered_name,
        "champion"
    )

    print("Registered model:", registered_name)
    print("Champion version:", champion.version)
    print("Champion run ID:", champion.run_id)

    # Load the champion model from MLflow.
    model_uri = f"models:/{registered_name}@champion"

    model = mlflow.pyfunc.load_model(model_uri)

    print("\nChampion model loaded successfully.")

    # Load original holdout data.
    raw_holdout = pd.read_csv(holdout_path)

    print("\nOriginal holdout shape:", raw_holdout.shape)

    if "record_id" not in raw_holdout.columns:
        raise ValueError("Holdout dataset is missing record_id.")

    if raw_holdout["record_id"].isnull().any():
        raise ValueError("Holdout contains missing record IDs.")

    if raw_holdout["record_id"].duplicated().any():
        raise ValueError("Holdout contains duplicate record IDs.")

    missing_columns = [
        column for column in feature_columns
        if column not in raw_holdout.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    # Apply the existing Data Lead cleaning function.
    cleaned_holdout = clean_data(raw_holdout.copy())

    # Select only the eight model input features.
    X_holdout = cleaned_holdout[feature_columns].copy()

    if X_holdout.isnull().any().any():
        raise ValueError(
            "Missing feature values remain after cleaning."
        )

    # Validate numeric features.
    numeric_values = X_holdout[numeric_features].to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Holdout contains non-finite numeric values."
        )

    print("Features supplied to model:", feature_columns)
    print("Records to predict:", len(X_holdout))

    # Generate predictions.
    predictions = np.asarray(
        model.predict(X_holdout)
    ).reshape(-1)
    # Solar energy production cannot be negative.
    predictions = np.maximum(predictions, 0.0)

    if len(predictions) != len(raw_holdout):
        raise RuntimeError(
            "Prediction count does not match holdout row count."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    # Construct the exact required output.
    output = pd.DataFrame({
        "record_id": raw_holdout["record_id"].to_numpy(),
        "prediction": predictions.astype(float)
    })

    # Save predictions.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(output_path, index=False)

    # Read back and verify the saved file.
    saved_output = pd.read_csv(output_path)

    expected_columns = ["record_id", "prediction"]

    if list(saved_output.columns) != expected_columns:
        raise RuntimeError(
            "Output CSV does not have the required columns."
        )

    if len(saved_output) != 250:
        raise RuntimeError(
            f"Expected 250 predictions, got {len(saved_output)}."
        )

    if saved_output["prediction"].isnull().any():
        raise RuntimeError(
            "Saved predictions contain missing values."
        )

    print("\n========== HOLDOUT PREDICTIONS COMPLETE ==========")
    print("Output file:", output_path)
    print("Prediction count:", len(saved_output))
    print("Prediction MAE: Not available until ground truth is supplied.")
    print("\nPrediction summary:")
    print(saved_output["prediction"].describe())

    print("\nFirst five predictions:")
    print(saved_output.head().to_string(index=False))

    return output_path


if __name__ == "__main__":
    generate_holdout_predictions()
