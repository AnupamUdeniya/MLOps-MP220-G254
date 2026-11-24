
from pathlib import Path

import yaml
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.data_cleaning import clean_data

from src.model_tracking import (
    calculate_data_sha256,
    initialize_mlflow,
    log_candidate_run,
    register_v1,
    update_tracking_findings
)


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """Load model configuration from params.yaml."""

    config_path = PROJECT_ROOT / "params.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_cleaned_data(config):
    """Load original training data and apply Data Lead cleaning."""

    data_path = PROJECT_ROOT / config["data"]["train_path"]

    raw_data = pd.read_csv(data_path)

    # Applying Anupam's existing cleaning function.
    cleaned_data = clean_data(raw_data)

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    target = config["data"]["target"]

    required_columns = (
        numeric_features + categorical_features + [target]
    )

    missing_columns = [
        column for column in required_columns
        if column not in cleaned_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = cleaned_data[numeric_features + categorical_features]
    y = cleaned_data[target]

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Missing values remain after cleaning.")

    return X, y, raw_data.shape


def build_preprocessor(config):
    """Create preprocessing for numeric and categorical features."""

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer,
             categorical_features)
        ]
    )

    return preprocessor


def build_model(model_config):
    """Construct a regression model from params.yaml."""

    model_type = model_config["type"]
    model_params = model_config["params"]

    if model_type == "ridge":
        return Ridge(**model_params)

    elif model_type == "random_forest":
        return RandomForestRegressor(**model_params)

    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(**model_params)

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )


def build_pipeline(config, model_config):
    """Combine preprocessing and regression model."""

    preprocessor = build_preprocessor(config)
    estimator = build_model(model_config)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    return pipeline



def train_models():
    """Train, evaluate, track and register the best v1 model."""

    config = load_config()

    print("\nLoading and cleaning training data...")

    X, y, raw_shape = load_cleaned_data(config)

    print("Original training shape:", raw_shape)
    print("Training samples:", len(X))

    # Hash the original CSV, before data cleaning.
    train_path = PROJECT_ROOT / config["data"]["train_path"]
    data_sha256 = calculate_data_sha256(train_path)

    print("Original data SHA-256:", data_sha256)

    # Initialize MLflow.
    client, experiment_id = initialize_mlflow(config)

    experiment_name = config["mlflow"]["experiment_name"]
    registered_model_name = config["mlflow"]["registered_model_name"]

    print("MLflow experiment:", experiment_name)

    # Reproducible training-validation split.
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print("Training split:", len(X_train))
    print("Validation split:", len(X_val))

    results = []
    best_mae = float("inf")
    best_config = None
    best_run_id = None

    print("\n========== MODEL COMPARISON ==========")

    for model_config in config["models"]:

        model_name = model_config["name"]

        print(f"\nTraining: {model_name}")

        pipeline = build_pipeline(config, model_config)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)

        # Log this candidate experiment in MLflow.
        run_id = log_candidate_run(
            experiment_id=experiment_id,
            model_config=model_config,
            pipeline=pipeline,
            validation_mae=mae,
            data_sha256=data_sha256,
            test_size=test_size,
            random_state=random_state
        )

        results.append({
            "model": model_name,
            "mae": float(mae),
            "run_id": run_id
        })

        print(f"Validation MAE: {mae:.5f}")
        print(f"MLflow Run ID: {run_id}")

        if mae < best_mae:
            best_mae = mae
            best_config = model_config
            best_run_id = run_id

    results.sort(key=lambda item: item["mae"])

    print("\n========== RESULTS ==========")

    for result in results:
        print(
            f"{result['model']}: "
            f"MAE = {result['mae']:.5f}"
        )

    print("\nBest model:", best_config["name"])
    print("Best validation MAE:", round(best_mae, 5))
    print("Best candidate run ID:", best_run_id)

    # Refit the selected model on all cleaned records.
    print("\nRefitting selected model on all cleaned data...")

    final_model = build_pipeline(config, best_config)
    final_model.fit(X, y)

    # Save the complete preprocessing and model pipeline.
    model_path = PROJECT_ROOT / config["output"]["model_path"]

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(final_model, model_path)

    print("Model saved:", model_path)

    # Register the final v1 model.
    print("\n========== MODEL REGISTRATION ==========")

    champion_version, registration_run_id = register_v1(
        config=config,
        client=client,
        experiment_id=experiment_id,
        final_model=final_model,
        X=X,
        data_sha256=data_sha256,
        best_model_name=best_config["name"],
        best_validation_mae=best_mae
    )

    # Update only the tracking section of findings.yaml.
    update_tracking_findings(
        experiment_name=experiment_name,
        best_run_id=best_run_id,
        registered_model_name=registered_model_name,
        champion_version=champion_version
    )

    print("\n========== TRAINING COMPLETE ==========")

    return {
        "results": results,
        "best_model": best_config["name"],
        "best_validation_mae": float(best_mae),
        "best_run_id": best_run_id,
        "registration_run_id": registration_run_id,
        "registered_model_name": registered_model_name,
        "champion_version": str(champion_version),
        "model_path": str(model_path)
    }

if __name__ == "__main__":
    training_results = train_models()

    print("\n========== TRAINING COMPLETE ==========")
    print(training_results)
