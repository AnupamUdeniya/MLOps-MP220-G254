
import json
import re
from pathlib import Path
import hashlib
import yaml
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def calculate_data_sha256(data_path):
    """Calculate SHA-256 of the original, unmodified training CSV."""

    sha256 = hashlib.sha256()

    with open(data_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def initialize_mlflow(config):
    """Initialize MLflow tracking and return its client."""

    tracking_uri = config["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    client = MlflowClient(tracking_uri=tracking_uri)

    return client, experiment.experiment_id


def log_candidate_run(
    experiment_id,
    model_config,
    pipeline,
    validation_mae,
    data_sha256,
    test_size,
    random_state
):
    """Log one candidate model experiment in MLflow."""

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=model_config["name"]
    ) as run:

        mlflow.log_param(
            "data_sha256", data_sha256
        )

        mlflow.log_param(
            "model_name", model_config["name"]
        )

        mlflow.log_param(
            "model_type", model_config["type"]
        )

        mlflow.log_param(
            "test_size", test_size
        )

        mlflow.log_param(
            "random_state", random_state
        )

        for key, value in model_config["params"].items():
            mlflow.log_param(
                key,
                "None" if value is None else value
            )

        # The assignment requires an MAE metric.
        mlflow.log_metric(
            "mae", float(validation_mae)
        )

        mlflow.log_metric(
            "mae_val", float(validation_mae)
        )

        mlflow.set_tags({
            "contributor": "Naman Sharma",
            "role": "Model Lead",
            "git_branch": "naman/model-lead",
            "stage": "v1_candidate"
        })

        # Save the candidate model as an MLflow artifact.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="candidate_model",
            serialization_format="cloudpickle"
        )

        run_id = run.info.run_id

    return run_id


def register_v1(
    config,
    client,
    experiment_id,
    final_model,
    X,
    data_sha256,
    best_model_name,
    best_validation_mae
):
    """Register v1 once and assign the champion alias."""

    registered_name = config["mlflow"]["registered_model_name"]

    existing_versions = client.search_model_versions(
        filter_string=f"name='{registered_name}'"
    )

    # Prevent accidental duplicate v1 registration on reruns.
    if existing_versions:

        try:
            champion = client.get_model_version_by_alias(
                registered_name, "champion"
            )

            print(
                "Existing champion detected:",
                champion.version
            )

            return champion.version, champion.run_id

        except MlflowException:

            version_one = next(
                (
                    version for version in existing_versions
                    if str(version.version) == "1"
                ),
                None
            )

            if version_one is None:
                raise RuntimeError(
                    "Registry already contains versions, "
                    "but no champion or version 1 was found."
                )

            client.set_registered_model_alias(
                registered_name,
                "champion",
                "1"
            )

            return version_one.version, version_one.run_id

    # Register the final model in a dedicated MLflow run.
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="v1_final_registration"
    ) as run:

        registration_run_id = run.info.run_id

        mlflow.log_params({
            "model_name": best_model_name,
            "data_sha256": data_sha256,
            "training_samples": len(X),
            "model_stage": "v1_final"
        })

        mlflow.log_metric(
            "mae_val", float(best_validation_mae)
        )

        mlflow.set_tags({
            "contributor": "Naman Sharma",
            "role": "Model Lead",
            "stage": "v1_final_registration"
        })

        example = X.head(5)
        predictions = final_model.predict(example)

        signature = infer_signature(
            example,
            predictions
        )

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=example,
            registered_model_name=registered_name
        )

    versions = client.search_model_versions(
        filter_string=f"name='{registered_name}'"
    )

    if not versions:
        raise RuntimeError(
            "Model logging succeeded but no registered "
            "model version was found."
        )

    version = max(
        versions,
        key=lambda item: int(item.version)
    ).version

    if str(version) != "1":
        raise RuntimeError(
            f"Expected v1 to be registry version 1, got {version}."
        )

    client.set_registered_model_alias(
        registered_name,
        "champion",
        str(version)
    )

    print("Registered model:", registered_name)
    print("Registered version:", version)
    print("Champion alias assigned.")

    return version, registration_run_id



def update_tracking_findings(
    experiment_name,
    best_run_id,
    registered_model_name,
    champion_version
):
    """Update only four tracking values without reformatting findings.yaml."""

    findings_path = PROJECT_ROOT / "reports" / "findings.yaml"

    content = findings_path.read_text(encoding="utf-8")

    updates = {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "registered_model_name": registered_model_name,
        "champion_version": int(champion_version)
    }

    for key, value in updates.items():

        pattern = rf"(?m)^(\s*{re.escape(key)}:\s*).*$"

        serialized_value = (
            json.dumps(value)
            if isinstance(value, str)
            else str(value)
        )

        content, count = re.subn(
            pattern,
            lambda match: match.group(1) + serialized_value,
            content,
            count=1
        )

        if count != 1:
            raise ValueError(
                f"Could not uniquely update tracking field: {key}"
            )

    findings_path.write_text(content, encoding="utf-8")

    print("Updated only the four MLflow tracking fields.")
