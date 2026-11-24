from pathlib import Path
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "mlflow.db"
MLRUNS = ROOT / "mlruns"


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not MLRUNS.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {MLRUNS}")

    conn = sqlite3.connect(DB_PATH)

    try:
        # Back up the database outside the project.
        with tempfile.NamedTemporaryFile(
            prefix="mp220-mlflow-backup-",
            suffix=".db",
            delete=False
        ) as temp:
            backup_path = Path(temp.name)

        with sqlite3.connect(backup_path) as backup:
            conn.backup(backup)

        print(f"Database backup: {backup_path}")

        # Prepare experiment paths.
        experiments = conn.execute(
            "SELECT experiment_id FROM experiments"
        ).fetchall()

        experiment_updates = []

        for (experiment_id,) in experiments:
            path = MLRUNS / str(experiment_id)
            path.mkdir(parents=True, exist_ok=True)
            experiment_updates.append((path.resolve().as_uri(), experiment_id))

        # Prepare run artifact paths.
        runs = conn.execute(
            "SELECT run_uuid, experiment_id FROM runs"
        ).fetchall()

        run_updates = []

        for run_id, experiment_id in runs:
            path = MLRUNS / str(experiment_id) / run_id / "artifacts"
            path.mkdir(parents=True, exist_ok=True)
            run_updates.append((path.resolve().as_uri(), run_id))

        # Validate and prepare logged-model artifact paths.
        logged_models = conn.execute(
            "SELECT model_id, experiment_id FROM logged_models"
        ).fetchall()

        model_updates = []

        for model_id, experiment_id in logged_models:
            path = (
                MLRUNS / str(experiment_id) /
                "models" / model_id / "artifacts"
            )

            if not (path / "MLmodel").is_file():
                raise FileNotFoundError(
                    f"Missing MLflow model artifact: {path / 'MLmodel'}"
                )

            model_updates.append((path.resolve().as_uri(), model_id))

        # Update paths transactionally.
        conn.execute("BEGIN IMMEDIATE")

        conn.executemany(
            "UPDATE experiments SET artifact_location = ? "
            "WHERE experiment_id = ?",
            experiment_updates
        )

        conn.executemany(
            "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
            run_updates
        )

        conn.executemany(
            "UPDATE logged_models SET artifact_location = ? "
            "WHERE model_id = ?",
            model_updates
        )

        conn.commit()

        print("\nMLflow paths repaired successfully.")
        print(f"Experiments updated: {len(experiment_updates)}")
        print(f"Runs updated: {len(run_updates)}")
        print(f"Logged models updated: {len(model_updates)}")
        print(f"Artifact root: {MLRUNS.resolve()}")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
