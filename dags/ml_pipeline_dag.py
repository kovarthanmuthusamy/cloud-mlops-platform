"""Airflow DAG for the cloud-mlops-platform training pipeline.

Daily schedule:

    prepare_data >> train_model >> evaluate_model >> check_drift
                                                       |
                                                       v
                                              trigger_retraining (BranchPythonOperator)
                                                       |
                                                       v
                                              log_no_retrain_needed

- ``evaluate_model`` fails the run when test accuracy drops below
  ``ACCURACY_THRESHOLD`` — a bad model never reaches the drift stage.
- ``check_drift`` writes a DRIFT / NO_DRIFT marker via XCom;
  ``trigger_retraining`` branches on it. In this portfolio version the
  "retrain" branch simply re-runs training synchronously; in production it
  would be a TriggerDagRunOperator starting a dedicated retraining DAG.

Requires Airflow >= 2.8 (TaskFlow API). Paths assume the repository root is
available to the workers (locally: mounted by docker-compose).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ACCURACY_THRESHOLD = 0.90

default_args = {
    "owner": "kovarthanam",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ml_pipeline",
    description="Train, evaluate and drift-check the wine classifier daily.",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "training", "drift"],
)
def ml_pipeline():
    @task
    def prepare_data() -> dict:
        """Load the training data and report basic shape statistics."""
        from cloud_mlops.data import load_wine_frame

        frame = load_wine_frame()
        return {"rows": len(frame), "columns": list(frame.columns)}

    @task
    def train_model() -> dict:
        """Train the model and log the run to MLflow."""
        from cloud_mlops.train import train_model as _train

        result = _train()
        return {"accuracy": result.accuracy, "run_id": result.run_id}

    @task
    def evaluate_model(training: dict) -> float:
        """Quality gate: fail the DAG run if accuracy is too low."""
        accuracy = training["accuracy"]
        if accuracy < ACCURACY_THRESHOLD:
            raise ValueError(
                f"Accuracy {accuracy:.3f} below threshold {ACCURACY_THRESHOLD}; "
                "refusing to promote this model."
            )
        return accuracy

    @task
    def check_drift() -> str:
        """Compare the latest production batch against the training data."""
        from cloud_mlops.data import load_wine_frame, make_drifted_frame
        from cloud_mlops.drift import detect_drift

        reference = load_wine_frame()
        # Placeholder for the real production batch loader: the latest batch
        # would be read from S3 here. A drifted copy stands in for the demo.
        current = make_drifted_frame(reference)
        report = detect_drift(reference, current)
        return "DRIFT" if report.drift_detected else "NO_DRIFT"

    def _branch(**context) -> str:
        marker = context["ti"].xcom_pull(task_ids="check_drift")
        return "retrain_model" if marker == "DRIFT" else "log_no_retrain_needed"

    @task(task_id="retrain_model")
    def retrain() -> str:
        """Drift was detected: retrain on the freshest data.

        In production this task would be replaced by a TriggerDagRunOperator
        launching the dedicated retraining DAG with the new data snapshot.
        """
        from cloud_mlops.train import train_model as _train

        result = _train()
        return f"retrained run_id={result.run_id} accuracy={result.accuracy:.3f}"

    @task(task_id="log_no_retrain_needed")
    def no_retrain() -> str:
        return "No drift detected; current model stays in production."

    data_info = prepare_data()
    training = train_model()
    accuracy = evaluate_model(training)
    drift_marker = check_drift()

    branch = BranchPythonOperator(
        task_id="trigger_retraining",
        python_callable=_branch,
    )

    data_info >> training >> accuracy >> drift_marker >> branch
    branch >> [retrain(), no_retrain()]


ml_pipeline()
