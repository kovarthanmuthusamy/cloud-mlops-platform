"""Model training with MLflow experiment tracking.

Trains a scikit-learn pipeline (StandardScaler + RandomForest) on the wine
dataset, logs parameters and metrics to MLflow, and writes the fitted pipeline
to ``models/model.pkl`` for the serving app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cloud_mlops.data import split_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
# Local tracking by default: a sqlite file database in the project root.
# (MLflow >= 3 put the legacy ./mlruns file store into maintenance mode and
# refuses to write to it, so a local sqlite DB is the supported equivalent.
# Point MLFLOW_TRACKING_URI at a tracking server to override.)
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "cloud-mlops-wine"


@dataclass
class TrainingResult:
    model_path: Path
    accuracy: float
    f1_weighted: float
    run_id: str | None


def build_pipeline(n_estimators: int = 200, max_depth: int | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                ),
            ),
        ]
    )


def train_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    tracking_uri: str | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> TrainingResult:
    """Train, evaluate, log to MLflow and persist the model. Returns metrics."""
    model_path = Path(model_path)
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test = split_data()
    pipeline = build_pipeline(n_estimators=n_estimators, max_depth=max_depth)

    with mlflow.start_run() as run:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "features": x_train.shape[1],
                "train_rows": x_train.shape[0],
            }
        )
        mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1})
        # cloudpickle: MLflow 3's default skops format rejects tree-model
        # internals as untrusted types; cloudpickle is fine for our own runs.
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        run_id = run.info.run_id

    return TrainingResult(
        model_path=model_path, accuracy=float(accuracy), f1_weighted=float(f1), run_id=run_id
    )


if __name__ == "__main__":
    result = train_model()
    print(f"run_id={result.run_id}")
    print(f"accuracy={result.accuracy:.4f} f1_weighted={result.f1_weighted:.4f}")
    print(f"model saved to {result.model_path}")
