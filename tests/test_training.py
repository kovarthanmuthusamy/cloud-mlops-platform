"""Training must produce a loadable model artifact with sane accuracy."""

import joblib

from cloud_mlops.train import train_model


def test_training_produces_model_and_accuracy(tmp_path):
    model_path = tmp_path / "model.pkl"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = train_model(model_path=model_path, tracking_uri=tracking_uri, n_estimators=50)

    assert result.model_path.exists()
    assert result.accuracy > 0.8
    assert result.run_id

    model = joblib.load(model_path)
    assert hasattr(model, "predict")
