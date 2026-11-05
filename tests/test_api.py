"""API tests: /health responds and /predict returns a valid wine class."""

import pytest
from fastapi.testclient import TestClient

from cloud_mlops.data import class_names, load_wine_frame
from cloud_mlops.train import train_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("api")
    model_path = tmp_path / "model.pkl"
    train_model(
        model_path=model_path,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        n_estimators=50,
    )

    import app.main as main_module

    main_module._state["model"] = None
    main_module.MODEL_PATH = model_path
    with TestClient(main_module.app) as test_client:
        yield test_client
    main_module._state["model"] = None


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_class(client):
    frame = load_wine_frame().drop(columns=["target"])
    sample = frame.iloc[0].to_dict()
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1, 2)
    assert body["class_name"] in class_names()
    # The first wine sample is a class_0 cultivar; the model should agree.
    assert body["prediction"] == 0


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predict_requests_total" in response.text
