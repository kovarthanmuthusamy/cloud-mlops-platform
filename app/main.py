"""FastAPI serving app for the wine classifier.

Loads the trained pipeline once at startup (path from the ``MODEL_PATH``
environment variable, defaulting to ``models/model.pkl``) and exposes:

- ``GET  /health``  — liveness + whether the model is loaded
- ``POST /predict`` — typed wine-feature request, returns label + class name
- ``GET  /metrics`` — Prometheus metrics (request counters + latency)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from cloud_mlops.data import class_names, feature_names

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
FEATURES = feature_names()
CLASSES = class_names()

REQUEST_COUNT = Counter("predict_requests_total", "Total /predict requests", ["status"])
PREDICTION_LATENCY = Histogram("predict_latency_seconds", "Latency of /predict in seconds")

_state: dict = {"model": None}


def load_model(path: Path | None = None):
    """Load (and cache) the trained pipeline from disk."""
    if path is None:
        path = MODEL_PATH
    if _state["model"] is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model file not found at {path}. Run scripts/run_training.py first."
            )
        _state["model"] = joblib.load(path)
    return _state["model"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except FileNotFoundError:
        # The app still starts so /health can report the missing model;
        # /predict will answer 503 until a model is in place.
        pass
    yield


app = FastAPI(title="cloud-mlops-platform API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Wine chemistry measurements (the sklearn wine feature names).

    One sklearn feature contains a slash (``od280/od315_of_diluted_wines``),
    which is not a valid Python identifier, so that field uses an alias;
    ``populate_by_name`` also accepts the underscore spelling.
    """

    model_config = ConfigDict(populate_by_name=True)

    alcohol: float = Field(examples=[13.2])
    malic_acid: float = Field(examples=[1.78])
    ash: float = Field(examples=[2.14])
    alcalinity_of_ash: float = Field(examples=[11.2])
    magnesium: float = Field(examples=[100.0])
    total_phenols: float = Field(examples=[2.65])
    flavanoids: float = Field(examples=[2.76])
    nonflavanoid_phenols: float = Field(examples=[0.26])
    proanthocyanins: float = Field(examples=[1.28])
    color_intensity: float = Field(examples=[4.38])
    hue: float = Field(examples=[1.05])
    od280_od315_of_diluted_wines: float = Field(
        alias="od280/od315_of_diluted_wines", examples=[3.4]
    )
    proline: float = Field(examples=[1050.0])


class PredictResponse(BaseModel):
    prediction: int
    class_name: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state["model"] is not None else "model_not_loaded",
        "model_loaded": _state["model"] is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    frame = pd.DataFrame([request.model_dump(by_alias=True)], columns=FEATURES)
    with PREDICTION_LATENCY.time():
        prediction = int(model.predict(frame)[0])
    REQUEST_COUNT.labels(status="ok").inc()
    return PredictResponse(prediction=prediction, class_name=CLASSES[prediction])


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
