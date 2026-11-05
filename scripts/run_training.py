#!/usr/bin/env python3
"""CLI entry point: train the model and log the run to MLflow.

Usage:
    python scripts/run_training.py [--model-path models/model.pkl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_mlops.train import DEFAULT_MODEL_PATH, train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the wine classifier.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    result = train_model(model_path=args.model_path, n_estimators=args.n_estimators)
    print(f"run_id={result.run_id}")
    print(f"accuracy={result.accuracy:.4f} f1_weighted={result.f1_weighted:.4f}")
    print(f"model saved to {result.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
