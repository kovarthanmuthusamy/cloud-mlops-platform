#!/usr/bin/env bash
# End-to-end demo of the cloud-mlops-platform in ~2 minutes:
#   1. train the model (logs the run to MLflow)
#   2. drift check on clean data   -> NO_DRIFT, exit 0
#   3. drift check on shifted data -> DRIFT_DETECTED, exit 1
#   4. serve the API and call /health, /predict and /metrics
#
# Usage:  bash scripts/demo.sh        (uses .venv if present, else python3)
# The full walkthrough with captured output lives in docs/demo.md.
set -u
cd "$(dirname "$0")/.."

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
PORT="${DEMO_PORT:-8000}"

step() { printf '\n===== %s =====\n' "$1"; }

step "1/4 Train the model (MLflow tracking)"
PYTHONPATH=src "$PY" scripts/run_training.py

step "2/4 Drift check - clean production batch"
PYTHONPATH=src "$PY" scripts/check_drift.py
echo "exit code: $?"

step "3/4 Drift check - shifted production batch (simulated drift)"
PYTHONPATH=src "$PY" scripts/check_drift.py --drifted
echo "exit code: $? (1 = DRIFT_DETECTED, this is what triggers retraining)"

step "4/4 Serve the API and call the endpoints"
PYTHONPATH=src:. "$PY" -m uvicorn app.main:app --port "$PORT" >/tmp/mlops-demo-uvicorn.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
sleep 5

echo "--- GET /health"
curl -s "http://127.0.0.1:${PORT}/health"; echo
echo "--- POST /predict (a class_0 wine sample)"
curl -s -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2, "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76, "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38, "hue": 1.05, "od280/od315_of_diluted_wines": 3.4, "proline": 1050.0}'; echo
echo "--- GET /metrics (Prometheus, prediction counters)"
curl -s "http://127.0.0.1:${PORT}/metrics" | grep -E "^predict_requests_total|^predict_latency_seconds_count"

kill $SERVER_PID 2>/dev/null
trap - EXIT
printf '\nDemo complete. Interactive API docs: http://localhost:%s/docs when the server is running.\n' "$PORT"
