#!/usr/bin/env python3
"""CLI entry point: check production data for drift against the training data.

Exits 0 when no drift is detected and 1 (printing DRIFT_DETECTED) when drift
is flagged. In the Airflow pipeline that non-zero exit is the trigger that
would start a retraining run.

Usage:
    python scripts/check_drift.py [--drifted]   # --drifted simulates drifted
                                                # production data for the demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_mlops.data import load_wine_frame, make_drifted_frame  # noqa: E402
from cloud_mlops.drift import detect_drift  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run drift detection.")
    parser.add_argument(
        "--drifted",
        action="store_true",
        help="Use mean-shifted data as the 'current' batch (demo of drift).",
    )
    args = parser.parse_args()

    reference = load_wine_frame()
    current = make_drifted_frame(reference) if args.drifted else reference.copy()

    report = detect_drift(reference, current)
    for feature in report.features:
        status = "DRIFT" if feature.drifted else "ok"
        print(
            f"{feature.feature:<28} psi={feature.psi:>7.3f} "
            f"ks_p={feature.ks_pvalue:>8.4f} [{status}]"
        )
    if report.drift_detected:
        print(f"DRIFT_DETECTED flagged_features={report.flagged_features}")
        return 1
    print("NO_DRIFT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
