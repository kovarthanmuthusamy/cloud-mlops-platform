"""Optional Evidently HTML drift report.

Evidently is an optional extra (``pip install evidently``) and is deliberately
not a core dependency: the always-on drift gate is the lightweight PSI + KS
implementation in :mod:`cloud_mlops.drift`. This module renders a rich,
shareable HTML report for humans when Evidently is available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path | str = "reports/drift_report.html",
) -> Path:
    """Render an Evidently data-drift report and return the output path.

    Raises ImportError with an actionable message if evidently is missing.
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "evidently is not installed; install the optional extra with "
            "`pip install evidently` to generate HTML drift reports. "
            "Programmatic drift detection works without it (cloud_mlops.drift)."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_df.drop(columns=["target"], errors="ignore"),
        current_data=current_df.drop(columns=["target"], errors="ignore"),
    )
    snapshot.save_html(str(output_path))
    return output_path
