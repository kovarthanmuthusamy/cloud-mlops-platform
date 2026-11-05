"""Drift detection between a reference dataset and current production data.

Two complementary statistics are computed per numeric feature:

- **Population Stability Index (PSI)** — bins the reference distribution and
  measures how far the current distribution has moved. Rules of thumb:
  PSI < 0.1 no significant shift, 0.1–0.25 moderate shift, > 0.25 major shift.
- **Kolmogorov–Smirnov (KS) test** — a p-value below ``KS_PVALUE_THRESHOLD``
  rejects the hypothesis that both samples come from the same distribution.

A feature is flagged when either statistic crosses its threshold, and the
overall dataset is flagged when at least ``MIN_FLAGGED_FEATURES`` features are
flagged. In production this overall flag is what triggers retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05
MIN_FLAGGED_FEATURES = 1
_EPSILON = 1e-6


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute PSI of ``actual`` against the ``expected`` reference sample."""
    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_perc = np.clip(expected_counts / len(expected), _EPSILON, None)
    actual_perc = np.clip(actual_counts / len(actual), _EPSILON, None)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


@dataclass
class FeatureDrift:
    """Drift statistics for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


@dataclass
class DriftReport:
    """Aggregate drift result for a whole dataset."""

    features: list[FeatureDrift] = field(default_factory=list)
    drift_detected: bool = False

    @property
    def flagged_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = PSI_THRESHOLD,
    ks_pvalue_threshold: float = KS_PVALUE_THRESHOLD,
    min_flagged_features: int = MIN_FLAGGED_FEATURES,
    ignore_columns: tuple[str, ...] = ("target",),
) -> DriftReport:
    """Compare ``current_df`` against ``reference_df`` feature by feature.

    Returns a :class:`DriftReport` with per-feature PSI/KS scores and an
    overall ``drift_detected`` flag (True when at least
    ``min_flagged_features`` features cross a threshold).
    """
    report = DriftReport()
    shared = [
        column
        for column in reference_df.columns
        if column in current_df.columns and column not in ignore_columns
    ]
    for column in shared:
        reference = reference_df[column].dropna()
        current = current_df[column].dropna()
        psi = population_stability_index(reference, current)
        ks_statistic, ks_pvalue = stats.ks_2samp(reference, current)
        drifted = psi >= psi_threshold or ks_pvalue < ks_pvalue_threshold
        report.features.append(
            FeatureDrift(
                feature=column,
                psi=psi,
                ks_statistic=float(ks_statistic),
                ks_pvalue=float(ks_pvalue),
                drifted=drifted,
            )
        )
    report.drift_detected = len(report.flagged_features) >= min_flagged_features
    return report
