"""Dataset loading utilities.

The demo dataset is scikit-learn's bundled wine dataset, which ships with the
library itself, so nothing is downloaded at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_wine_frame() -> pd.DataFrame:
    """Return the full wine dataset as a DataFrame (features + ``target``)."""
    wine = load_wine()
    frame = pd.DataFrame(wine.data, columns=wine.feature_names)
    frame[TARGET_COLUMN] = wine.target
    return frame


def feature_names() -> list[str]:
    """Return the wine feature column names (without the target)."""
    return list(load_wine().feature_names)


def class_names() -> list[str]:
    """Return the human-readable wine class names, indexed by label."""
    return list(load_wine().target_names)


def split_data(
    frame: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a wine frame into train/test feature and label sets."""
    if frame is None:
        frame = load_wine_frame()
    features = frame.drop(columns=[TARGET_COLUMN])
    labels = frame[TARGET_COLUMN]
    return train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )


def make_drifted_frame(
    frame: pd.DataFrame | None = None,
    mean_shift: float = 1.5,
    columns: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a copy of the frame with selected feature means shifted.

    Simulates production data drift: each chosen numeric feature is shifted by
    ``mean_shift`` standard deviations, which both PSI and the KS test should
    clearly flag.
    """
    if frame is None:
        frame = load_wine_frame()
    drifted = frame.copy()
    rng = np.random.default_rng(random_state)
    if columns is None:
        columns = [c for c in frame.columns if c != TARGET_COLUMN][:6]
    for column in columns:
        std = drifted[column].std()
        drifted[column] = drifted[column] + mean_shift * std + rng.normal(0, 0.01, len(drifted))
    return drifted
