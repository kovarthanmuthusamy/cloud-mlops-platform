"""Drift detection: shifted data must be flagged, identical data must not."""

from cloud_mlops.data import load_wine_frame, make_drifted_frame
from cloud_mlops.drift import detect_drift, population_stability_index


def test_identical_data_has_no_drift():
    frame = load_wine_frame()
    report = detect_drift(frame, frame.copy())
    assert report.drift_detected is False
    assert report.flagged_features == []


def test_shifted_data_is_flagged_as_drift():
    reference = load_wine_frame()
    drifted = make_drifted_frame(reference, mean_shift=1.5)
    report = detect_drift(reference, drifted)
    assert report.drift_detected is True
    assert len(report.flagged_features) >= 3


def test_psi_is_near_zero_for_identical_samples():
    frame = load_wine_frame()
    psi = population_stability_index(frame["alcohol"], frame["alcohol"])
    assert psi < 0.01
