import numpy as np

from calculate_eer import compute_eer as official_compute_eer
from src.metrics.eer import compute_eer


def test_eer_is_zero_for_separable_scores():
    eer, threshold = compute_eer([0.8, 0.9], [0.1, 0.2])
    assert eer == 0.0
    assert 0.2 <= threshold <= 0.8


def test_eer_is_invariant_to_monotonic_shift():
    bonafide = np.array([0.2, 0.4, 0.8])
    spoof = np.array([0.1, 0.3, 0.7])
    first, _ = compute_eer(bonafide, spoof)
    second, _ = compute_eer(bonafide + 10, spoof + 10)
    assert first == second


def test_eer_matches_provided_course_implementation():
    bonafide = np.array([0.2, 0.5, 0.5, 0.9])
    spoof = np.array([0.1, 0.5, 0.5, 0.8])

    expected_eer, expected_threshold = official_compute_eer(bonafide, spoof)
    actual_eer, actual_threshold = compute_eer(bonafide, spoof)

    assert actual_eer == expected_eer
    assert actual_threshold == expected_threshold
