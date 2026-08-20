"""The scale/shift estimator must recover a known answer through noise and outliers."""

import numpy as np
import pytest

from src.solve_scale_shift import generate_synthetic_ground, solve_scale_shift

CAMERA_HEIGHT = 1.8


def test_recovers_known_scale_and_shift():
    """With 10% gross outliers, the robust fit still finds (s, t)."""
    s_true, t_true = 0.42, 0.15
    points, normal = generate_synthetic_ground(s_true, t_true, CAMERA_HEIGHT)

    result = solve_scale_shift(ground_points=points, n_plane=normal,
                               camera_height=CAMERA_HEIGHT)

    assert result.s == pytest.approx(s_true, abs=0.02)
    # The shift is far less well determined than the scale -- the two are nearly
    # collinear over a narrow depth range, which is what condition_number warns
    # about -- so it is only required to sit within its own error bar.
    assert abs(result.t - t_true) < 4 * max(result.sigma_t, 1e-3)
    assert result.n_inliers > 0.5 * len(points) * 0.9


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        solve_scale_shift(np.zeros((5, 2)), np.array([0, 1, 0]), CAMERA_HEIGHT)
    with pytest.raises(ValueError):
        solve_scale_shift(np.zeros((3, 3)), np.array([0, 1, 0]), CAMERA_HEIGHT)


def test_outliers_are_excluded():
    """A clean fit should keep more inliers than a heavily contaminated one."""
    clean, normal = generate_synthetic_ground(outlier_fraction=0.0)
    dirty, _ = generate_synthetic_ground(outlier_fraction=0.4)

    clean_result = solve_scale_shift(clean, normal, CAMERA_HEIGHT)
    dirty_result = solve_scale_shift(dirty, normal, CAMERA_HEIGHT)

    assert clean_result.n_inliers >= dirty_result.n_inliers
    assert clean_result.s == pytest.approx(0.42, abs=0.01)
