"""Alignment must fix rotation and translation -- and must not fix scale."""

import numpy as np

from src.alignment import align, crop_to_overlap, interpolate_gt_at


def _circle(n=300, radius=20.0, duration=60.0):
    """A ground track: half a circle, so heading actually changes."""
    t = np.linspace(0, duration, n)
    angle = np.linspace(0, np.pi, n)
    return np.column_stack([t, radius * np.sin(angle), radius * (1 - np.cos(angle))])


def test_recovers_rotation_and_translation():
    """A rotated, translated copy of the truth aligns back onto it."""
    truth = _circle()
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    moved = truth.copy()
    moved[:, 1:3] = truth[:, 1:3] @ rotation.T + np.array([100.0, -50.0])
    estimate = np.column_stack([moved, np.zeros(len(moved))])

    aligned = align(truth, estimate)

    assert np.abs(aligned[:, 1:3] - truth[:, 1:3]).max() < 1e-6


def test_scale_error_survives_alignment():
    """Alignment must NOT absorb a scale error; that is the measurement."""
    truth = _circle()
    estimate = truth.copy()
    estimate[:, 1:3] *= 1.5
    estimate = np.column_stack([estimate, np.zeros(len(estimate))])

    aligned = align(truth, estimate)

    residual = np.linalg.norm(aligned[:, 1:3] - truth[:, 1:3], axis=1)
    assert residual.max() > 1.0


def test_interpolation_matches_pred_times():
    truth = _circle(n=60)
    estimate = _circle(n=600)
    interpolated, crop = interpolate_gt_at(truth.copy(), estimate)

    assert len(interpolated) == len(estimate)
    assert 0 < crop <= len(estimate)
    assert np.allclose(interpolated[:, 0], estimate[:, 0] - estimate[0, 0])


def test_crop_to_overlap():
    a = np.column_stack([np.arange(0.0, 100.0), np.zeros(100), np.zeros(100)])
    b = np.column_stack([np.arange(50.0, 200.0), np.zeros(150), np.zeros(150)])
    a_cropped, b_cropped = crop_to_overlap(a, b)

    assert a_cropped[0, 0] >= 50.0
    assert b_cropped[-1, 0] <= 99.0
