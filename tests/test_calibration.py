"""Calibration files round-trip, and malformed ones are rejected loudly."""

import numpy as np
import pytest

from src.calibration import Calibration, load_calibration, save_calibration

REPO_CALIBRATIONS = ["gopro_L", "gopro_N", "gopro_W", "gopro_SW", "tartan"]


@pytest.mark.parametrize("name", REPO_CALIBRATIONS)
def test_shipped_calibrations_load(name):
    """Every calibration shipped with the repo must parse."""
    from pathlib import Path
    calibration = load_calibration(
        Path(__file__).resolve().parent.parent / "calibration" / f"{name}.txt"
    )
    assert calibration.fx > 0 and calibration.fy > 0
    assert calibration.dist.size in (4, 5, 8, 12, 14)


def test_tartan_is_linear():
    """The synthetic camera has no distortion, and only four coefficients."""
    from pathlib import Path
    calibration = load_calibration(
        Path(__file__).resolve().parent.parent / "calibration" / "tartan.txt"
    )
    assert calibration.is_linear
    assert calibration.dist.size == 4


def test_round_trip(tmp_path):
    original = Calibration(1000.0, 1001.0, 640.0, 360.0,
                           np.array([0.1, -0.2, 0.001, 0.002, 0.05]))
    path = tmp_path / "cam.txt"
    save_calibration(path, original)
    loaded = load_calibration(path)

    assert loaded.fx == pytest.approx(original.fx, abs=1e-3)
    assert np.allclose(loaded.dist, original.dist)


def test_scaling_intrinsics():
    """Resizing scales the pinhole terms and leaves distortion alone."""
    calibration = Calibration(1000.0, 1000.0, 640.0, 360.0, np.array([0.1, 0, 0, 0, 0]))
    half = calibration.scaled(0.5)

    assert half.fx == 500.0 and half.cx == 320.0
    assert np.allclose(half.dist, calibration.dist)


def test_rejects_short_file(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("100 100 50 50\n")
    with pytest.raises(ValueError, match="at least 8 values"):
        load_calibration(path)


def test_rejects_bad_distortion_length(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("100 100 50 50 0.1 0.2 0.3 0.4 0.5 0.6\n")
    with pytest.raises(ValueError, match="distortion coefficients"):
        load_calibration(path)
