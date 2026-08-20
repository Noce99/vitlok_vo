"""trajectory.txt must survive a round trip, with or without its header."""

import numpy as np
import pytest

from src.trajectory import HEADER, load_trajectory, path_length


def test_reads_file_with_header(tmp_path):
    path = tmp_path / "trajectory.txt"
    data = np.arange(15, dtype=float).reshape(3, 5)
    np.savetxt(path, data, fmt="%.6f", header=HEADER, comments="")

    assert np.allclose(load_trajectory(path), data)


def test_reads_file_without_header(tmp_path):
    """Older runs wrote four bare columns; those must still load."""
    path = tmp_path / "trajectory.txt"
    data = np.arange(12, dtype=float).reshape(3, 4)
    np.savetxt(path, data, fmt="%.6f")

    assert np.allclose(load_trajectory(path), data)


def test_rejects_too_few_columns(tmp_path):
    path = tmp_path / "trajectory.txt"
    np.savetxt(path, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="at least 4 columns"):
        load_trajectory(path)


def test_path_length():
    straight = np.array([[0.0, 0, 0, 0], [1.0, 3, 4, 0], [2.0, 3, 4, 0]])
    assert path_length(straight) == pytest.approx(5.0)
    assert path_length(straight[:1]) == 0.0
