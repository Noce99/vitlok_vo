"""GPX parsing, cropping, and the ENU/NED distinction that quietly breaks metrics."""

import numpy as np
import pytest

from src.gpx import load_ground_truth, read_gpx

SAMPLE = """<?xml version="1.0"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="57.700000" lon="11.900000"><time>2026-08-20T09:00:00Z</time></trkpt>
    <trkpt lat="57.700100" lon="11.900000"><time>2026-08-20T09:00:10Z</time></trkpt>
    <trkpt lat="57.700200" lon="11.900000"><time>2026-08-20T09:00:20Z</time></trkpt>
    <trkpt lat="57.700300" lon="11.900000"><time>2026-08-20T09:00:30Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""


@pytest.fixture
def gpx_file(tmp_path):
    path = tmp_path / "track.gpx"
    path.write_text(SAMPLE)
    return path


def test_read_gpx(gpx_file):
    lats, lons, times = read_gpx(gpx_file)
    assert len(lats) == 4
    assert lats[0] == pytest.approx(57.7)
    assert times[1] - times[0] == pytest.approx(10.0)


def test_northward_track_projects_to_increasing_y(gpx_file):
    """Latitude increases, so northing (y) must increase and easting stay put."""
    gt = load_ground_truth(gpx_path=gpx_file)
    assert np.all(np.diff(gt[:, 2]) > 0)
    assert np.abs(gt[:, 1]).max() < 1e-3


def test_time_window_crops(gpx_file):
    _, _, times = read_gpx(gpx_file)
    gt = load_ground_truth(gpx_path=gpx_file, start_time=times[1], duration_s=10.0)
    assert len(gt) == 2


def test_empty_window_is_an_error(gpx_file):
    with pytest.raises(ValueError, match="window"):
        load_ground_truth(gpx_path=gpx_file, start_time=0.0, duration_s=1.0)


def test_ned_swaps_the_horizontal_axes(tmp_path):
    """NED input is the opposite handedness, so x and y come back swapped."""
    path = tmp_path / "gt.txt"
    np.savetxt(path, np.array([[0.0, 0.0, 0.0, 0.0],
                               [1.0, 10.0, 20.0, 0.0]]))

    enu = load_ground_truth(gt_trajectory_path=path, axes="enu")
    ned = load_ground_truth(gt_trajectory_path=path, axes="ned")

    assert enu[1, 1] == 10.0 and enu[1, 2] == 20.0
    assert ned[1, 1] == 20.0 and ned[1, 2] == 10.0


def test_requires_a_source():
    with pytest.raises(ValueError, match="--gpx or --gt-trajectory"):
        load_ground_truth()
