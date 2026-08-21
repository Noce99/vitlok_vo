"""Camera360Model registry/parsing, and the equirectangular->pinhole geometry."""

import numpy as np
import pytest

from src.camera360 import (
    KNOWN_360_CAMERAS,
    build_view,
    parse_direction,
    resolve_camera_model,
)


def test_gopromax_is_a_full_sphere():
    model = KNOWN_360_CAMERAS["gopromax"]
    assert model.fov_h_deg == 360.0
    assert model.fov_v_deg == 180.0


def test_resolve_known_camera_is_case_insensitive():
    assert resolve_camera_model("GoProMax", None).name == "gopromax"


def test_resolve_unknown_camera_explains_how_to_fix_it():
    with pytest.raises(ValueError, match="not a known camera"):
        resolve_camera_model("insta360x3", None)


def test_resolve_custom_camera_from_params():
    model = resolve_camera_model("custom", "fov_h=180,fov_v=90,yaw_offset=10")
    assert model.fov_h_deg == 180.0
    assert model.fov_v_deg == 90.0
    assert model.yaw_offset_deg == 10.0
    assert model.pitch_offset_deg == 0.0


def test_custom_params_requires_fov():
    with pytest.raises(ValueError, match="fov_h and fov_v"):
        resolve_camera_model("custom", "yaw_offset=10")


def test_custom_params_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown key"):
        resolve_camera_model("custom", "fov_h=360,fov_v=180,banana=1")


@pytest.mark.parametrize("text", ["1,2", "1,2,3,4", "a,b,c"])
def test_parse_direction_rejects_malformed_input(text):
    with pytest.raises(ValueError, match="pitch,yaw,roll"):
        parse_direction(text)


def test_parse_direction():
    assert parse_direction(" 1.5 , -2 , 0 ") == (1.5, -2.0, 0.0)


def test_forward_direction_maps_to_frame_centre():
    """Zero pitch/yaw/roll looks straight out the front, at the horizon --
    the centre of a full-sphere equirectangular frame."""
    model = KNOWN_360_CAMERAS["gopromax"]
    map_x, map_y, calibration = build_view(
        model, (0.0, 0.0, 0.0),
        source_width=3600, source_height=1800,
        out_width=101, out_height=101, out_fov_deg=90.0,
    )
    cx, cy = 50, 50  # centre pixel of an odd-sized output
    assert map_x[cy, cx] == pytest.approx(1800.0, abs=1.0)
    assert map_y[cy, cx] == pytest.approx(900.0, abs=1.0)
    assert calibration.fx > 0 and calibration.dist.sum() == 0


def test_yaw_pans_the_source_column_right():
    model = KNOWN_360_CAMERAS["gopromax"]
    map_x, _, _ = build_view(
        model, (0.0, 90.0, 0.0),
        source_width=3600, source_height=1800,
        out_width=1, out_height=1, out_fov_deg=10.0,
    )
    assert map_x[0, 0] == pytest.approx(2700.0, abs=1.0)  # 3/4 of the way across


def test_pitch_up_maps_to_the_top_row():
    model = KNOWN_360_CAMERAS["gopromax"]
    _, map_y, _ = build_view(
        model, (90.0, 0.0, 0.0),
        source_width=3600, source_height=1800,
        out_width=1, out_height=1, out_fov_deg=10.0,
    )
    assert map_y[0, 0] == pytest.approx(0.0, abs=1.0)  # zenith is the top row


def test_full_sphere_wraps_horizontally_instead_of_clipping():
    model = KNOWN_360_CAMERAS["gopromax"]
    map_x, _, _ = build_view(
        model, (0.0, 179.0, 0.0),
        source_width=3600, source_height=1800,
        out_width=1, out_height=1, out_fov_deg=10.0,
    )
    assert np.all(map_x >= 0) and np.all(map_x <= 3600)


def test_cropped_vertical_fov_marks_poles_invalid():
    model = resolve_camera_model("custom", "fov_h=360,fov_v=90")
    _, map_y, _ = build_view(
        model, (90.0, 0.0, 0.0),
        source_width=3600, source_height=900,
        out_width=1, out_height=1, out_fov_deg=10.0,
    )
    assert map_y[0, 0] == -1.0
