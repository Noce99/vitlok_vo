"""360-degree camera geometry: equirectangular source models and the rotation
that picks a rectilinear view out of the sphere.

A 360 video is a poor fit for ``cv2.undistort`` -- there is no single pinhole
image to correct into, only a full (or near-full) sphere. Instead, stage 1 in
360 mode (:mod:`src.undistortion_360`) treats the source frame as an
**equirectangular projection** of that sphere and resamples a virtual pinhole
camera pointed in a chosen direction out of it. Everything downstream then
receives the same thing it would from a normal lens: a linear video plus a
:class:`~src.calibration.Calibration` with zero distortion.

Two coordinate systems are in play:

* the source's own **body frame** -- x forward (the camera's own "front"),
  y right, z up -- in which longitude/latitude are the usual spherical
  coordinates (``lon=0, lat=0`` is straight out the front, at the horizon);
* the **view frame** of the virtual pinhole camera being extracted, with the
  same x-forward/y-right/z-up convention before ``--360-direction`` is
  applied.

``--360-direction pitch,yaw,roll`` rotates the view frame into the body frame:
positive yaw pans right, positive pitch tilts up, positive roll rotates the
image clockwise (the right edge swings down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .calibration import Calibration

#: Parameter names accepted by --360-camera-params, mapped to Camera360Model fields.
_PARAM_FIELDS = {
    "fov_h": "fov_h_deg",
    "fov_v": "fov_v_deg",
    "yaw_offset": "yaw_offset_deg",
    "pitch_offset": "pitch_offset_deg",
}


@dataclass(frozen=True)
class Camera360Model:
    """Equirectangular geometry of one 360 camera's stitched output.

    A full-sphere export maps image column 0..width to longitude
    ``-fov_h/2 .. +fov_h/2`` and row 0..height to latitude
    ``+fov_v/2 .. -fov_v/2`` (the top of the frame is up). Most consumer 360
    cameras, including the GoPro Max, export a genuine full sphere
    (``fov_h=360, fov_v=180``); the two offsets exist for exports that aren't
    centred on the camera's own forward direction and horizon.
    """

    name: str
    fov_h_deg: float = 360.0
    fov_v_deg: float = 180.0
    yaw_offset_deg: float = 0.0
    """Longitude of the source frame's centre column, relative to the camera's
    own forward direction."""
    pitch_offset_deg: float = 0.0
    """Latitude of the source frame's centre row, relative to the horizon."""


#: Registered cameras, keyed by the name passed to --360-camera-model (case-insensitive).
KNOWN_360_CAMERAS: dict[str, Camera360Model] = {
    "gopromax": Camera360Model("gopromax"),
}


def resolve_camera_model(name: str, params: Optional[str]) -> Camera360Model:
    """Look up *name* in :data:`KNOWN_360_CAMERAS`, or build one from *params*.

    Args:
        name: ``--360-camera-model`` value, e.g. ``"gopromax"`` or ``"custom"``.
        params: ``--360-camera-params`` value, e.g. ``"fov_h=360,fov_v=180"``.
            When given, it always wins -- this both defines unregistered
            cameras and lets a registered one be overridden.

    Raises:
        ValueError: *name* is not registered and *params* is not given; the
            message explains how to measure the missing numbers and pass them.
    """
    key = name.strip().lower()
    if params:
        return _parse_params(key, params)
    if key in KNOWN_360_CAMERAS:
        return KNOWN_360_CAMERAS[key]
    raise ValueError(_unknown_camera_message(key))


def _parse_params(name: str, text: str) -> Camera360Model:
    values: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--360-camera-params: expected key=value, got {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in _PARAM_FIELDS:
            raise ValueError(
                f"--360-camera-params: unknown key {key!r}; expected one of "
                f"{sorted(_PARAM_FIELDS)}"
            )
        try:
            values[_PARAM_FIELDS[key]] = float(value)
        except ValueError:
            raise ValueError(
                f"--360-camera-params: {key}={value!r} is not a number"
            ) from None
    if "fov_h_deg" not in values or "fov_v_deg" not in values:
        raise ValueError(
            "--360-camera-params must set at least fov_h and fov_v, e.g. "
            "'fov_h=360,fov_v=180'"
        )
    base = KNOWN_360_CAMERAS.get(name, Camera360Model(name))
    return replace(base, name=name, **values)


def _unknown_camera_message(name: str) -> str:
    known = ", ".join(sorted(KNOWN_360_CAMERAS)) or "(none yet)"
    return (
        f"--360-camera-model {name!r} is not a known camera ({known} known).\n"
        "\n"
        "Two numbers describe a stitched equirectangular video's field of view:\n"
        "  fov_h  - degrees of horizontal coverage (360 for a full sphere)\n"
        "  fov_v  - degrees of vertical coverage (180 for a full sphere; less if\n"
        "           the export crops the poles, e.g. a selfie-stick nadir patch)\n"
        "\n"
        "Find them on the camera's spec sheet, or measure them directly: open a\n"
        "single exported frame. A genuine full sphere is 2:1 (width:height) and\n"
        "has fov_h=360, fov_v=180; anything cropped will be wider than 2:1\n"
        "relative to its actual vertical coverage, in proportion to how much\n"
        "was cut.\n"
        "\n"
        "Once known, pass them with --360-camera-params, e.g.:\n"
        "  --360-camera-model custom --360-camera-params fov_h=360,fov_v=180\n"
        "\n"
        "Optional keys: yaw_offset (longitude of the frame's centre column) and\n"
        "pitch_offset (latitude of its centre row), both 0 for an export centred\n"
        "on the camera's own forward direction and horizon."
    )


def parse_direction(text: str) -> tuple[float, float, float]:
    """Parse ``"pitch,yaw,roll"`` (degrees) into a tuple of floats."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"--360-direction must be 'pitch,yaw,roll' in degrees, got {text!r}"
        )
    try:
        pitch, yaw, roll = (float(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"--360-direction must be 'pitch,yaw,roll' in degrees, got {text!r}"
        ) from None
    return pitch, yaw, roll


def build_view(
    model: Camera360Model,
    direction: tuple[float, float, float],
    source_width: int,
    source_height: int,
    out_width: int,
    out_height: int,
    out_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, Calibration]:
    """Build a ``cv2.remap`` lookup that extracts one rectilinear view.

    Args:
        model: Source equirectangular geometry.
        direction: ``(pitch, yaw, roll)`` in degrees, see the module docstring.
        source_width, source_height: Size of the equirectangular source frame.
        out_width, out_height: Size of the extracted pinhole view.
        out_fov_deg: Horizontal field of view of the extracted view, in degrees.

    Returns:
        ``(map_x, map_y, calibration)`` -- float32 maps for ``cv2.remap`` and the
        pinhole :class:`~src.calibration.Calibration` (zero distortion) that
        describes the extracted view.
    """
    pitch, yaw, roll = (math.radians(a) for a in direction)
    rotation = _rotation_matrix(pitch, yaw, roll)

    fov_out = math.radians(out_fov_deg)
    fx = out_width / (2.0 * math.tan(fov_out / 2.0))
    fy = fx
    cx = out_width / 2.0
    cy = out_height / 2.0

    u = np.arange(out_width, dtype=np.float64) + 0.5
    v = np.arange(out_height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)  # each (out_height, out_width)

    rays = np.stack(
        [np.ones_like(uu), (uu - cx) / fx, -(vv - cy) / fy], axis=-1
    )  # (H, W, 3), x-forward/y-right/z-up view frame
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    rotated = rays @ rotation.T  # into the body frame

    lon = np.degrees(np.arctan2(rotated[..., 1], rotated[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(rotated[..., 2], -1.0, 1.0)))

    lon_rel = _wrap180(lon - model.yaw_offset_deg)
    lat_rel = lat - model.pitch_offset_deg

    map_x = (lon_rel / model.fov_h_deg + 0.5) * source_width
    map_y = (0.5 - lat_rel / model.fov_v_deg) * source_height

    if model.fov_h_deg >= 360.0:
        map_x = np.mod(map_x, source_width)
    else:
        # Outside the source's horizontal coverage: point cv2.remap off-image
        # so BORDER_CONSTANT fills black instead of wrapping to the wrong side.
        outside = np.abs(lon_rel) > model.fov_h_deg / 2.0
        map_x = np.where(outside, -1.0, map_x)

    outside_v = np.abs(lat_rel) > model.fov_v_deg / 2.0
    map_y = np.where(outside_v, -1.0, map_y)

    calibration = Calibration(fx=fx, fy=fy, cx=cx, cy=cy, dist=np.zeros(5))
    return map_x.astype(np.float32), map_y.astype(np.float32), calibration


def _wrap180(deg: np.ndarray) -> np.ndarray:
    """Wrap degrees into ``[-180, 180)``."""
    return (deg + 180.0) % 360.0 - 180.0


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Rotation from the view frame into the body frame, in radians.

    Composed as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``: roll is applied first
    (around the viewing axis, so it only spins the image), then pitch (tilt),
    then yaw (pan) -- see the module docstring for the sign conventions.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, sr], [0.0, -sr, cr]])
    return rz @ ry @ rx
