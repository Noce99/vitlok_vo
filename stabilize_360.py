#!/usr/bin/env python3
"""Extract a rectilinear view from a 360 video that keeps facing the
direction of travel, instead of a single --360-direction fixed to the
camera housing.

    python stabilize_360.py GS010427.mp4 \
        --cori-csv "Telemetry/GS010427_GoPro Max-CORI.csv" \
        --gps-csv "Telemetry/GS010427_GoPro Max-GPS5.csv" \
        --base-direction "0,-110,0" --duration 15 --start 60 --clip-out preview.mp4

Why this exists
----------------
``--360-direction`` (see src/camera360.py) crops the same fixed direction,
relative to the camera housing, out of every frame. That is only "forward"
if the housing itself never rotates. On this bike dataset it does: CORI
(the camera's own per-frame orientation quaternion, in Telemetry/*-CORI.csv)
shows yaw drifting ~130 degrees and roll swinging +/-70 degrees over the
ride -- the camera is on a chest/helmet-style mount, not a rigid handlebar
bracket, so a "fixed" crop swings and tips with the rider's body instead of
tracking the road.

This script instead computes, per output frame, the body-frame rotation
that keeps the view:
  - level (roll/pitch cancelled using CORI's own frame-to-frame rotation),
  - pointed along the direction of travel (yaw driven by GPS course-over-
    ground, smoothed, rather than the camera's own yaw).

Method
------
CORI gives a quaternion ``R_cori(t)``: camera-body-frame axes at time t,
expressed in the fixed reference frame anchored at CORI's own t=0 (i.e.
``R_cori(t=0) = I``). That reference frame does not rotate over the video,
so it is a convenient place to define a *fixed* target direction that only
depends on GPS bearing, then rotate it into whatever the body frame happens
to be at time t:

  1. At a calibration instant t0 (default: first moment GPS reports a
     reliable speed), take --base-direction (the same "pitch,yaw,roll"
     meaning as --360-direction, eyeballed with check_360.py) as the
     body-frame forward at t0, and lift it into the reference frame:
     ``forward_ref0 = R_cori(t0) @ base_direction_vector``. "Up" in the
     reference frame is *measured*, not assumed level -- one sample of
     *-GRAV.csv at t0 gives which way is actually down in body-frame-at-t0.
     A third, "right" vector completes an orthonormal (forward, right, up)
     triad in the reference frame.
  2. For any other time t, rotate that whole triad rigidly within the
     reference frame's horizontal plane (Rodrigues, about the measured "up"
     axis) by the *change* in GPS bearing since t0 (unwrapped, heavily
     smoothed to ride out GPS jitter). Rotating the full triad -- rather
     than dropping to body frame first and re-deriving "right" from
     cross(up_body, forward_body) -- keeps it exactly orthonormal even when
     the camera points near straight up/down (e.g. a rider stopped and
     looking down at their handlebars), where that cross product degenerates.
  3. Drop the rotated triad into the current body frame with ``R_cori(t)^T``
     -- this is the rotation matrix build_view's ``rotation`` parameter
     would have been, had it varied per frame.

CORI supplies the leveling (steps 1-3 all route through it every frame);
GPS supplies only the *change* in yaw target relative to the calibration
instant, so its low rate (~17 Hz) and jitter matter far less than they
would for absolute heading.

Caveats:
- This dataset's route has essentially no sharp turns (GPS bearing never
  changes by more than a few tens of degrees in an 8-second window once
  low-speed jitter is filtered out -- checked separately), so the "track
  direction of travel" behaviour is only lightly exercised here; most of the
  visible improvement over a fixed --360-direction comes from the roll/pitch
  leveling. The yaw sign (--yaw-sign) was fixed by geometric derivation
  (Rodrigues about +up is CCW-from-above, compass bearing increases
  CW-from-above, so the bearing delta must be negated) and cross-checked
  against rendered clips, not a controlled large-turn test -- if applied to
  a route with sharper turns, sanity-check with --duration around a known
  turn before trusting a full run.
- Around a stop (rider dismounted, fiddling with something, etc.) the
  camera's own physical orientation can swing far outside anything seen
  while actually riding, and the extraction follows it faithfully -- this
  can look like a wrongly-rolled frame for a few seconds even though the
  math is doing exactly what it is told. Not a bug; just not very
  interesting output for those frames.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from src.camera360 import parse_direction, resolve_camera_model, _rotation_matrix
from src.undistortion import SIZE_MULTIPLE, _FrameWriter


def _cropped_size(width: int, height: int) -> tuple[int, int]:
    w = width - width % SIZE_MULTIPLE
    h = height - height % SIZE_MULTIPLE
    if w <= 0 or h <= 0:
        raise ValueError(f"output size {width}x{height} is below one {SIZE_MULTIPLE}-pixel block")
    return w, h


# --- telemetry loading -------------------------------------------------------

#: GoPro's raw IMU/CORI axes are not the x-forward/y-right/z-up convention
#: src/camera360.py uses for the equirectangular sphere. Fitted empirically
#: (see stabilize_360 module docstring): gravity at the start of this
#: recording sits almost entirely on GoPro's own middle axis, which lines up
#: with a rendered horizon only when that axis is treated as *down* and
#: swapped with the sphere's up axis -- i.e. (fwd, right, up) = (raw_x,
#: raw_z, -raw_y). Camera-model-specific, not dataset-specific; re-derive if
#: pointed at a different 360 camera's telemetry.
AXIS_PERMUTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


def load_cori(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (cts_ms sorted ascending, quats Nx4 as w,x,y,z), axis-corrected
    per :data:`AXIS_PERMUTATION`."""
    cts, quats = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header: cts,date,CameraOrientation,1,2,3,VPTS
        for row in r:
            cts.append(float(row[0]))
            raw = np.array([float(row[3]), float(row[4]), float(row[5])])
            x, y, z = AXIS_PERMUTATION @ raw
            quats.append((float(row[2]), x, y, z))
    cts = np.asarray(cts, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float64)
    order = np.argsort(cts)
    return cts[order], quats[order]


def load_grav_at(path: Path, t_ms: float) -> np.ndarray:
    """Axis-corrected gravity (points down) from *-GRAV.csv nearest t_ms."""
    best = None
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header: cts,date,Gravity Vector,1,2
        for row in r:
            cts = float(row[0])
            if best is None or abs(cts - t_ms) < abs(best[0] - t_ms):
                best = (cts, float(row[2]), float(row[3]), float(row[4]))
    raw = np.array(best[1:])
    return AXIS_PERMUTATION @ raw


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """w,x,y,z (need not be unit) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cori_matrix_at(cori_cts: np.ndarray, cori_quats: np.ndarray, t_ms: float) -> np.ndarray:
    """Nearest-sample CORI rotation matrix at time t_ms (rows are ~1/frame already)."""
    idx = int(np.searchsorted(cori_cts, t_ms))
    idx = min(max(idx, 0), len(cori_cts) - 1)
    if idx > 0 and abs(cori_cts[idx - 1] - t_ms) < abs(cori_cts[idx] - t_ms):
        idx -= 1
    return quat_to_matrix(cori_quats[idx])


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def load_gps_bearing(path: Path, stride: int, speed_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed, unwrapped GPS course-over-ground.

    Bearing is measured between fixes *stride* samples apart (not
    consecutive ones) and only kept where both endpoints exceed
    *speed_threshold* m/s -- raw consecutive-fix bearings are dominated by
    GPS jitter at typical GPS5 sampling rates (~17 Hz here).

    Returns (t_ms ascending, unwrapped bearing_deg) -- unwrapped so callers
    can np.interp through it without wraparound jumps.
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # cts,date,lat,lon,alt,2D speed,3D speed,fix,precision,altitude system
        for row in r:
            rows.append((float(row[0]), float(row[2]), float(row[3]), float(row[5])))

    t_ms, bearing = [], []
    for i in range(stride, len(rows)):
        c0, lat0, lon0, s0 = rows[i - stride]
        c1, lat1, lon1, s1 = rows[i]
        if s0 < speed_threshold or s1 < speed_threshold:
            continue
        t_ms.append((c0 + c1) / 2.0)
        bearing.append(_bearing_deg(lat0, lon0, lat1, lon1))

    if len(t_ms) < 2:
        raise ValueError(
            f"{path}: fewer than 2 usable GPS bearing samples above "
            f"{speed_threshold} m/s -- lower --speed-threshold or --gps-stride"
        )
    t_ms = np.asarray(t_ms, dtype=np.float64)
    bearing_unwrapped = np.degrees(np.unwrap(np.radians(bearing)))
    return t_ms, bearing_unwrapped


# --- per-frame direction ------------------------------------------------------

class Stabilizer:
    """Computes a per-frame body-frame rotation matrix that keeps the
    extracted view level and pointed along the smoothed GPS heading,
    anchored to --base-direction at a calibration instant. See module
    docstring for the derivation.
    """

    def __init__(
        self,
        cori_cts: np.ndarray,
        cori_quats: np.ndarray,
        gps_t_ms: np.ndarray,
        gps_bearing_deg: np.ndarray,
        base_direction: tuple[float, float, float],
        calibration_t_ms: float,
        yaw_sign: float,
        grav_at_calibration: np.ndarray,
    ) -> None:
        self._cori_cts = cori_cts
        self._cori_quats = cori_quats
        self._gps_t_ms = gps_t_ms
        self._gps_bearing = gps_bearing_deg
        self._yaw_sign = yaw_sign

        r0 = cori_matrix_at(cori_cts, cori_quats, calibration_t_ms)
        # up_ref: the reference frame is body-frame-at-calibration-time, which
        # need not itself be level, so "up" there is measured (via GRAV), not
        # assumed to be (0,0,1).
        up_body_t0 = -grav_at_calibration
        up_body_t0 /= np.linalg.norm(up_body_t0)
        self._up_ref = r0 @ up_body_t0

        base_rot = _rotation_matrix(*(math.radians(a) for a in base_direction))
        forward_body_t0 = base_rot @ np.array([1.0, 0.0, 0.0])
        self._forward_ref0 = r0 @ forward_body_t0
        # right_ref0 completes an orthonormal (forward, right, up) triad in
        # the *reference* frame, once, at calibration. Rotating this whole
        # triad rigidly (same Rodrigues rotation applied to all three) and
        # only then dropping into body frame via R_cori(t)^T keeps it exactly
        # orthonormal for every t, however tilted the camera happens to be --
        # unlike deriving "right" from cross(up_body, forward_body) in body
        # frame, which is well-conditioned generally but degenerates when the
        # camera points near straight down/up (forward_body -> parallel to
        # up_body), which does happen for real -- e.g. a rider stopped and
        # looking down at their handlebars/phone.
        right_ref0 = np.cross(self._up_ref, self._forward_ref0)
        self._right_ref0 = right_ref0 / np.linalg.norm(right_ref0)
        self._bearing0 = float(np.interp(calibration_t_ms, gps_t_ms, gps_bearing_deg))

    def _bearing_at(self, t_ms: float) -> float:
        return float(np.interp(t_ms, self._gps_t_ms, self._gps_bearing))

    def _rodrigues(self, v: np.ndarray, theta: float) -> np.ndarray:
        k = self._up_ref
        c, s = math.cos(theta), math.sin(theta)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)

    def rotation_at(self, t_ms: float) -> np.ndarray:
        delta = self._bearing_at(t_ms) - self._bearing0
        delta = (delta + 180.0) % 360.0 - 180.0
        theta = math.radians(self._yaw_sign * delta)
        # Rotate about up_ref (measured, not exactly the reference frame's
        # z-axis -- rotating about z here instead of up_ref was an earlier
        # bug: for the ~10deg tilt this dataset's up_ref has off z, it barely
        # mattered at small theta but compounded badly for the ~100deg+
        # bearing swings later in the ride).
        forward_ref = self._rodrigues(self._forward_ref0, theta)
        right_ref = self._rodrigues(self._right_ref0, theta)

        r_cori = cori_matrix_at(self._cori_cts, self._cori_quats, t_ms)
        forward_body = r_cori.T @ forward_ref
        right_body = r_cori.T @ right_ref
        up_body = r_cori.T @ self._up_ref
        return np.column_stack([forward_body, right_body, up_body])


def build_rays(out_width: int, out_height: int, out_fov_deg: float) -> np.ndarray:
    """View-frame unit ray directions for every output pixel (H,W,3), reused
    across frames -- only the rotation applied to them changes."""
    fov_out = math.radians(out_fov_deg)
    fx = out_width / (2.0 * math.tan(fov_out / 2.0))
    cx, cy = out_width / 2.0, out_height / 2.0
    u = np.arange(out_width, dtype=np.float64) + 0.5
    v = np.arange(out_height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([np.ones_like(uu), (uu - cx) / fx, -(vv - cy) / fx], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def rays_to_map(rays: np.ndarray, rotation: np.ndarray, model, source_w: int, source_h: int):
    rotated = rays @ rotation.T
    lon = np.degrees(np.arctan2(rotated[..., 1], rotated[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(rotated[..., 2], -1.0, 1.0)))
    lon_rel = (lon - model.yaw_offset_deg + 180.0) % 360.0 - 180.0
    lat_rel = lat - model.pitch_offset_deg
    map_x = (lon_rel / model.fov_h_deg + 0.5) * source_w
    map_y = (0.5 - lat_rel / model.fov_v_deg) * source_h
    if model.fov_h_deg >= 360.0:
        map_x = np.mod(map_x, source_w)
    else:
        outside = np.abs(lon_rel) > model.fov_h_deg / 2.0
        map_x = np.where(outside, -1.0, map_x)
    outside_v = np.abs(lat_rel) > model.fov_v_deg / 2.0
    map_y = np.where(outside_v, -1.0, map_y)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--cori-csv", type=Path, required=True)
    p.add_argument("--gps-csv", type=Path, required=True)
    p.add_argument("--grav-csv", type=Path, required=True,
                    help="*-GRAV.csv; used once, at --calibration-time, to measure which way is "
                         "actually up in the reference frame instead of assuming it is level")
    p.add_argument("--360-camera-model", dest="camera_model", default="gopromax")
    p.add_argument("--360-fov", dest="fov", type=float, default=90.0)
    p.add_argument("--360-out-width", dest="out_width", type=int, default=1920)
    p.add_argument("--360-out-height", dest="out_height", type=int, default=1080)
    p.add_argument("--base-direction", default="0,-110,0",
                    help="pitch,yaw,roll (degrees) at the calibration instant, same meaning as --360-direction")
    p.add_argument("--calibration-time", type=float, default=None,
                    help="seconds into the video to anchor --base-direction (default: first reliable-speed GPS sample)")
    p.add_argument("--speed-threshold", type=float, default=3.5,
                    help="m/s; GPS bearing samples below this on either endpoint are discarded as jitter")
    p.add_argument("--gps-stride", type=int, default=15,
                    help="GPS5 rows between bearing baseline endpoints (samples are ~58ms apart)")
    p.add_argument("--yaw-sign", type=float, default=-1.0, choices=[1.0, -1.0],
                    help="flip if the view turns the wrong way through a turn. Default -1: compass "
                         "bearing increases clockwise-from-above, but positive Rodrigues rotation "
                         "about an 'up' axis is counterclockwise-from-above, so the bearing delta "
                         "needs negating to turn the view the same way the bearing turned.")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None, help="default: whole video")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--clip-out", type=Path, required=True)
    args = p.parse_args()

    model = resolve_camera_model(args.camera_model, None)
    base_direction = parse_direction(args.base_direction)

    cori_cts, cori_quats = load_cori(args.cori_csv)
    gps_t_ms, gps_bearing = load_gps_bearing(args.gps_csv, args.gps_stride, args.speed_threshold)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    calibration_t_ms = args.calibration_time * 1000 if args.calibration_time is not None else gps_t_ms[0]
    grav_t0 = load_grav_at(args.grav_csv, calibration_t_ms)
    stab = Stabilizer(cori_cts, cori_quats, gps_t_ms, gps_bearing, base_direction,
                       calibration_t_ms, args.yaw_sign, grav_t0)

    width, height = _cropped_size(args.out_width, args.out_height)
    rays = build_rays(width, height, args.fov)

    start_frame = round(args.start * fps)
    n_frames = total - start_frame if args.duration is None else min(round(args.duration * fps), total - start_frame)
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"[stabilize360] {model.name} {source_w}x{source_h} -> {width}x{height}, "
          f"calibration_t={calibration_t_ms/1000:.1f}s base_direction={base_direction}, "
          f"{n_frames} frames from {args.start:.1f}s @ {fps:.3f} fps")

    args.clip_out.parent.mkdir(parents=True, exist_ok=True)
    writer = _FrameWriter(args.clip_out, width, height, fps, args.crf)
    written = 0
    try:
        while written < n_frames:
            ok, frame = capture.read()
            if not ok:
                break
            t_ms = (start_frame + written) / fps * 1000.0
            rotation = stab.rotation_at(t_ms)
            map_x, map_y = rays_to_map(rays, rotation, model, source_w, source_h)
            rectilinear = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            writer.write(rectilinear)
            written += 1
            if written % 100 == 0:
                print(f"\r[stabilize360] {written}/{n_frames} frames", end="", flush=True)
    finally:
        capture.release()
        writer.close()
    print(f"\r[stabilize360] {written} frames -> {args.clip_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
