"""Putting an estimated trajectory into the ground truth's frame.

A visual-odometry trajectory starts at the origin, facing whichever way the
camera happened to face, on whatever plane the motion happened to lie in. The GPS
track starts somewhere on the map, oriented north-up. Before they can be
compared, one has to be moved onto the other:

1. :func:`flatten_and_rotate` fits the trajectory's own plane by SVD and rotates
   it flat, then matches its initial heading to the ground truth's over the first
   minute, then re-zeros it at the origin;
2. :func:`translate_to` shifts it onto the ground truth's starting point.

That is a rotation and a translation and **nothing else**. In particular no scale
is fitted -- no Umeyama, no Sim(3) alignment. Scale is the quantity the whole
pipeline exists to recover, so fitting it away here would hide exactly the error
we want to measure.

Heading is matched over a window rather than at a single point because the first
few seconds of both tracks are noisy: GPS scatters by metres at rest, and DPVO's
scale is unconverged until it has seen some parallax.
"""

from __future__ import annotations

import numpy as np

#: Seconds of each track averaged when matching initial heading.
HEADING_WINDOW_S = 60.0


def flatten_and_rotate(
    gt_txy: np.ndarray,
    txyz: np.ndarray,
    seconds: float = HEADING_WINDOW_S,
    return_rotation: bool = False,
):
    """Rotate an estimate onto the ground plane and into the ground truth's heading.

    The trajectory's dominant plane is found by SVD -- a walked or driven path is
    very nearly planar, so its smallest singular direction is the plane normal --
    and rotated onto the horizontal. The remaining freedom is a yaw, fixed by
    averaging both tracks' displacement from their own start over the first
    *seconds* and matching those directions.

    Args:
        gt_txy: ``(N, 3)`` ground truth ``[time, x, y]``.
        txyz: ``(M, 4)`` estimate ``[time, x, y, z]``. Modified in place.
        seconds: Heading-averaging window, clipped to each track's length.
        return_rotation: Also return the plane rotation and the yaw applied.

    Returns:
        The rotated ``txyz``, or ``(txyz, plane_rotation, yaw)``.
    """
    xyz = txyz[:, 1:4]
    centroid = np.mean(xyz, axis=0)
    centered = xyz - centroid

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        plane_rotation = _rotation_onto_z(normal)
    except np.linalg.LinAlgError:
        plane_rotation = np.eye(3)

    txyz[:, 1:4] = (plane_rotation @ centered.T).T + centroid

    yaw = _heading_difference(gt_txy, txyz, seconds)
    origin = txyz[0].copy()
    txyz[:, [1, 2]] = _rotate_2d(txyz[:, [1, 2]] - origin[1:3], yaw) + origin[1:3]
    txyz[:] -= txyz[0]

    if return_rotation:
        return txyz, plane_rotation, yaw
    return txyz


def translate_to(gt_txy: np.ndarray, txy: np.ndarray) -> np.ndarray:
    """Shift *txy* so its first point coincides with the ground truth's."""
    txy[:, [1, 2]] += gt_txy[0, [1, 2]] - txy[0, [1, 2]]
    return txy


def align(gt_txy: np.ndarray, txyz: np.ndarray,
          seconds: float = HEADING_WINDOW_S) -> np.ndarray:
    """Full alignment: flatten, match heading, translate. Returns ``(N, 3)``."""
    rotated = flatten_and_rotate(gt_txy, txyz.copy(), seconds=seconds)
    return translate_to(gt_txy, rotated[:, :3])


def crop_to_overlap(
    gt_txy: np.ndarray, txy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop both tracks to the time window they share."""
    start = max(gt_txy[0, 0], txy[0, 0])
    finish = min(gt_txy[-1, 0], txy[-1, 0])
    gt_slice = slice(
        int(np.argmin(np.abs(gt_txy[:, 0] - start))),
        int(np.argmin(np.abs(gt_txy[:, 0] - finish))) + 1,
    )
    est_slice = slice(
        int(np.argmin(np.abs(txy[:, 0] - start))),
        int(np.argmin(np.abs(txy[:, 0] - finish))) + 1,
    )
    return gt_txy[gt_slice], txy[est_slice]


def interpolate_gt_at(gt_txy: np.ndarray, txy: np.ndarray) -> tuple[np.ndarray, int]:
    """Resample the ground truth onto the estimate's timestamps.

    GPS logs at 1 Hz while the estimate has one pose per video frame, so a
    per-point comparison needs the sparser series interpolated onto the denser
    one. Both time axes are made relative to their own first sample first: the
    two clocks share an origin (the video's first frame) but not necessarily an
    epoch.

    Returns:
        ``(gt_at_pred_times, crop_index)`` where *crop_index* is the last
        estimate index still covered by the ground truth -- beyond it
        ``np.interp`` would be extrapolating, so callers should slice there.
    """
    gt_time = gt_txy[:, 0] - gt_txy[0, 0]
    pred_time = txy[:, 0] - txy[0, 0]
    crop_index = int(np.argmin(np.abs(pred_time - gt_time[-1])))
    x = np.interp(pred_time, gt_time, gt_txy[:, 1])
    y = np.interp(pred_time, gt_time, gt_txy[:, 2])
    return np.column_stack([pred_time, x, y]), crop_index


# --- helpers --------------------------------------------------------------

def _rotation_onto_z(normal: np.ndarray) -> np.ndarray:
    """Rodrigues rotation taking *normal* onto +Z."""
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z_axis)
    cosine = float(np.dot(normal, z_axis))
    sine = float(np.linalg.norm(axis))
    if sine <= 1e-10:
        return np.eye(3)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1 - cosine) / sine ** 2)


def _heading_difference(gt_txy: np.ndarray, txyz: np.ndarray,
                        seconds: float) -> float:
    """Yaw, in radians, taking the estimate's initial heading onto the truth's."""
    gt_origin = gt_txy[0]
    origin = txyz[0]
    seconds = min(seconds,
                  gt_txy[-1, 0] - gt_origin[0],
                  txyz[-1, 0] - origin[0])

    gt_end = int(np.argmin(np.abs(gt_txy[:, 0] - gt_origin[0] - seconds)))
    end = int(np.argmin(np.abs(txyz[:, 0] - origin[0] - seconds)))
    # A single index would be an empty slice; keep at least one sample.
    gt_end = max(gt_end, 1)
    end = max(end, 1)

    gt_direction = np.mean(gt_txy[:gt_end, [1, 2]] - gt_origin[1:3], axis=0)
    direction = np.mean(txyz[:end, [1, 2]] - origin[1:3], axis=0)
    return float(
        np.arctan2(gt_direction[1], gt_direction[0])
        - np.arctan2(direction[1], direction[0])
    )


def _rotate_2d(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``(N, 2)`` points by *angle* radians about the origin."""
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T
