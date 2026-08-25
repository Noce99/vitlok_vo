"""Ground-plane orientation from a segmented ground point cloud.

Stage 3 needs a plane normal to write down its constraint ``n . X = h``. Rather
than fitting one global plane through the ground points -- which a handful of
non-coplanar points can swing badly -- a local normal is estimated at every point
(PCA over its k nearest neighbours) and the unit normals are averaged. That
degrades gracefully on gently undulating ground, which is what forest tracks and
field paths actually look like.

One deliberate simplification: only the **roll** component of the normal is kept.
Camera pitch is very nearly aliased with depth scale and shift -- tilting the
camera down and scaling the depth up move the fitted ground plane in almost the
same way -- so estimating pitch here and feeding it into the scale/shift solve
makes the system ill-conditioned. Zeroing it costs little and stabilises the fit.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: Camera-frame "up" direction. The pipeline flips unprojected points to Y-up
#: before they get here, so the ground normal should point along +Y.
UP = (0.0, 1.0, 0.0)


def ground_normal(
    points: np.ndarray,
    knn: int = 30,
    up: tuple[float, float, float] = UP,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Estimate the ground-plane normal by averaging per-point surface normals.

    A normal is estimated at each point by local PCA over its *knn* nearest
    neighbours (including itself): the eigenvector of the neighbourhood's
    covariance matrix with the smallest eigenvalue is the direction of least
    spread, i.e. the surface normal. Those normals have an arbitrary sign, so
    each is flipped into the same hemisphere as *up* before averaging; the mean
    of unit vectors is the mean direction, which is then renormalised.

    Args:
        points: ``(N, 3)`` ground points in camera coordinates, Y-up.
        knn: Neighbours per local PCA.
        up: Direction every normal is flipped towards.

    Returns:
        ``(normal, angle_z, angle_x, roll_only_normal)`` where *normal* is the
        full unit normal with a non-negative Y component, *angle_z* is the tilt
        in the XY plane (roll) and *angle_x* the tilt in the YZ plane (pitch),
        both in radians, and *roll_only_normal* is *normal* with its pitch
        component removed -- the one the scale/shift solve should use.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")

    k = min(knn, len(pts))
    _, neighbour_idx = cKDTree(pts).query(pts, k=k)
    neighbour_idx = neighbour_idx.reshape(len(pts), k)
    neighbourhoods = pts[neighbour_idx]                        # (N, k, 3)
    centered = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    covariances = np.einsum("nki,nkj->nij", centered, centered) / k
    eigvals, eigvecs = np.linalg.eigh(covariances)             # ascending order
    normals = eigvecs[:, :, 0]                                 # smallest eigenvalue

    up_vector = np.asarray(up, dtype=np.float64)
    up_vector = up_vector / np.linalg.norm(up_vector)
    flip = (normals @ up_vector) < 0
    normals[flip] = -normals[flip]

    normal = normals.mean(axis=0)
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-12:
        # Normals cancelled out entirely: nothing better to say than "up".
        normal = up_vector.copy()
    else:
        normal = normal / magnitude
    if normal[1] < 0:
        normal = -normal

    angle_z = float(np.arctan2(normal[0], normal[1]))
    angle_x = float(np.arctan2(-normal[2], normal[1]))
    roll_only = np.array([np.sin(angle_z), np.cos(angle_z), 0.0])
    return normal, angle_z, angle_x, roll_only
