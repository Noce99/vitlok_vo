"""Drawing trajectories onto a georeferenced map.

A "georeferenced map" here is an ordinary raster image plus an ESRI **world
file** -- a six-line sidecar (``.pgw`` next to a ``.png``) giving the affine
transform from pixel to map coordinates::

    x_map = A * col + B * row + E
    y_map = C * col + D * row + F

Inverting it turns any point in the map's CRS into a pixel, which is all that is
needed to paint a GPS track and an estimate onto the map.

Estimated points are also drawn with an **uncertainty disc**: the radius is the
running sum of the per-frame ``metric_error`` the tracker reported for itself, so
the discs widen as accumulated drift grows. Every disc is painted onto one
separate layer that is blended over the map exactly once, so overlapping discs
keep a constant tint instead of stacking into an opaque blob wherever the path is
dense.

Without a map, the same content is plotted on plain axes instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

#: Colour of the GPS track, BGR.
GPS_COLOR = (200, 130, 0)

#: Radius in pixels of a plotted trajectory point.
POINT_RADIUS = 3

#: How strongly the uncertainty discs tint the map.
ERROR_ALPHA = 0.15

#: Margin in pixels kept around the ground truth when cropping the map.
CROP_MARGIN = 20


def read_world_file(path: Path) -> list[float]:
    """Read the six affine coefficients ``[A, B, C, D, E, F]``."""
    values = [
        float(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if len(values) != 6:
        raise ValueError(
            f"{path}: a world file has 6 lines, found {len(values)}"
        )
    return values


def find_world_file(image_path: Path) -> Optional[Path]:
    """Locate the world file beside *image_path*, trying the usual extensions."""
    image_path = Path(image_path)
    for extension in (".pgw", ".jgw", ".tfw", ".gfw", ".wld"):
        candidate = image_path.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def map_to_pixel(x_map: float, y_map: float,
                 world: Sequence[float]) -> tuple[int, int]:
    """Invert the world-file affine: map coordinates to ``(col, row)``."""
    a, b, c, d, e, f = world
    matrix = np.array([[a, b], [c, d]])
    col, row = np.linalg.solve(matrix, np.array([x_map - e, y_map - f]))
    return int(round(col)), int(round(row))


def plot_on_map(
    gt_txy: np.ndarray,
    tracks: Sequence[tuple[np.ndarray, tuple[int, int, int], str]],
    map_path: Path,
    world_path: Optional[Path],
    destination: Path,
    errors: Optional[dict[str, np.ndarray]] = None,
    crop: bool = True,
    map_alpha: float = 0.35,
) -> Path:
    """Draw the ground truth and each track onto the map image.

    Args:
        gt_txy: ``(N, 3)`` ground truth ``[time, x, y]`` in the map's CRS.
        tracks: ``(txy, bgr_colour, label)`` triples to overlay.
        map_path: Georeferenced raster.
        world_path: Its world file; discovered automatically when ``None``.
        destination: Where to write the rendered PNG.
        errors: Per-track arrays of per-point metric error, keyed by label.
        crop: Crop to the ground truth's extent plus a margin.
        map_alpha: How much of the map shows through; lower washes it out so the
            tracks stand out.

    Returns:
        *destination*.
    """
    map_path = Path(map_path)
    world_path = Path(world_path) if world_path else find_world_file(map_path)
    if world_path is None:
        raise FileNotFoundError(
            f"no world file found for {map_path}; pass --world explicitly"
        )
    world = read_world_file(world_path)

    image = cv2.imread(str(map_path))
    if image is None:
        raise ValueError(f"cannot read map image: {map_path}")
    if map_alpha < 1.0:
        image = cv2.addWeighted(image, map_alpha,
                                np.full_like(image, 255), 1 - map_alpha, 0)

    bounds = _draw_track(image, gt_txy, world, GPS_COLOR)
    if bounds is None:
        raise ValueError(
            f"no ground-truth point falls inside {map_path.name}. "
            "Are --epsg and the map's world file consistent?"
        )

    if errors:
        _draw_error_discs(image, tracks, world, errors)

    for txy, colour, label in tracks:
        drawn = _draw_track(image, txy, world, colour)
        inside = 0 if drawn is None else drawn[4]
        print(f"[map] {label}: {inside}/{len(txy)} points inside the map")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if crop:
        min_col, max_col, min_row, max_row, _ = bounds
        image = image[
            max(0, min_row - CROP_MARGIN):min(image.shape[0], max_row + CROP_MARGIN),
            max(0, min_col - CROP_MARGIN):min(image.shape[1], max_col + CROP_MARGIN),
        ]
    cv2.imwrite(str(destination), image)
    return destination


def _draw_track(image: np.ndarray, txy: np.ndarray, world: Sequence[float],
                colour: tuple[int, int, int]):
    """Draw one track; returns its pixel bounding box and how many points landed."""
    min_col, max_col = image.shape[1], 0
    min_row, max_row = image.shape[0], 0
    inside = 0
    for point in txy:
        try:
            col, row = map_to_pixel(point[1], point[2], world)
        except np.linalg.LinAlgError:
            continue
        if not (0 <= col < image.shape[1] and 0 <= row < image.shape[0]):
            continue
        cv2.circle(image, (col, row), POINT_RADIUS, colour[:3], -1)
        inside += 1
        min_col, max_col = min(min_col, col), max(max_col, col)
        min_row, max_row = min(min_row, row), max(max_row, row)
    if inside == 0:
        return None
    return min_col, max_col, min_row, max_row, inside


def _draw_error_discs(image: np.ndarray, tracks, world: Sequence[float],
                      errors: dict[str, np.ndarray]) -> None:
    """Paint accumulated-uncertainty discs under the tracks, blending once."""
    metres_per_pixel = abs(world[0])
    layer = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for txy, colour, label in tracks:
        if label not in errors:
            continue
        cumulative = np.cumsum(errors[label])
        for i, point in enumerate(txy):
            if i >= len(cumulative):
                break
            try:
                col, row = map_to_pixel(point[1], point[2], world)
            except np.linalg.LinAlgError:
                continue
            if not (0 <= col < image.shape[1] and 0 <= row < image.shape[0]):
                continue
            radius = max(1, int(round(cumulative[i] / metres_per_pixel)))
            cv2.circle(layer, (col, row), radius, colour[:3], -1)
            cv2.circle(mask, (col, row), radius, 255, -1)

    blended = cv2.addWeighted(layer, ERROR_ALPHA, image, 1 - ERROR_ALPHA, 0)
    selection = mask.astype(bool)
    image[selection] = blended[selection]


def plot_without_map(
    gt_txy: np.ndarray,
    tracks: Sequence[tuple[np.ndarray, tuple[int, int, int], str]],
    destination: Path,
    errors: Optional[dict[str, np.ndarray]] = None,
) -> Path:
    """Plot ground truth and tracks on plain equal-aspect axes, in metres."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 10))
    axes.plot(gt_txy[:, 1], gt_txy[:, 2], color="tab:blue",
              label="ground truth", linewidth=1.5)

    for txy, colour, label in tracks:
        rgb = (colour[2] / 255, colour[1] / 255, colour[0] / 255)
        if errors and label in errors:
            # A light opaque fill, drawn underneath, so overlapping discs do not
            # stack into a darker smear along the densest parts of the track.
            tint = tuple(c * ERROR_ALPHA + (1 - ERROR_ALPHA) for c in rgb)
            cumulative = np.cumsum(errors[label])
            for i, point in enumerate(txy):
                if i >= len(cumulative):
                    break
                axes.add_patch(plt.Circle((point[1], point[2]), cumulative[i],
                                          color=tint, linewidth=0, zorder=0))
        axes.plot(txy[:, 1], txy[:, 2], color=rgb, label=label, linewidth=1.5)

    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.legend()
    axes.grid(alpha=0.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination
