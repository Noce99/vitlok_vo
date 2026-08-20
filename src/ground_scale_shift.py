"""Stage 3 -- recover the depth maps' metric scale and shift from the ground.

A monocular depth network's output is metric only up to an unknown affine
transform ``d_metric = s * d_pred + t``. This stage estimates ``s`` and ``t`` for
every frame, using the one measurement we actually have: the camera's height
above the ground.

Per frame:

1. **Unproject** the bottom slice of the depth map into a camera-frame point
   cloud (only the bottom, because that is where the ground is), then flip to
   Y-up.
2. **Segment the ground** with CSF, the Cloth Simulation Filter: an inverted
   cloth is dropped onto the point cloud and the points it settles on are ground.
   It is purely geometric -- no learned segmentation network is involved, and
   none is needed.
3. **Estimate the plane normal** by averaging local surface normals, keeping only
   the roll component (see :mod:`src.normals`).
4. **Solve for (s, t)** by asserting that the ground plane lies exactly
   ``camera_height`` below the camera (see :mod:`src.solve_scale_shift`).

Fitting every frame would be wasteful: the solve is single-threaded and by far
the slowest CPU work in the pipeline, while ``s`` and ``t`` drift slowly. So the
fit runs every ``--sas-stride`` frames and the series is interpolated back onto
every frame afterwards.

Frames where the fit fails -- no ground found, too few usable rays -- record NaN
and are filled in by interpolation from their neighbours, so a failure never
shifts the rest of the series out of alignment with the video.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from .config import RunConfig
from .depth import DepthMaps
from .normals import ground_normal
from .solve_scale_shift import ScaleShiftResult, solve_scale_shift
from .undistortion import LinearVideo
from .workdir import WorkDir

#: Take every N-th pixel in each direction when unprojecting. The ground is a
#: large smooth surface; full resolution adds cost, not information.
POINT_SUBSAMPLE = 2

#: Depth range considered when looking for ground, in metres.
MIN_GROUND_DEPTH = 0.1

#: Inlier threshold for the scale/shift RANSAC, in metres of plane distance.
RANSAC_THRESHOLD = 0.15

#: Fewest ground points worth attempting a fit with.
MIN_GROUND_POINTS = 10


@dataclass(frozen=True)
class ScaleShiftSeries:
    """Per-frame depth corrections, one entry per depth map."""

    path: Path
    scale: np.ndarray
    shift: np.ndarray
    n_fitted: int
    """Frames where the fit succeeded."""
    n_attempted: int
    """Frames the fit was attempted on (roughly ``n_frames / sas_stride``)."""
    mean_inlier_ratio: float
    mean_condition_number: float

    @property
    def success_rate(self) -> float:
        return self.n_fitted / self.n_attempted if self.n_attempted else 0.0

    def summary(self) -> dict:
        """Compact provenance record for the run metadata."""
        return {
            "median_scale": float(np.median(self.scale)),
            "median_shift": float(np.median(self.shift)),
            "scale_std": float(np.std(self.scale)),
            "shift_std": float(np.std(self.shift)),
            "frames_fitted": self.n_fitted,
            "frames_attempted": self.n_attempted,
            "success_rate": self.success_rate,
            "mean_inlier_ratio": self.mean_inlier_ratio,
            "mean_condition_number": self.mean_condition_number,
        }


def fit_ground_scale_shift(
    cfg: RunConfig,
    video: LinearVideo,
    depth_maps: DepthMaps,
    work: WorkDir,
) -> ScaleShiftSeries:
    """Fit per-frame depth scale and shift, and write them to ``work.corrections``.

    Args:
        cfg: Supplies ``camera_height``, ``sas_stride``, ``bottom_fraction`` and
            the smoothing sigma.
        video: The linear video, for its intrinsics.
        depth_maps: Stage 2's output.
        work: Destination for ``sas_corrections.h5``.

    Returns:
        The interpolated, full-length :class:`ScaleShiftSeries`.

    Raises:
        RuntimeError: if the ground was never found in any frame.
    """
    calibration = depth_maps.intrinsics_for(video)
    fx, fy, cx, cy = calibration.fx, calibration.fy, calibration.cx, calibration.cy

    fitted_indices: list[int] = []
    scales: list[float] = []
    shifts: list[float] = []
    inlier_ratios: list[float] = []
    condition_numbers: list[float] = []

    attempted = 0
    print(f"[ground] fitting every {cfg.sas_stride} frame(s) of "
          f"{depth_maps.n_frames}, camera height {cfg.camera_height} m")

    # CSF drops a cloth_nodes.txt into the working directory; keep it in the
    # scratch folder rather than wherever the user happened to launch from.
    with _working_directory(work.path):
        for frame_index, depth in _iter_depth(depth_maps, cfg.sas_stride):
            attempted += 1
            result = _fit_one_frame(
                depth, fx, fy, cx, cy,
                camera_height=cfg.camera_height,
                bottom_fraction=cfg.bottom_fraction,
                max_depth=cfg.max_ground_depth,
            )
            fitted_indices.append(frame_index)
            if result is None:
                scales.append(np.nan)
                shifts.append(np.nan)
            else:
                scales.append(result.s)
                shifts.append(result.t)
                total = max(int(result.inlier_mask.size), 1)
                inlier_ratios.append(result.n_inliers / total)
                condition_numbers.append(result.condition_number)
            if attempted % 20 == 0:
                ok = int(np.count_nonzero(~np.isnan(scales)))
                print(f"\r[ground] {attempted} fits attempted, {ok} succeeded",
                      end="", flush=True)
    print()

    n_fitted = int(np.count_nonzero(~np.isnan(scales)))
    if n_fitted == 0:
        raise RuntimeError(
            "the ground plane was never found.\n"
            "Common causes: --camera-height is wrong, the camera does not see "
            "the ground, or --bottom-fraction is too small."
        )

    scale, shift = _densify(
        np.asarray(fitted_indices, dtype=float),
        np.asarray(scales, dtype=float),
        np.asarray(shifts, dtype=float),
        n_frames=depth_maps.n_frames,
        smooth_sigma=cfg.sas_smooth_sigma,
    )

    with h5py.File(work.corrections, "w") as handle:
        handle.create_dataset("scale", data=scale.astype(np.float32),
                              compression="gzip")
        handle.create_dataset("shift", data=shift.astype(np.float32),
                              compression="gzip")
        handle.attrs["camera_height"] = cfg.camera_height
        handle.attrs["stride"] = cfg.sas_stride
        handle.attrs["smooth_sigma"] = cfg.sas_smooth_sigma

    series = ScaleShiftSeries(
        path=work.corrections,
        scale=scale,
        shift=shift,
        n_fitted=n_fitted,
        n_attempted=attempted,
        mean_inlier_ratio=float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        mean_condition_number=(
            float(np.mean(condition_numbers)) if condition_numbers else float("nan")
        ),
    )
    print(f"[ground] {n_fitted}/{attempted} fits succeeded; "
          f"scale {np.median(scale):.4f}, shift {np.median(shift):+.4f} m "
          f"(medians) -> {work.corrections}")
    return series


# --- per-frame fit --------------------------------------------------------

def _fit_one_frame(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    camera_height: float,
    bottom_fraction: float,
    max_depth: float,
) -> Optional[ScaleShiftResult]:
    """Fit one frame, or return ``None`` if the ground could not be resolved."""
    height = depth.shape[0]
    row_start = int(height * (1.0 - bottom_fraction))
    points = unproject(depth, fx, fy, cx, cy, row_start=row_start,
                       min_depth=MIN_GROUND_DEPTH, max_depth=max_depth)
    if len(points) < MIN_GROUND_POINTS:
        return None
    # Camera coordinates have +Y pointing down; flip so the ground is below.
    points[:, 1] *= -1

    ground_mask = _csf_ground_mask(points)
    if ground_mask is None or int(ground_mask.sum()) < MIN_GROUND_POINTS:
        return None
    ground_points = points[ground_mask]

    try:
        _, _, _, roll_only = ground_normal(ground_points)
        return solve_scale_shift(
            ground_points=ground_points,
            # Negated so that n . X = +h for points on the ground, i.e. the
            # normal points from the ground towards the camera.
            n_plane=-np.asarray(roll_only),
            camera_height=camera_height,
            ransac_thresh=RANSAC_THRESHOLD,
        )
    except (ValueError, np.linalg.LinAlgError):
        # Degenerate geometry for this frame; interpolation will cover it.
        return None


def unproject(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    row_start: int = 0,
    min_depth: float = MIN_GROUND_DEPTH,
    max_depth: float = 30.0,
) -> np.ndarray:
    """Back-project a depth map's lower rows into camera-frame 3D points.

    Standard pinhole unprojection ``X = (u - cx) * d / fx``, ``Y = (v - cy) * d / fy``,
    ``Z = d``, over a grid subsampled by :data:`POINT_SUBSAMPLE` and restricted to
    ``[min_depth, max_depth]``.

    Args:
        depth: ``(H, W)`` depth map in metres.
        fx, fy, cx, cy: Intrinsics matching *depth*'s resolution.
        row_start: First image row to use; rows above it are ignored.
        min_depth: Nearer points are dropped (usually encoder or network noise).
        max_depth: Further points are dropped (too imprecise to constrain a plane).

    Returns:
        ``(N, 3)`` float64 points in camera coordinates (+Y down, +Z forward).
    """
    region = depth[row_start:]
    region_height, region_width = region.shape
    us = np.arange(0, region_width, POINT_SUBSAMPLE, dtype=np.float64)
    vs = np.arange(row_start, row_start + region_height, POINT_SUBSAMPLE,
                   dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)
    depths = region[::POINT_SUBSAMPLE, ::POINT_SUBSAMPLE].astype(np.float64).ravel()

    valid = (depths >= min_depth) & (depths <= max_depth) & np.isfinite(depths)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    u = grid_u.ravel()[valid]
    v = grid_v.ravel()[valid]
    d = depths[valid]
    return np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], axis=1)


def _csf_ground_mask(points: np.ndarray) -> Optional[np.ndarray]:
    """Segment ground with CSF, returning a boolean mask over *points*.

    CSF expects the height axis last, whereas our points are Y-up, so the Y and Z
    columns are swapped on the way in. CSF also chatters on stdout, which is
    silenced at the file-descriptor level because the noise comes from its C++
    side and ``contextlib.redirect_stdout`` cannot reach it.
    """
    try:
        import CSF
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "the ground fit needs the CSF package:\n"
            "    pip install cloth-simulation-filter"
        ) from exc

    filter_ = CSF.CSF()
    filter_.params.bSloopSmooth = False
    filter_.params.cloth_resolution = 0.5
    filter_.params.rigidness = 1
    filter_.setPointCloud(points[:, [0, 2, 1]])

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    with _suppress_stdout():
        filter_.do_filtering(ground, non_ground, exportCloth=False)

    indices = np.asarray(ground, dtype=int)
    if indices.size == 0:
        return None
    mask = np.zeros(len(points), dtype=bool)
    mask[indices] = True
    return mask


# --- series post-processing ----------------------------------------------

def _densify(
    fitted_indices: np.ndarray,
    scales: np.ndarray,
    shifts: np.ndarray,
    n_frames: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill failed fits, interpolate onto every frame, then optionally smooth."""
    scales = _fill_nans(fitted_indices, scales)
    shifts = _fill_nans(fitted_indices, shifts)

    all_frames = np.arange(n_frames, dtype=float)
    scale = np.interp(all_frames, fitted_indices, scales)
    shift = np.interp(all_frames, fitted_indices, shifts)

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        scale = gaussian_filter1d(scale, sigma=smooth_sigma, mode="nearest")
        shift = gaussian_filter1d(shift, sigma=smooth_sigma, mode="nearest")
    return scale, shift


def _fill_nans(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Replace NaNs by linear interpolation from the surrounding valid samples."""
    valid = ~np.isnan(y)
    if valid.all():
        return y
    return np.interp(x, x[valid], y[valid])


def _iter_depth(depth_maps: DepthMaps, stride: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, depth_map)`` every *stride* frames."""
    with h5py.File(depth_maps.path, "r") as handle:
        images = handle["images"]
        for index in range(0, depth_maps.n_frames, stride):
            yield index, np.asarray(images[index], dtype=np.float32)


@contextlib.contextmanager
def _working_directory(path: Path):
    """Temporarily chdir into *path*."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _suppress_stdout():
    """Silence writes to fd 1, including those from C extensions."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    sys.stdout.flush()
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)
