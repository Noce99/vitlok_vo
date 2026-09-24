"""Stage 4 -- run DPVO over the linear video and bring its output to metric scale.

DPVO (Deep Patch Visual Odometry) tracks small image patches across frames and
solves a sliding-window bundle adjustment. On its own it is monocular and
therefore scale-free: it recovers the *shape* of the camera's path, not its size.

The vendored DPVO patch (``third_party/dpvo/PATCHES.md``) turns that into a metric
trajectory by feeding it the corrected depth from stage 3 twice: it informs patch
selection and patch inverse-depth initialisation, and bundle adjustment then holds
each patch's inverse depth **fixed** at that value and solves for camera poses
only. A patch's depth never drifts from its metric-informed initial value, so
poses come out fully determined relative to it, rather than sitting in the flat,
weakly-conditioned direction that lets ordinary monocular BA's scale drift over a
sequence -- no post-hoc rescaling of the trajectory is needed.

The trajectory is also annotated with a per-frame ``metric_error``: the mean
reprojection residual over nearby keyframe edges, converted from pixels into
metres via ``pixel_error * median_depth / fx``. It is a self-assessment, computed
without any ground truth, and the evaluation script draws it as a growing
uncertainty disc along the estimated path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import RunConfig
from .depth import DepthMaps
from .ground_scale_shift import ScaleShiftSeries
from .rgbd_stream import RGBDStream
from .undistortion import LinearVideo

# --- metric-error self-assessment parameters --------------------------------

#: Depth range trusted for the metric-error estimate, in metres. Nearer than this
#: the network's relative error is large; further, patch parallax is too small.
MIN_DEPTH_FOR_RESCALING = 0.6

#: Keyframe separation counted as "nearby" when measuring reprojection residuals.
NEARBY_KEYFRAME_WINDOW = 5


@dataclass
class TrajectoryResult:
    """A finished trajectory plus what is needed to describe how it was made."""

    txyz: np.ndarray
    """``(N, 4)`` array of ``[time_s, x, y, z]`` in metres, world axes."""
    metric_error: np.ndarray
    """``(N,)`` per-frame self-assessed positional uncertainty, in metres."""
    n_keyframes: int
    n_poses: int

    def as_columns(self) -> np.ndarray:
        """``(N, 5)`` array laid out as the ``trajectory.txt`` columns."""
        return np.column_stack([self.txyz, self.metric_error])


@torch.no_grad()
def run_dpvo(
    cfg: RunConfig,
    video: LinearVideo,
    depth_maps: DepthMaps,
    corrections: Optional[ScaleShiftSeries],
) -> TrajectoryResult:
    """Track *video* with DPVO and return a metric-scale trajectory.

    The whole stage runs under ``torch.no_grad()``: DPVO's poses and patches are
    leaf tensors that require grad, so reprojecting them afterwards to measure
    residuals would otherwise build a graph over the entire sequence.

    Args:
        cfg: Supplies the DPVO weights and config, and ``stride``.
        video: The linear video from stage 1.
        depth_maps: Stage 2's depth file.
        corrections: Stage 3's per-frame scale and shift, applied while reading.

    Returns:
        A :class:`TrajectoryResult` in world coordinates with Z up.
    """
    _add_dpvo_to_path(cfg)
    from dpvo.config import cfg as dpvo_cfg

    dpvo_cfg.merge_from_file(str(cfg.dpvo_config))
    if cfg.dpvo_opts:
        dpvo_cfg.merge_from_list(cfg.dpvo_opts)
    if cfg.random_patch_ratio is not None:
        dpvo_cfg.CENTROID_SEL_RANDOM_RATIO = cfg.random_patch_ratio

    stream = RGBDStream(
        video_path=video.path,
        depth_path=depth_maps.path,
        calibration=video.calibration,
        stride=cfg.stride,
        corrections_path=corrections.path if corrections is not None else None,
    )
    print(f"[dpvo] tracking {len(stream)} frames "
          f"(stride {cfg.stride}), patch selection "
          f"{dpvo_cfg.CENTROID_SEL_STRAT} @ {dpvo_cfg.CENTROID_SEL_RANDOM_RATIO}")

    slam, depth_at_patches, intrinsics, shape = _track(stream, dpvo_cfg,
                                                       cfg.dpvo_weights)
    poses, tstamps = slam.terminate()

    _, metric_error = _patch_weights_and_error(
        slam, depth_at_patches, intrinsics, len(poses)
    )

    time_s = tstamps * cfg.stride / video.fps
    # DPVO's camera frame is Y-down / Z-forward; the pipeline works in a world
    # frame with Z up: world x = dpvo x, world y = dpvo z, world z = -dpvo y.
    txyz = np.column_stack([time_s, poses[:, 0], poses[:, 2], -poses[:, 1]])

    return TrajectoryResult(
        txyz=txyz,
        metric_error=metric_error,
        n_keyframes=int(slam.n),
        n_poses=int(len(poses)),
    )


# --- tracking -------------------------------------------------------------

@torch.no_grad()
def _track(stream: RGBDStream, dpvo_cfg, weights_path: Path):
    """Feed every frame to DPVO, recording metric depth at each keyframe's patches.

    Returns ``(slam, depth_at_patches, intrinsics, (H, W))`` where
    *depth_at_patches* maps a keyframe timestamp to the ``(96,)`` metric depths
    sampled at that keyframe's patch centres.
    """
    from dpvo.dpvo import DPVO

    slam = None
    depth_at_patches: dict[int, np.ndarray] = {}
    intrinsics_t = None
    height = width = None
    total = len(stream)

    for index, (image, depth, intrinsics_t) in enumerate(stream):
        image = image[0].cuda()
        intrinsics_t = intrinsics_t.cuda()

        if slam is None:
            _, height, width = image.shape
            slam = DPVO(dpvo_cfg, str(weights_path), ht=height, wd=width, viz=False)

        slam(index, image, intrinsics_t, depth)

        # patches_[slam.n - 1] is always the most recently accepted keyframe,
        # even after the ring buffer shifts. Key by its timestamp so the mapping
        # survives keyframe removal.
        if slam.n > 0:
            timestamp = int(slam.pg.tstamps_[slam.n - 1])
            if timestamp not in depth_at_patches:
                patches = slam.pg.patches_[slam.n - 1]
                # patches is (96, 3, 3, 3): 96 patches, channels (x, y, inv_depth),
                # over a 3x3 patch footprint in feature space. [1, 1] is its centre.
                u = (patches[:, 0, 1, 1] * slam.RES).long().clamp(0, width - 1)
                v = (patches[:, 1, 1, 1] * slam.RES).long().clamp(0, height - 1)
                depth_at_patches[timestamp] = (
                    depth.squeeze()[v.cpu(), u.cpu()].numpy()
                )
        if index % 100 == 0:
            print(f"\r[dpvo] {index}/{total} frames, {slam.n} keyframes",
                  end="", flush=True)
    print()

    if slam is None:
        raise RuntimeError("DPVO processed no frames; is the linear video empty?")
    return slam, depth_at_patches, intrinsics_t, (height, width)


# --- self-assessment -------------------------------------------------------

def _patch_weights_and_error(
    slam, depth_at_patches: dict[int, np.ndarray], intrinsics, n_poses: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch reprojection confidence, and a per-frame metric error estimate.

    Each patch is reprojected with the final converged poses and inverse depths
    and compared against the optical-flow target stored when its edge was
    deactivated. Only edges between keyframes at most
    :data:`NEARBY_KEYFRAME_WINDOW` apart are used: long-range edges were driven
    to zero residual by the global bundle adjustment, so their residuals say
    nothing about depth consistency.
    """
    from dpvo import projective_ops as pops
    from dpvo.lietorch import SE3

    n_patches = slam.n * slam.M
    kk = torch.cat([slam.pg.kk_inac, slam.pg.kk])
    ii = torch.cat([slam.pg.ii_inac, slam.pg.ii])
    jj = torch.cat([slam.pg.jj_inac, slam.pg.jj])
    targets = torch.cat([slam.pg.target_inac[0], slam.pg.target[0]])

    nearby = ((jj - ii).abs() <= NEARBY_KEYFRAME_WINDOW) & (kk < n_patches)
    metric_error_per_keyframe = np.full(slam.n, np.nan, dtype=np.float32)

    if int(nearby.sum()) == 0:
        print("[dpvo] no nearby edges; falling back to uniform patch weights",
              file=sys.stderr)
        return np.ones(n_patches, dtype=np.float32), np.zeros(n_poses)

    kk_n, ii_n, jj_n = kk[nearby], ii[nearby], jj[nearby]
    targets_n = targets[nearby].cpu()

    coords = pops.transform(SE3(slam.poses), slam.patches, slam.intrinsics,
                            ii_n, jj_n, kk_n)
    centres = coords[0, :, slam.P // 2, slam.P // 2, :].cpu()
    residuals = (targets_n - centres).norm(dim=-1)

    kk_cpu = kk_n.cpu()
    error_sum = torch.zeros(n_patches)
    counts = torch.zeros(n_patches)
    error_sum.scatter_add_(0, kk_cpu, residuals)
    counts.scatter_add_(0, kk_cpu, torch.ones(kk_cpu.shape[0]))

    mean_error = error_sum / counts.clamp(min=1)
    weights = (1.0 / (1.0 + mean_error))
    weights[counts == 0] = 0.0
    weights = weights.numpy()

    focal_x = float(intrinsics[0].item())
    for i in range(slam.n):
        per_patch = mean_error[i * slam.M:(i + 1) * slam.M]
        has_edges = counts[i * slam.M:(i + 1) * slam.M] > 0
        if int(has_edges.sum()) == 0:
            continue
        pixel_error = float(per_patch[has_edges].mean())
        depths = depth_at_patches.get(int(slam.pg.tstamps_[i]))
        if depths is None or depths.size == 0:
            continue
        usable = depths[depths > MIN_DEPTH_FOR_RESCALING]
        if usable.size:
            # A pixel of reprojection error at depth d subtends d / fx metres.
            metric_error_per_keyframe[i] = pixel_error * float(np.median(usable)) / focal_x

    keyframe_times = np.array([int(slam.pg.tstamps_[i]) for i in range(slam.n)])
    valid = ~np.isnan(metric_error_per_keyframe)
    if valid.any():
        metric_error = interpolate_scattered(
            keyframe_times[valid],
            metric_error_per_keyframe[valid].astype(float),
            n_poses,
        )
    else:
        metric_error = np.zeros(n_poses)
    return weights, metric_error


def interpolate_scattered(
    timestamps: np.ndarray, values: np.ndarray, length: int
) -> np.ndarray:
    """Spread values known at scattered integer timestamps over ``[0, length)``.

    Known samples land on their own index; everything between is linearly
    interpolated, and the series is held constant beyond the first and last
    known sample.
    """
    return np.interp(np.arange(length), timestamps, values,
                     left=values[0], right=values[-1])


def _add_dpvo_to_path(cfg: RunConfig) -> None:
    """Make the vendored DPVO importable."""
    dpvo_root = cfg.dpvo_config.parent.parent
    if str(dpvo_root) not in sys.path:
        sys.path.insert(0, str(dpvo_root))
