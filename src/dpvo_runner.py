"""Stage 4 -- run DPVO over the linear video and bring its output to metric scale.

DPVO (Deep Patch Visual Odometry) tracks small image patches across frames and
solves a sliding-window bundle adjustment. On its own it is monocular and
therefore scale-free: it recovers the *shape* of the camera's path, not its size.

Two mechanisms turn that into a metric trajectory, both fed by the corrected
depth from stage 3:

**Inside the tracker.** The vendored DPVO patch (``third_party/dpvo/PATCHES.md``)
passes the depth map into patch selection and patch inverse-depth
initialisation, so patches are placed on nearby, well-conditioned surfaces and
start at roughly the right depth instead of a random one.

**After the fact** (``--scaling depth_ratio``, the default). For each keyframe the
metric depth is sampled at the patch centres and compared with the inverse depth
DPVO converged on. The ratio ``d_metric * inv_d`` is DPVO's scale error at that
keyframe; it is averaged over patches, interpolated onto every frame, smoothed,
and applied to the incremental displacements before re-integrating. Doing it per
frame rather than once globally lets the correction track scale *drift*, which is
monocular VO's characteristic failure mode.

Patches are weighted towards the middle of the usable depth range: very close
depths are dominated by the network's near-field error and very far ones carry
almost no parallax.

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

# --- rescaling parameters (carried over from the research pipeline) --------

#: Depth range trusted for the scale ratio, in metres. Nearer than this the
#: network's relative error is large; further, patch parallax is too small.
MIN_DEPTH_FOR_RESCALING = 0.6
MAX_DEPTH_FOR_RESCALING = 8.0

#: Inverse depths at or below this are unconverged patches, not real geometry.
MIN_INVERSE_DEPTH = 1e-6

#: Fewest usable patches in a keyframe before its ratio is trusted.
MIN_VALID_PATCHES = 10

#: Gaussian smoothing of the per-frame ratio series, in frames.
RATIO_SMOOTHING_SIGMA = 20

#: Keyframe separation counted as "nearby" when measuring reprojection residuals.
NEARBY_KEYFRAME_WINDOW = 5

#: Blend of the two patch weightings used for the ratio average. The reprojection
#: term is currently disabled (weight 0): it was measured to add noise rather
#: than signal, but it is still computed because the per-frame metric error
#: reported alongside the trajectory is derived from the same residuals.
LAMBDA_DISTANCE = 1.0
LAMBDA_REPROJECTION = 0.0


@dataclass
class TrajectoryResult:
    """A finished trajectory plus what is needed to describe how it was made."""

    txyz: np.ndarray
    """``(N, 4)`` array of ``[time_s, x, y, z]`` in metres, world axes."""
    metric_error: np.ndarray
    """``(N,)`` per-frame self-assessed positional uncertainty, in metres."""
    scaling: str
    """Which scaling was applied (``depth_ratio`` or ``none``)."""
    mean_ratio: float
    ratio_std: float
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
        cfg: Supplies the DPVO weights and config, ``stride`` and ``scaling``.
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

    weights, metric_error = _patch_weights_and_error(
        slam, depth_at_patches, intrinsics, len(poses)
    )
    ratios, densified = _scale_ratios(slam, depth_at_patches, weights, len(poses))

    time_s = tstamps * cfg.stride / video.fps
    # DPVO's camera frame is Y-down / Z-forward; the pipeline works in a world
    # frame with Z up: world x = dpvo x, world y = dpvo z, world z = -dpvo y.
    txyz = np.column_stack([time_s, poses[:, 0], poses[:, 2], -poses[:, 1]])

    if cfg.scaling == "depth_ratio":
        txyz = _apply_ratio_scaling(txyz, densified)

    return TrajectoryResult(
        txyz=txyz,
        metric_error=metric_error,
        scaling=cfg.scaling,
        mean_ratio=float(np.mean(ratios)) if len(ratios) else float("nan"),
        ratio_std=float(np.std(ratios)) if len(ratios) else float("nan"),
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


# --- rescaling ------------------------------------------------------------

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


def _scale_ratios(
    slam, depth_at_patches: dict[int, np.ndarray], weights: np.ndarray, n_poses: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-keyframe scale ratios, densified and smoothed onto every pose.

    For a patch at true metric depth ``d`` that DPVO converged to inverse depth
    ``inv_d``, the product ``d * inv_d`` is 1 when DPVO's scale is correct and
    otherwise is exactly the factor its displacements are off by.
    """
    from scipy import ndimage

    timestamps: list[int] = []
    ratios: list[float] = []

    for i in range(slam.n):
        timestamp = int(slam.pg.tstamps_[i])
        metric_depth = depth_at_patches.get(timestamp)
        if metric_depth is None:
            continue
        inverse_depth = slam.pg.patches_[i, :, 2, 1, 1].cpu().numpy()
        usable = (
            (metric_depth > MIN_DEPTH_FOR_RESCALING)
            & (metric_depth < MAX_DEPTH_FOR_RESCALING)
            & (inverse_depth > MIN_INVERSE_DEPTH)
        )
        if int(usable.sum()) <= MIN_VALID_PATCHES:
            continue

        patch_ratios = metric_depth[usable] * inverse_depth[usable]
        patch_weights = _blend_weights(
            metric_depth[usable], weights[i * slam.M:(i + 1) * slam.M][usable]
        )
        if patch_weights.sum() > 0:
            ratio = float(np.average(patch_ratios, weights=patch_weights))
        else:
            ratio = float(np.median(patch_ratios))
        timestamps.append(timestamp)
        ratios.append(ratio)

    if not ratios:
        print("[dpvo] no usable depth/patch pairs; leaving the scale untouched",
              file=sys.stderr)
        return np.asarray(ratios), np.ones(n_poses)

    ratio_array = np.asarray(ratios, dtype=float)
    densified = interpolate_scattered(np.asarray(timestamps), ratio_array, n_poses)
    densified = ndimage.gaussian_filter1d(densified, sigma=RATIO_SMOOTHING_SIGMA)
    print(f"[dpvo] scale ratio {ratio_array.mean():.4f} +/- {ratio_array.std():.4f} "
          f"over {len(ratio_array)} keyframes")
    return ratio_array, densified


def _blend_weights(metric_depth: np.ndarray, reprojection_weight: np.ndarray) -> np.ndarray:
    """Weight patches by depth (and, if enabled, reprojection confidence).

    The depth term is an inverted parabola peaking midway between
    :data:`MIN_DEPTH_FOR_RESCALING` and :data:`MAX_DEPTH_FOR_RESCALING` and
    falling to zero at both ends.
    """
    span = MAX_DEPTH_FOR_RESCALING - MIN_DEPTH_FOR_RESCALING
    curvature = -4.0 / span ** 2
    distance_weight = np.clip(
        curvature
        * (metric_depth - MIN_DEPTH_FOR_RESCALING)
        * (metric_depth - MAX_DEPTH_FOR_RESCALING),
        0, 1,
    )
    return (
        LAMBDA_DISTANCE * distance_weight
        + LAMBDA_REPROJECTION * reprojection_weight
    ) / (LAMBDA_DISTANCE + LAMBDA_REPROJECTION)


def _apply_ratio_scaling(txyz: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """Rescale incremental displacements by *ratios* and re-integrate.

    Scaling the steps rather than the positions is what lets a *drifting* scale
    be corrected: each segment of the path is stretched by the ratio that was
    valid while the camera travelled it.
    """
    scaled = txyz.copy()
    steps = np.diff(txyz[:, 1:], axis=0)
    scaled[1:, 1:] = txyz[0, 1:] + np.cumsum(steps * ratios[1:, None], axis=0)
    return scaled


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
