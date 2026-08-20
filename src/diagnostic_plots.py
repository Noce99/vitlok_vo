"""Diagnostic plots: *where* and *how* an estimate goes wrong, not just how much.

A single ATE number says an estimate is off by so many metres; it does not say
whether the camera turned too sharply, walked too fast, or drifted steadily. The
figures here decompose the trajectory into the quantities that actually go wrong
in monocular odometry and plot each against the ground truth:

**Cumulative distance** -- the clearest read on scale. A line that climbs faster
than the ground truth's means the estimate thinks it travelled further than it
did, and the gap widening over time means scale is drifting rather than merely
being wrong.

**Per-step distance** -- instantaneous speed, in metres per step. Spikes are
tracking failures; a systematic offset is a scale error.

**Heading** and **heading change** -- rotational behaviour. Because ATE is
dominated by early heading errors, a heading plot often explains a bad ATE that
the position plot cannot.

Each quantity is drawn twice: the value alongside the ground truth's, and the
signed or absolute error against it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def trajectory_stats(txy: np.ndarray, smooth: float = 0.0) -> dict:
    """Per-point motion statistics of a ``[time, x, y]`` track.

    Returns a dict of equal-length arrays: ``time`` (relative seconds),
    ``distance`` (metres per step), ``cumulative_distance``, ``heading``
    (radians) and ``heading_change`` (radians per step, wrapped to
    ``[-pi, pi]``).
    """
    from scipy.ndimage import gaussian_filter1d

    time = txy[:, 0] - txy[0, 0]
    dx = np.diff(txy[:, 1])
    dy = np.diff(txy[:, 2])

    step = np.sqrt(dx ** 2 + dy ** 2)
    cumulative = np.concatenate(([0.0], np.cumsum(step)))
    # Repeat the last sample so every series is as long as the trajectory.
    step = np.concatenate((step, step[-1:]))

    heading = np.arctan2(dy, dx)
    heading_change = _wrap(np.diff(heading))
    heading = np.concatenate((heading, heading[-1:]))
    heading_change = np.concatenate((heading_change, [0.0, 0.0]))

    if smooth > 0:
        step = gaussian_filter1d(step, sigma=smooth)
        heading = gaussian_filter1d(heading, sigma=smooth)

    return {
        "time": time,
        "distance": step,
        "cumulative_distance": cumulative,
        "heading": heading,
        "heading_change": heading_change,
    }


def plot_diagnostics(
    gt_txy: np.ndarray,
    txy: np.ndarray,
    destination: Path,
    label: str = "estimate",
    metric_error: Optional[np.ndarray] = None,
) -> Path:
    """Write the six-panel diagnostic figure comparing *txy* against *gt_txy*.

    The ground truth is resampled onto the estimate's timestamps first, so both
    series are directly comparable point for point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .alignment import interpolate_gt_at
    from .metrics import per_point_errors

    gt_at_pred, crop = interpolate_gt_at(gt_txy.copy(), txy)
    estimate = trajectory_stats(txy[:crop])
    truth = trajectory_stats(gt_at_pred[:crop])
    position_error = per_point_errors(gt_txy, txy)

    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    time = estimate["time"]

    _compare(axes[0, 0], time, truth["cumulative_distance"],
             estimate["cumulative_distance"], label,
             "Cumulative distance", "m")
    _compare(axes[1, 0], time, truth["distance"], estimate["distance"], label,
             "Distance per step", "m")
    _compare(axes[2, 0], time, np.degrees(truth["heading"]),
             np.degrees(estimate["heading"]), label, "Heading", "deg")

    axes[0, 1].plot(time[:len(position_error)], position_error,
                    color="tab:red", linewidth=1.2)
    if metric_error is not None:
        cumulative = np.cumsum(metric_error[:crop])
        axes[0, 1].plot(time[:len(cumulative)], cumulative, color="tab:gray",
                        linestyle="--", linewidth=1.2,
                        label="self-reported (cumulative)")
        axes[0, 1].legend(fontsize=8)
    _label(axes[0, 1], "Position error vs ground truth", "m")

    axes[1, 1].plot(time, estimate["cumulative_distance"]
                    - truth["cumulative_distance"],
                    color="tab:red", linewidth=1.2)
    axes[1, 1].axhline(0, color="black", linewidth=0.6)
    _label(axes[1, 1], "Cumulative distance error (scale drift)", "m")

    heading_error = np.degrees(_abs_wrap(estimate["heading"] - truth["heading"]))
    axes[2, 1].plot(time, heading_error, color="tab:red", linewidth=1.2)
    _label(axes[2, 1], "Absolute heading error", "deg")

    for axis in axes[2, :]:
        axis.set_xlabel("Time (s)")

    figure.suptitle(f"{label} vs ground truth", fontsize=13)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


# --- helpers --------------------------------------------------------------

def _compare(axis, time, truth, estimate, label, title, unit) -> None:
    axis.plot(time, truth, color="tab:blue", linewidth=1.2, label="ground truth")
    axis.plot(time, estimate, color="tab:orange", linewidth=1.2, label=label)
    axis.legend(fontsize=8)
    _label(axis, title, unit)


def _label(axis, title: str, unit: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_ylabel(unit)
    axis.grid(alpha=0.3)


def _wrap(angles: np.ndarray) -> np.ndarray:
    """Wrap angles into ``[-pi, pi]``."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _abs_wrap(angles: np.ndarray) -> np.ndarray:
    """Absolute angular difference, never more than pi."""
    return np.abs(_wrap(angles))
