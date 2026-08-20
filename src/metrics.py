"""Trajectory error metrics.

Four complementary views of how wrong an estimate is. They disagree on purpose:
a trajectory can have a large ATE while being locally excellent (one early
heading error that everything after inherits), or a small ATE while being locally
poor (errors that cancel).

**ATE** -- Absolute Trajectory Error. Distance between corresponding points after
alignment. The intuitive "how far off was it", but dominated by early errors,
since a trajectory cannot recover from a wrong turn.

**RTE** -- Relative Trajectory Error. Over every sub-segment spanning a fixed
number of points, compare the estimate's displacement with the truth's. Being a
difference of displacements it is blind to global position, so it measures local
drift only.

**FPE** -- Final-Point Error, divided by the number of points. The raw
end-of-run distance grows with run length; normalising makes runs of different
length comparable.

**KITTI** -- drift over fixed-*distance* sub-trajectories, in percent of distance
travelled and degrees per metre. See :mod:`src.kitti_metrics`.

All of these are computed in 2D (x, y). Height is measured against GPS altitude,
which is far less accurate than its horizontal fix, so including it would mostly
measure the ground truth's own error.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .alignment import interpolate_gt_at
from .kitti_metrics import kitti_metrics

#: Points spanned by each RTE sub-segment.
RTE_DELTA = 100

#: Sub-trajectory lengths in metres for the KITTI metrics. Shorter than KITTI's
#: own 100-800 m because these are walking sequences, not motorway drives.
KITTI_LENGTHS = (50, 100, 150, 200, 300)


def paired(gt_txy: np.ndarray, txy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth and estimate as matching ``(N, 2)`` position arrays.

    The ground truth is interpolated onto the estimate's timestamps and both are
    cropped to where the ground truth actually covers the estimate.
    """
    interpolated, crop = interpolate_gt_at(gt_txy.copy(), txy)
    return txy[:crop, 1:3], interpolated[:crop, 1:3]


def compute_ate(gt_txy: np.ndarray, txy: np.ndarray) -> dict:
    """Absolute Trajectory Error statistics, in metres."""
    estimate, truth = paired(gt_txy, txy)
    return _stats(np.linalg.norm(estimate - truth, axis=1))


def compute_rte(gt_txy: np.ndarray, txy: np.ndarray,
                delta: int = RTE_DELTA) -> dict:
    """Relative Trajectory Error over *delta*-point sub-segments, in metres."""
    estimate, truth = paired(gt_txy, txy)
    if len(estimate) <= delta:
        return _stats(np.array([np.nan]))
    estimate_steps = estimate[delta:] - estimate[:-delta]
    truth_steps = truth[delta:] - truth[:-delta]
    return _stats(np.linalg.norm(estimate_steps - truth_steps, axis=1))


def compute_fpe(gt_txy: np.ndarray, txy: np.ndarray) -> dict:
    """Final-Point Error: end-of-run distance, and that divided by point count."""
    estimate, truth = paired(gt_txy, txy)
    n = len(estimate)
    if n == 0:
        return {"n": 0, "final_distance_m": float("nan"), "value": float("nan")}
    final = float(np.linalg.norm(estimate[-1] - truth[-1]))
    return {"n": n, "final_distance_m": final, "value": final / n}


def compute_kitti(gt_txy: np.ndarray, txy: np.ndarray,
                  lengths=KITTI_LENGTHS) -> Optional[dict]:
    """KITTI drift metrics, or ``None`` when the run is too short for any segment."""
    estimate, truth = paired(gt_txy, txy)
    if len(estimate) < 2:
        return None
    try:
        return kitti_metrics(estimate, truth, lengths=lengths)
    except ValueError:
        return None


def compute_all(gt_txy: np.ndarray, txy: np.ndarray,
                rte_delta: int = RTE_DELTA,
                kitti_lengths=KITTI_LENGTHS) -> dict:
    """Every metric at once, keyed ``ate``, ``rte``, ``fpe``, ``kitti``."""
    return {
        "ate": compute_ate(gt_txy, txy),
        "rte": compute_rte(gt_txy, txy, delta=rte_delta),
        "fpe": compute_fpe(gt_txy, txy),
        "kitti": compute_kitti(gt_txy, txy, lengths=kitti_lengths),
    }


def per_point_errors(gt_txy: np.ndarray, txy: np.ndarray) -> np.ndarray:
    """Distance from the ground truth at each estimated pose, in metres."""
    estimate, truth = paired(gt_txy, txy)
    return np.linalg.norm(estimate - truth, axis=1)


def _stats(errors: np.ndarray) -> dict:
    """Summary statistics shared by the ATE and RTE tables."""
    return {
        "n": int(len(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "max": float(np.max(errors)),
    }


def format_tables(metrics: dict, label: str = "estimate",
                  rte_delta: int = RTE_DELTA,
                  kitti_lengths=KITTI_LENGTHS) -> str:
    """Render :func:`compute_all`'s output as printable tables."""
    from prettytable import PrettyTable

    lines: list[str] = []

    table = PrettyTable(["Metric", "Points", "RMSE (m)", "Mean (m)",
                         "Median (m)", "Std (m)", "Max (m)"])
    table.float_format = "0.3"
    for key, name in (("ate", "ATE"), ("rte", f"RTE ({rte_delta}-pt segments)")):
        s = metrics[key]
        table.add_row([name, s["n"], s["rmse"], s["mean"],
                       s["median"], s["std"], s["max"]])
    lines.append(f"=== {label} ===")
    lines.append(str(table))

    fpe = metrics["fpe"]
    lines.append(
        f"\nFinal-point error: {fpe['final_distance_m']:.2f} m "
        f"over {fpe['n']} points ({fpe['value']:.5f} m/point)"
    )

    kitti = metrics.get("kitti")
    if kitti:
        kitti_table = PrettyTable(["Length (m)", "Trans (%)", "Rot (deg/m)"])
        kitti_table.float_format = "0.4"
        for length in kitti_lengths:
            entry = kitti["per_length"].get(length)
            if not entry or entry[0] is None:
                kitti_table.add_row([length, "-", "-"])
            else:
                kitti_table.add_row([length, entry[0], entry[1]])
        kitti_table.add_row(["all", kitti["trans_error_pct"],
                             kitti["rot_error_deg_per_m"]])
        lines.append(f"\n=== KITTI drift ({kitti['num_segments']} segments) ===")
        lines.append(str(kitti_table))
    else:
        lines.append("\nKITTI drift: run too short for any sub-trajectory.")

    return "\n".join(lines)
