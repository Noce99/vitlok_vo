"""Reading and writing the pipeline's one deliverable.

A run produces three files::

    <output>/<video stem>/
    |-- trajectory.txt    time x y z metric_error, one row per frame
    |-- metadata.json     how that trajectory was produced
    `-- trajectory.png    top-down (x, y) plot of the path

``trajectory.txt`` is whitespace-separated with a one-line header, so
``np.loadtxt(path, skiprows=1)`` reads it and so does a spreadsheet. Coordinates
are metres in a local world frame with **Z up**, origin at the first pose.
``metric_error`` is the tracker's own per-frame positional uncertainty estimate,
in metres (see :mod:`src.dpvo_runner`).

The research pipeline instead wrote timestamped files into one of seventeen
``trajectories_dpvo_<variant>_<network>/`` folders, and readers picked whichever
had the newest name. That existed to let many experimental variants coexist; with
one variant per run, a fixed name and a metadata file say more and guess less.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: Column header written at the top of ``trajectory.txt``.
HEADER = "time x y z metric_error"

TRAJECTORY_NAME = "trajectory.txt"
METADATA_NAME = "metadata.json"
PLOT_NAME = "trajectory.png"


def save_trajectory(
    result,
    cfg,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write ``trajectory.txt`` and ``metadata.json`` into ``cfg.run_dir``.

    Args:
        result: The :class:`~src.dpvo_runner.TrajectoryResult` to write.
        cfg: The run configuration, recorded verbatim in the metadata.
        extra: Additional metadata (stage timings, depth and ground-fit
            summaries) merged into the JSON.

    Returns:
        Path to the written ``trajectory.txt``.
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.run_dir / TRAJECTORY_NAME
    np.savetxt(destination, result.as_columns(), fmt="%.6f",
               header=HEADER, comments="")

    metadata: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "tool": "video_to_trajectory",
        "git_commit": _git_commit(),
        "config": cfg.to_dict(),
        "trajectory": {
            "n_poses": int(result.n_poses),
            "n_keyframes": int(result.n_keyframes),
            "duration_s": float(result.txyz[-1, 0]) if len(result.txyz) else 0.0,
            "path_length_m": path_length(result.txyz),
        },
    }
    if extra:
        metadata.update(extra)
    (cfg.run_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")

    plot_path = plot_trajectory(result.txyz, cfg.run_dir / PLOT_NAME)

    print(f"[output] {destination}")
    print(f"[output] {cfg.run_dir / METADATA_NAME}")
    print(f"[output] {plot_path}")
    return destination


def plot_trajectory(txyz: np.ndarray, destination: Path) -> Path:
    """Plot a top-down (x, y) view of a ``[time, x, y, z]`` trajectory.

    The path is drawn east-north (x, y), matching the ENU world frame the rest
    of the pipeline uses, with start and end points marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 8))
    if len(txyz):
        axis.plot(txyz[:, 1], txyz[:, 2], color="tab:blue", linewidth=1.2)
        axis.scatter(txyz[0, 1], txyz[0, 2], color="tab:green", s=60,
                     zorder=3, label="start")
        axis.scatter(txyz[-1, 1], txyz[-1, 2], color="tab:red", s=60,
                     zorder=3, label="end")
        axis.legend(fontsize=8)
    axis.set_xlabel("east (m)")
    axis.set_ylabel("north (m)")
    axis.set_title(f"Trajectory -- {path_length(txyz):.1f} m")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def load_trajectory(path: Path) -> np.ndarray:
    """Read a ``trajectory.txt`` into an ``(N, 4)`` or ``(N, 5)`` array.

    Tolerates files with or without the header line, and files written without
    the ``metric_error`` column (as older runs were).

    Raises:
        ValueError: if the file has too few columns to be a trajectory.
    """
    path = Path(path)
    skip = 0
    with path.open() as handle:
        first = handle.readline().strip()
    if first:
        try:
            float(first.split()[0])
        except (ValueError, IndexError):
            skip = 1

    data = np.loadtxt(path, skiprows=skip, ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(
            f"{path}: expected at least 4 columns (time x y z), "
            f"found {data.shape[1]}"
        )
    return data


def load_metadata(path: Path) -> dict[str, Any]:
    """Read the ``metadata.json`` sitting next to a trajectory, or ``{}``."""
    candidate = Path(path)
    if candidate.is_file() and candidate.name == METADATA_NAME:
        return json.loads(candidate.read_text())
    sibling = candidate.parent / METADATA_NAME
    if sibling.is_file():
        return json.loads(sibling.read_text())
    return {}


def path_length(txyz: np.ndarray) -> float:
    """Total distance travelled along a trajectory, in metres."""
    if len(txyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(txyz[:, 1:4], axis=0), axis=1).sum())


def _git_commit() -> Optional[str]:
    """Short commit hash of this repository, when it is a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
