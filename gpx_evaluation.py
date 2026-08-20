#!/usr/bin/env python3
"""Evaluate an estimated trajectory against a GPS ground truth.

    python gpx_evaluation.py output/walk/trajectory.txt \
        --gpx walk.gpx --start-time 2026-08-20T09:15:00Z \
        --map site.png --world site.pgw --epsg 3006

Reads a ``trajectory.txt`` written by ``video_to_trajectory.py``, aligns it to
the ground truth with a rotation and a translation -- and deliberately no scale
fit, since scale is the thing being measured -- then reports:

* ``metrics.json`` and a printed table: ATE, RTE, final-point error and
  KITTI-style drift (see :mod:`src.metrics`);
* ``map_overprint.png``: both tracks drawn on the georeferenced map, with the
  tracker's own accumulated uncertainty as discs (see :mod:`src.map_overlay`);
* ``diagnostics.png``: distance, heading and scale-drift panels
  (see :mod:`src.diagnostic_plots`);
* ``estimate.gpx``: the estimate as a GPX track, when an anchor fix is available.

Ground truth comes from ``--gpx`` (cropped to the video's window with
``--start-time``) or from a pre-computed ``--gt-trajectory`` in local metres.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.alignment import align
from src.diagnostic_plots import plot_diagnostics
from src.gpx import load_ground_truth, read_gpx, video_duration_s
from src.gpx_export import trajectory_to_gpx
from src.map_overlay import plot_on_map, plot_without_map
from src.metrics import KITTI_LENGTHS, RTE_DELTA, compute_all, format_tables
from src.trajectory import load_metadata, load_trajectory, path_length

#: Colour of the estimated track on the map, BGR.
ESTIMATE_COLOR = (0, 0, 220)


def main() -> int:
    args = parse_args()

    trajectory = load_trajectory(args.trajectory)
    metadata = load_metadata(args.trajectory)
    label = args.label or _label_from(metadata, args.trajectory)
    metric_error = trajectory[:, 4] if trajectory.shape[1] > 4 else None

    start_time = _resolve_start_time(args)
    duration = _resolve_duration(args, metadata, trajectory)

    ground_truth = load_ground_truth(
        gpx_path=args.gpx,
        gt_trajectory_path=args.gt_trajectory,
        start_time=start_time,
        duration_s=duration,
        target_epsg=args.epsg if args.map else None,
        gps_epsg=args.gps_epsg,
        axes=args.gt_axes,
    )
    print(f"[eval] ground truth: {len(ground_truth)} points, "
          f"{path_length(np.column_stack([ground_truth, np.zeros(len(ground_truth))])):.0f} m")
    print(f"[eval] estimate:     {len(trajectory)} poses, "
          f"{path_length(trajectory):.0f} m")

    aligned = align(ground_truth, trajectory[:, :4].copy())

    out_dir = args.out or args.trajectory.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_all(ground_truth, aligned, rte_delta=args.rte_delta)
    report = format_tables(metrics, label=label, rte_delta=args.rte_delta)
    print()
    print(report)

    (out_dir / "metrics.json").write_text(json.dumps({
        "trajectory": str(args.trajectory),
        "label": label,
        "ground_truth": str(args.gpx or args.gt_trajectory),
        "rte_delta_points": args.rte_delta,
        "kitti_lengths_m": list(KITTI_LENGTHS),
        "gt_points": int(len(ground_truth)),
        "estimate_poses": int(len(trajectory)),
        "gt_path_length_m": float(_gt_length(ground_truth)),
        "estimate_path_length_m": path_length(trajectory),
        "metrics": metrics,
    }, indent=2, default=_jsonable) + "\n")
    print(f"\n[eval] {out_dir / 'metrics.json'}")

    errors = {label: metric_error} if metric_error is not None else None
    tracks = [(aligned, ESTIMATE_COLOR, label)]
    if args.map:
        written = plot_on_map(ground_truth, tracks, args.map, args.world,
                              out_dir / "map_overprint.png", errors=errors,
                              crop=not args.no_crop, map_alpha=args.map_alpha)
    else:
        written = plot_without_map(ground_truth, tracks,
                                   out_dir / "map_overprint.png", errors=errors)
    print(f"[eval] {written}")

    print(f"[eval] {plot_diagnostics(ground_truth, aligned, out_dir / 'diagnostics.png', label=label, metric_error=metric_error)}")

    anchor = _gpx_anchor(args.gpx)
    if anchor is not None:
        gpx_path = trajectory_to_gpx(
            trajectory[:, :4], out_dir / "estimate.gpx",
            origin_lat=anchor[0], origin_lon=anchor[1], start_time=start_time,
            name=label,
        )
        print(f"[eval] {gpx_path}")
    else:
        print("[eval] estimate.gpx skipped: --gpx is needed to anchor the "
              "trajectory to a real-world position", file=sys.stderr)
    return 0


# --- argument handling ----------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("trajectory", type=Path,
                        help="trajectory.txt from video_to_trajectory.py.")

    truth = parser.add_argument_group("ground truth")
    truth.add_argument("--gpx", type=Path, default=None,
                       help="GPX track recorded alongside the video.")
    truth.add_argument("--gt-trajectory", dest="gt_trajectory", type=Path,
                       default=None,
                       help="Pre-computed 'time x y z' ground truth in local metres.")
    truth.add_argument("--start-time", dest="start_time", default=None,
                       help="UTC time of the video's first frame, as an ISO 8601 "
                            "string or a Unix timestamp. Used to crop the GPX.")
    truth.add_argument("--video", type=Path, default=None,
                       help="Source video, used only to measure its duration.")
    truth.add_argument("--gps-epsg", dest="gps_epsg", default="4326",
                       help="CRS of the GPX coordinates.")
    truth.add_argument("--gt-axes", dest="gt_axes", choices=("enu", "ned"),
                       default="enu",
                       help="Axis order of --gt-trajectory: 'enu' (x east, "
                            "y north) or 'ned' (x north, y east), which is what "
                            "TartanAir and most simulators export. Getting this "
                            "wrong inflates every metric, because alignment "
                            "never reflects.")

    mapping = parser.add_argument_group("map")
    mapping.add_argument("--map", type=Path, default=None,
                         help="Georeferenced map image to draw on.")
    mapping.add_argument("--world", type=Path, default=None,
                         help="Its world file (.pgw); found automatically if omitted.")
    mapping.add_argument("--epsg", default=None,
                         help="CRS of the map. Required with --map.")
    mapping.add_argument("--map-alpha", dest="map_alpha", type=float, default=0.35,
                         help="How much of the map shows through, in [0, 1].")
    mapping.add_argument("--no-crop", dest="no_crop", action="store_true",
                         help="Render the whole map instead of cropping to the track.")

    output = parser.add_argument_group("output")
    output.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: <trajectory>/../evaluation).")
    output.add_argument("--label", default=None,
                        help="Name for the estimate in tables and plots.")
    output.add_argument("--rte-delta", dest="rte_delta", type=int, default=RTE_DELTA,
                        help="Points spanned by each RTE sub-segment.")

    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"no such file: {args.trajectory}")
    if args.gpx is None and args.gt_trajectory is None:
        parser.error("give either --gpx or --gt-trajectory")
    if args.map is not None and args.epsg is None:
        parser.error("--map needs --epsg, the CRS the map is georeferenced in")
    return args


def _resolve_start_time(args) -> Optional[float]:
    """Parse ``--start-time`` from an ISO 8601 string or a Unix timestamp."""
    if args.start_time is None:
        return None
    try:
        return float(args.start_time)
    except ValueError:
        pass
    text = str(args.start_time).replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(
            f"--start-time: cannot parse {args.start_time!r}; expected an ISO 8601 "
            "timestamp such as 2026-08-20T09:15:00Z, or a Unix timestamp"
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _resolve_duration(args, metadata: dict, trajectory: np.ndarray) -> Optional[float]:
    """How long the video ran, preferring the most direct evidence available."""
    if args.video is not None and args.video.is_file():
        return video_duration_s(args.video)
    video = metadata.get("video")
    if video and video.get("fps") and video.get("n_frames"):
        return float(video["n_frames"]) / float(video["fps"])
    if len(trajectory):
        return float(trajectory[-1, 0])
    return None


def _label_from(metadata: dict, path: Path) -> str:
    """A readable name for the estimate, from its metadata when possible."""
    config = metadata.get("config", {})
    model = config.get("depth_model")
    scaling = metadata.get("trajectory", {}).get("scaling")
    if model and scaling:
        return f"DPVO + {model} ({scaling})"
    return path.parent.name or path.stem


def _gpx_anchor(gpx_path: Optional[Path]) -> Optional[tuple[float, float]]:
    """First fix of the GPX track, used as the export's origin."""
    if gpx_path is None:
        return None
    lats, lons, _ = read_gpx(gpx_path)
    return float(lats[0]), float(lons[0])


def _gt_length(gt_txy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(gt_txy[:, 1:3], axis=0), axis=1).sum())


def _jsonable(value):
    """Make numpy scalars and tuple keys survive ``json.dumps``."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
