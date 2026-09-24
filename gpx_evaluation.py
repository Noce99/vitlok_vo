#!/usr/bin/env python3
"""Evaluate an estimated trajectory against a GPS ground truth.

    python gpx_evaluation.py output/walk/trajectory.txt \
        --gpx walk.gpx --start-time 2026-08-20T09:15:00Z \
        --map site.png --world site.pgw --epsg 3006

    python gpx_evaluation.py output/GS010427/trajectory.txt --config example_360.yaml

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

Ground truth comes from ``--gpx``, a GPS telemetry ``--csv`` (e.g. a GoPro GPS5
export -- both cropped to the video's window with ``--start-time``), or a
pre-computed ``--gt-trajectory`` in local metres.

``video_to_trajectory.py --config`` runs this evaluation by itself, right after
the trajectory is written, whenever that config names a ground-truth source.

Any flag can instead be supplied via ``--config``, a YAML file such as
``example_360.yaml``: it may set ``gpx`` or ``gps_csv`` for the ground-truth
path, plus any other flag below by its long-flag name. A flag given on the
command line always overrides the same key in the config file. Unrelated keys
(from a config file shared with ``video_to_trajectory.py``) are ignored.

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
import yaml

from src.alignment import align
from src.diagnostic_plots import plot_diagnostics
from src.gpx import load_ground_truth, read_csv, read_gpx, video_duration_s
from src.gpx_export import trajectory_to_gpx
from src.map_overlay import plot_on_map, plot_without_map
from src.metrics import KITTI_LENGTHS, RTE_DELTA, compute_all, format_tables
from src.trajectory import load_metadata, load_trajectory, path_length

#: Colour of the estimated track on the map, BGR.
ESTIMATE_COLOR = (0, 0, 220)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    trajectory = load_trajectory(args.trajectory)
    metadata = load_metadata(args.trajectory)
    label = args.label or _label_from(metadata, args.trajectory)
    metric_error = trajectory[:, 4] if trajectory.shape[1] > 4 else None

    start_time = _resolve_start_time(args)
    duration = _resolve_duration(args, metadata, trajectory)

    ground_truth = load_ground_truth(
        gpx_path=args.gpx,
        csv_path=args.csv,
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
        "ground_truth": str(args.gpx or args.csv or args.gt_trajectory),
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

    anchor = _track_anchor(args.gpx, args.csv)
    if anchor is not None:
        gpx_path = trajectory_to_gpx(
            trajectory[:, :4], out_dir / "estimate.gpx",
            origin_lat=anchor[0], origin_lon=anchor[1], start_time=start_time,
            name=label,
        )
        print(f"[eval] {gpx_path}")
    else:
        print("[eval] estimate.gpx skipped: --gpx or --csv is needed to anchor "
              "the trajectory to a real-world position", file=sys.stderr)
    return 0


# --- argument handling ----------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("trajectory", type=Path,
                        help="trajectory.txt from video_to_trajectory.py.")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file supplying any flag below by its long-flag "
                             "name (e.g. 'gpx' or 'gps_csv' for ground truth). "
                             "A flag given on the command line wins over the same "
                             "key in the file; keys this script does not "
                             "recognise -- e.g. from a config shared with "
                             "video_to_trajectory.py -- are ignored.")

    truth = parser.add_argument_group("ground truth")
    truth.add_argument("--gpx", type=Path, default=None,
                       help="GPX track recorded alongside the video.")
    truth.add_argument("--csv", type=Path, default=None,
                       help="GPS telemetry CSV (e.g. a GoPro GPS5 export), as an "
                            "alternative to --gpx.")
    truth.add_argument("--gt-trajectory", dest="gt_trajectory", type=Path,
                       default=None,
                       help="Pre-computed 'time x y z' ground truth in local metres.")
    truth.add_argument("--start-time", dest="start_time", default=None,
                       help="UTC time of the video's first frame, as an ISO 8601 "
                            "string or a Unix timestamp. Used to crop the GPX/CSV.")
    truth.add_argument("--video", type=Path, default=None,
                       help="Source video, used only to measure its duration.")
    truth.add_argument("--gps-epsg", dest="gps_epsg", default=None,
                       help="CRS of the GPX/CSV coordinates (default: 4326).")
    truth.add_argument("--gt-axes", dest="gt_axes", choices=("enu", "ned"),
                       default=None,
                       help="Axis order of --gt-trajectory: 'enu' (x east, "
                            "y north) or 'ned' (x north, y east), which is what "
                            "TartanAir and most simulators export (default: "
                            "enu). Getting this wrong inflates every metric, "
                            "because alignment never reflects.")

    mapping = parser.add_argument_group("map")
    mapping.add_argument("--map", type=Path, default=None,
                         help="Georeferenced map image to draw on. Without one, "
                              "both tracks are plotted on a blank background.")
    mapping.add_argument("--world", type=Path, default=None,
                         help="Its world file (.pgw); found automatically if omitted.")
    mapping.add_argument("--epsg", default=None,
                         help="CRS of the map. Required with --map.")
    mapping.add_argument("--map-alpha", dest="map_alpha", type=float, default=None,
                         help="How much of the map shows through, in [0, 1] "
                              "(default: 0.35).")
    mapping.add_argument("--no-crop", dest="no_crop", action="store_true",
                         help="Render the whole map instead of cropping to the track.")

    output = parser.add_argument_group("output")
    output.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: <trajectory>/../evaluation).")
    output.add_argument("--label", default=None,
                        help="Name for the estimate in tables and plots.")
    output.add_argument("--rte-delta", dest="rte_delta", type=int, default=None,
                        help=f"Points spanned by each RTE sub-segment "
                             f"(default: {RTE_DELTA}).")

    args = parser.parse_args(argv)
    if args.config is not None:
        _apply_config(args, args.config)

    args.gps_epsg = args.gps_epsg if args.gps_epsg is not None else "4326"
    args.gt_axes = args.gt_axes if args.gt_axes is not None else "enu"
    args.map_alpha = args.map_alpha if args.map_alpha is not None else 0.35
    args.rte_delta = args.rte_delta if args.rte_delta is not None else RTE_DELTA

    if not args.trajectory.is_file():
        parser.error(f"no such file: {args.trajectory}")
    given = [name for name, value in
             (("--gpx", args.gpx), ("--csv", args.csv),
              ("--gt-trajectory", args.gt_trajectory)) if value is not None]
    if not given:
        parser.error("give --gpx, --csv or --gt-trajectory (as a flag, or as "
                      "'gpx'/'gps_csv'/'gt_trajectory' in --config)")
    if len(given) > 1:
        parser.error(f"{' and '.join(given)} are alternative ground-truth "
                      "sources; give only one")
    if args.map is not None and args.epsg is None:
        parser.error("--map needs --epsg, the CRS the map is georeferenced in")
    return args


#: --config keys accepted by this script, mapped to their argparse dest. A
#: config file may carry a superset of these (e.g. example_360.yaml, shared
#: with video_to_trajectory.py --config); unrecognised keys are ignored rather
#: than rejected.
_CONFIG_KEYS = {
    "gpx": "gpx",
    "gps_csv": "csv",
    "gt_trajectory": "gt_trajectory",
    "start_time": "start_time",
    "video": "video",
    "gps_epsg": "gps_epsg",
    "gt_axes": "gt_axes",
    "map": "map",
    "world": "world",
    "epsg": "epsg",
    "map_alpha": "map_alpha",
    "out": "out",
    "label": "label",
    "rte_delta": "rte_delta",
}

#: Config keys naming a ground-truth source; ``video_to_trajectory.py`` runs
#: this evaluation automatically when its ``--config`` sets any of them.
GROUND_TRUTH_KEYS = ("gpx", "gps_csv", "gt_trajectory")


def config_has_ground_truth(config_path: Path) -> bool:
    """Whether a ``--config`` YAML names a ground-truth source to evaluate against."""
    loaded = yaml.safe_load(Path(config_path).read_text()) or {}
    return isinstance(loaded, dict) and any(
        loaded.get(key) is not None for key in GROUND_TRUTH_KEYS)


#: Which of those dests get coerced to Path, matching their argparse type=.
_CONFIG_PATH_DESTS = {"gpx", "csv", "gt_trajectory", "video", "map", "world", "out"}


def _apply_config(args: argparse.Namespace, config_path: Path) -> None:
    """Fill in flags left unset on the command line from a --config YAML file."""
    if not config_path.is_file():
        raise SystemExit(f"--config: no such file: {config_path}")
    loaded = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"--config must contain a YAML mapping: {config_path}")
    for yaml_key, dest in _CONFIG_KEYS.items():
        if yaml_key not in loaded or getattr(args, dest) is not None:
            continue
        value = loaded[yaml_key]
        setattr(args, dest, Path(value) if dest in _CONFIG_PATH_DESTS else value)


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
    if model:
        return f"DPVO + {model}"
    return path.parent.name or path.stem


def _track_anchor(
    gpx_path: Optional[Path], csv_path: Optional[Path]
) -> Optional[tuple[float, float]]:
    """First fix of the GPX/CSV track, used as the GPX export's origin."""
    if gpx_path is not None:
        lats, lons, _ = read_gpx(gpx_path)
    elif csv_path is not None:
        lats, lons, _ = read_csv(csv_path)
    else:
        return None
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
