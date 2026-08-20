#!/usr/bin/env python3
"""Measure a camera's intrinsics from a video of a chessboard.

    python calibrate.py calibration_video/gopro_sw.mp4 --name gopro_SW

Film a 10x7-square chessboard (9x6 inner corners) for 30-60 seconds, moving it
around so it visits the frame's corners and is seen at a range of angles and
distances -- distortion is estimated from how straight lines bend near the edges,
so a board that never leaves the centre yields a poor fit. Every 10th frame is
searched by default.

The result is written to ``calibration/<name>.txt`` as a single line::

    fx fy cx cy k1 k2 p1 p2 k3

which is what ``video_to_trajectory.py --calibration`` expects. Detected boards
and before/after undistortion previews are saved under ``--log-dir`` so the fit
can be checked by eye; straight edges in the scene should come out straight.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.calibration import calibrate_camera_from_video, save_calibration

REPO_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Video of the chessboard.")
    parser.add_argument("--name", default=None,
                        help="Calibration name (default: the video's stem).")
    parser.add_argument("--stride", type=int, default=10,
                        help="Search every N-th frame for the board.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "calibration",
                        help="Where to write <name>.txt.")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "calibration_logs",
                        help="Where to write detection and undistortion previews.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing calibration file.")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"no such file: {args.video}")

    name = args.name or args.video.stem
    destination = args.out_dir / f"{name}.txt"
    if destination.exists() and not args.force:
        parser.error(f"{destination} already exists; pass --force to overwrite")

    logs = args.log_dir / name
    calibration, error = calibrate_camera_from_video(
        args.video,
        stride=args.stride,
        chessboard_dir=logs / "chessboards",
        undistortion_dir=logs / "undistortion",
    )

    save_calibration(destination, calibration)

    print()
    print(f"  fx, fy        {calibration.fx:.3f}, {calibration.fy:.3f}")
    print(f"  cx, cy        {calibration.cx:.1f}, {calibration.cy:.1f}")
    print(f"  distortion    {', '.join(f'{c:+.5f}' for c in calibration.dist)}")
    print(f"  reprojection  {error:.4f} px RMS")
    if error > 1.0:
        print("  NOTE: a reprojection error above ~1 px usually means too few "
              "views, or a board that stayed near the image centre.", file=sys.stderr)
    print()
    print(f"written to {destination}")
    print(f"previews in {logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
