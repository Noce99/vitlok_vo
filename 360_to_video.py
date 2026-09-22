#!/usr/bin/env python3
"""Linearize a video -- stage 1 of video_to_trajectory.py, on its own.

    python 360_to_video.py walk.mp4 --calibration calibration/gopro_SW.txt
    python 360_to_video.py GS010427.mp4 --360-camera-model --360-direction "0,-110,0"

Runs only :mod:`src.undistortion` (or :mod:`src.undistortion_360` when
``--360-camera-model`` is given) and writes the resulting distortion-free video
to ``<output>/<video stem>/linear.mp4``, alongside the same
``undistortion_preview.jpg`` sanity check that ``video_to_trajectory.py``
produces. Useful for inspecting the linearised footage, or feeding it to other
tools, without paying for depth, ground-plane fitting or DPVO.

Accepts the same flags and ``--config`` YAML as ``video_to_trajectory.py``;
anything belonging to a later stage (``--camera-height``, ``--depth-model``,
...) is simply unused.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import shutil
import sys
import time

from src.config import RunConfig, build_parser
from src.undistortion import undistort_video
from src.undistortion_360 import undistort_360_video
from src.workdir import WorkDir


def main() -> int:
    parser = build_parser()
    parser.prog = "360_to_video.py"
    parser.description = (
        "Linearize a video: undistort a normal-lens one, or extract a "
        "rectilinear view out of a 360 one. Stage 1 of video_to_trajectory.py, "
        "on its own."
    )
    cfg = RunConfig.from_args(parser.parse_args())
    start = time.perf_counter()

    with WorkDir(cfg) as work:
        if cfg.three_sixty_camera_model is not None:
            video = undistort_360_video(cfg, work)
        else:
            video = undistort_video(cfg, work)

        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        destination = cfg.run_dir / "linear.mp4"
        shutil.move(str(video.path), str(destination))

    print(f"[done] {destination} ({time.perf_counter() - start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
