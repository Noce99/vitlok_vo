#!/usr/bin/env python3
"""Estimate a metric-scale camera trajectory from a single video.

    python video_to_trajectory.py walk.mp4 \
        --calibration calibration/gopro_SW.txt --camera-height 1.8

Four stages run back to back, each implemented in ``src/``:

1. :mod:`src.undistortion` -- undistort the video into a linear one. Pass
   ``--360-camera-model`` to switch to :mod:`src.undistortion_360` instead,
   which extracts a rectilinear view out of an equirectangular 360 video.
2. :mod:`src.depth` -- predict a metric depth map for every frame.
3. :mod:`src.ground_scale_shift` -- recover the depth's true scale and shift by
   fitting the ground plane against the known camera height.
4. :mod:`src.dpvo_runner` -- track with DPVO and apply the depth-derived scale.

The result is ``output/<video stem>/trajectory.txt``; evaluate it against a GPS
track with ``gpx_evaluation.py``. The large intermediates live in a scratch
directory that is removed on the way out -- pass ``--keep-intermediates`` to
retain them.

This file is deliberately thin: it wires the stages together and times them, and
nothing else.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import sys
import time

from src.config import RunConfig
from src.depth import compute_depth
from src.dpvo_runner import run_dpvo
from src.ground_scale_shift import fit_ground_scale_shift
from src.trajectory import save_trajectory
from src.undistortion import undistort_video
from src.undistortion_360 import undistort_360_video
from src.workdir import WorkDir


def main() -> int:
    cfg = RunConfig.from_cli()
    # Fail now rather than after an hour of depth inference.
    cfg.check_runtime_requirements()
    timings: dict[str, float] = {}

    with WorkDir(cfg) as work:
        with _stage("undistort", timings):
            if cfg.three_sixty_camera_model is not None:
                video = undistort_360_video(cfg, work)
            else:
                video = undistort_video(cfg, work)
        with _stage("depth", timings):
            depth_maps = compute_depth(cfg, video, work)
        with _stage("ground_scale_shift", timings):
            corrections = fit_ground_scale_shift(cfg, video, depth_maps, work)
        with _stage("dpvo", timings):
            result = run_dpvo(cfg, video, depth_maps, corrections)

    save_trajectory(result, cfg, extra={
        "stage_seconds": timings,
        "video": {
            "fps": video.fps,
            "n_frames": video.n_frames,
            "width": video.width,
            "height": video.height,
        },
        "depth": {"model": depth_maps.model, "n_frames": depth_maps.n_frames},
        "ground_scale_shift": corrections.summary(),
    })
    print(f"[done] {sum(timings.values()) / 60:.1f} min total")
    return 0


class _stage:
    """Time one stage and announce it."""

    def __init__(self, name: str, timings: dict[str, float]) -> None:
        self.name = name
        self.timings = timings

    def __enter__(self) -> "_stage":
        self.start = time.perf_counter()
        print(f"\n=== {self.name} ===")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self.start
        self.timings[self.name] = elapsed
        if exc_type is None:
            print(f"=== {self.name}: {elapsed:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
