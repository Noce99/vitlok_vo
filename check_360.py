#!/usr/bin/env python3
"""Preview a --360-direction by extracting a short rectilinear clip from it.

    python check_360.py --config example_360.yaml
    python check_360.py 360_walk.mp4 --360-camera-model gopromax \
        --360-direction "0,90,0" --duration 10

Runs exactly the extraction that stage 1 of ``video_to_trajectory.py`` would
run in 360 mode (:mod:`src.undistortion_360`), but over a short clip instead
of the whole video, so a chosen ``--360-camera-model``/``--360-direction``/
``--360-fov`` combination can be eyeballed in seconds rather than waiting for
a full run's undistortion stage. Takes the same ``--config`` file or the same
``--360-*`` flags ``video_to_trajectory.py`` does; anything unrelated to the
360 extraction (``--camera-height``, depth/DPVO settings, ...) is accepted but
ignored, so an existing config for a real run can be pointed at directly.

The clip is written to ``<output>/<video stem>/check_360.mp4``.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from src.camera360 import build_view, parse_direction, resolve_camera_model
from src.config import RunConfig, build_parser
from src.undistortion import _FrameWriter
from src.undistortion_360 import _cropped_size


def main() -> int:
    parser = build_parser()
    parser.prog = "check_360.py"
    parser.description = __doc__.split("\n\n")[0]
    parser.add_argument("--duration", type=float, default=20.0,
                         help="Length of the extracted clip, in seconds.")
    parser.add_argument("--start", type=float, default=0.0,
                         help="Where in the source video to start, in seconds.")
    parser.add_argument("--clip-out", type=Path, default=None, dest="clip_out",
                         help="Where to write the extracted clip "
                              "(default: <output>/<video stem>/check_360.mp4).")
    args = parser.parse_args()

    cfg = RunConfig.from_args(args)
    if cfg.three_sixty_camera_model is None:
        parser.error(
            "check_360.py is for previewing --360-direction; pass "
            "--360-camera-model (a bare flag defaults to 'gopromax')"
        )
    if args.duration <= 0:
        parser.error(f"--duration must be positive, got {args.duration}")
    if args.start < 0:
        parser.error(f"--start must be non-negative, got {args.start}")

    model = resolve_camera_model(cfg.three_sixty_camera_model, cfg.three_sixty_camera_params)
    direction = parse_direction(cfg.three_sixty_direction)

    capture = cv2.VideoCapture(str(cfg.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {cfg.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise RuntimeError(f"{cfg.video}: unusable frame rate ({fps})")

    start_frame = round(args.start * fps)
    if start_frame >= total:
        raise ValueError(
            f"--start {args.start}s is past the end of the video "
            f"({total / fps:.1f}s)"
        )
    n_frames = min(round(args.duration * fps), total - start_frame)

    width, height = _cropped_size(cfg.three_sixty_out_width, cfg.three_sixty_out_height)
    map_x, map_y, _ = build_view(
        model, direction, source_w, source_h, width, height, cfg.three_sixty_fov,
    )

    destination = args.clip_out or (cfg.run_dir / "check_360.mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"[check360] {model.name} {source_w}x{source_h} -> {width}x{height} "
          f"(fov {cfg.three_sixty_fov:.1f} deg, "
          f"pitch/yaw/roll={direction[0]:.1f}/{direction[1]:.1f}/{direction[2]:.1f})")
    print(f"[check360] frames {start_frame}..{start_frame + n_frames} of {total} "
          f"({args.start:.1f}s + {n_frames / fps:.1f}s @ {fps:.3f} fps)")

    # CAP_PROP_POS_FRAMES seeks to the nearest keyframe rather than an exact
    # frame -- fine for a quick preview, unlike the frame-exact sequential read
    # the real pipeline uses.
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = _FrameWriter(destination, width, height, fps, cfg.undistort_crf)
    written = 0
    try:
        while written < n_frames:
            ok, frame = capture.read()
            if not ok:
                break
            rectilinear = cv2.remap(
                frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            writer.write(rectilinear)
            written += 1
    finally:
        capture.release()
        writer.close()

    if written == 0:
        raise RuntimeError(f"{cfg.video}: no frames could be read from {args.start}s")

    print(f"[check360] {written} frames -> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
