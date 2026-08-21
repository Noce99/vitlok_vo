"""Stage 1 (360 variant) -- extract a rectilinear view out of an equirectangular video.

Used instead of :mod:`src.undistortion` whenever ``--360-camera-model`` is given.
There is no lens distortion to remove here; instead a virtual pinhole camera,
pointed in the direction given by ``--360-direction``, is resampled out of the
source sphere with a single ``cv2.remap`` lookup table built once from
:func:`src.camera360.build_view` and reused for every frame. The result is a
:class:`~src.undistortion.LinearVideo` exactly like the normal pipeline
produces, so every later stage is unaware a 360 source was ever involved.
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera360 import build_view, parse_direction, resolve_camera_model
from .config import RunConfig
from .undistortion import SIZE_MULTIPLE, LinearVideo, _FrameWriter
from .workdir import WorkDir


def undistort_360_video(cfg: RunConfig, work: WorkDir) -> LinearVideo:
    """Extract the view described by ``cfg``'s 360 settings into a linear video.

    Args:
        cfg: Supplies the source video, ``--360-camera-model``,
            ``--360-camera-params``, ``--360-direction``, ``--360-fov`` and the
            output size.
        work: Destination for the linear video.

    Returns:
        A :class:`LinearVideo` describing the file that was written.
    """
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

    width, height = _cropped_size(cfg.three_sixty_out_width, cfg.three_sixty_out_height)
    map_x, map_y, calibration = build_view(
        model, direction, source_w, source_h, width, height, cfg.three_sixty_fov,
    )

    print(f"[undistort360] {model.name} {source_w}x{source_h} -> {width}x{height} "
          f"(fov {cfg.three_sixty_fov:.1f} deg, "
          f"pitch/yaw/roll={direction[0]:.1f}/{direction[1]:.1f}/{direction[2]:.1f}), "
          f"{total} frames @ {fps:.3f} fps")

    writer = _FrameWriter(work.undistorted_video, width, height, fps, cfg.undistort_crf)
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rectilinear = cv2.remap(
                frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            writer.write(rectilinear)
            written += 1
            if written % 200 == 0:
                print(f"\r[undistort360] {written}/{total} frames", end="", flush=True)
    finally:
        capture.release()
        writer.close()
    print(f"\r[undistort360] {written} frames -> {work.undistorted_video}")

    if written == 0:
        raise RuntimeError(f"{cfg.video}: no frames could be read")

    _save_preview(cfg, work, map_x, map_y)

    return LinearVideo(
        path=work.undistorted_video,
        calibration=calibration,
        fps=fps,
        n_frames=written,
        width=width,
        height=height,
    )


def _cropped_size(width: int, height: int) -> tuple[int, int]:
    """*width*/*height* cropped down to a multiple of :data:`SIZE_MULTIPLE`."""
    w = width - width % SIZE_MULTIPLE
    h = height - height % SIZE_MULTIPLE
    if w <= 0 or h <= 0:
        raise ValueError(
            f"--360-out-width/--360-out-height {width}x{height} is below one "
            f"{SIZE_MULTIPLE}-pixel block"
        )
    return w, h


def _save_preview(cfg: RunConfig, work: WorkDir, map_x: np.ndarray, map_y: np.ndarray) -> None:
    """Write a side-by-side source/extracted still, resized to match.

    Checking this image is the quickest way to confirm ``--360-direction``
    points where intended.
    """
    capture = cv2.VideoCapture(str(cfg.video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return
    rectilinear = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    source_preview = cv2.resize(frame, (rectilinear.shape[1], rectilinear.shape[0]))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.run_dir / "undistortion_preview.jpg"
    cv2.imwrite(str(destination), np.hstack((source_preview, rectilinear)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"[undistort360] preview -> {destination}")
