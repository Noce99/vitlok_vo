"""Stage 1 -- turn a distorted video into a linear (rectilinear) one.

Every later stage assumes a pinhole camera with no distortion. Doing the
undistortion once, up front, and writing a real video file buys three things:

* **Correctness.** The research pipeline undistorted the RGB frames on the fly
  but never the depth maps, so metric depth ended up being sampled at
  *undistorted* pixel coordinates out of a *distorted* depth image. Predicting
  depth from already-linear frames removes the mismatch entirely.
* **Simplicity.** Downstream code carries four intrinsics and no distortion
  model; there is no calibration file to thread through the depth stage, the
  ground fit and DPVO.
* **Inspectability.** The linear video can be played back and eyeballed, which is
  the fastest way to catch a wrong calibration file.

``cv2.undistort`` is called with the *input* camera matrix rather than an optimal
new one, so ``fx, fy, cx, cy`` survive the operation unchanged (up to the resize
applied here) and the frame keeps its field of view, with invalid border pixels
left black.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import Calibration, load_calibration
from .config import RunConfig
from .workdir import WorkDir

#: Frame dimensions are cropped down to a multiple of this. DPVO's feature
#: pyramid needs the image size to divide evenly, and h264's yuv420p needs even
#: dimensions; 8 satisfies both.
SIZE_MULTIPLE = 8


@dataclass(frozen=True)
class LinearVideo:
    """A distortion-free video plus the intrinsics that describe it."""

    path: Path
    calibration: Calibration
    """Pinhole intrinsics of :attr:`path`; distortion is all zeros."""
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def undistort_video(cfg: RunConfig, work: WorkDir) -> LinearVideo:
    """Undistort, resize and re-encode ``cfg.video`` into ``work.undistorted_video``.

    Args:
        cfg: Supplies the source video, calibration file, ``resize`` and CRF.
        work: Destination for the linear video.

    Returns:
        A :class:`LinearVideo` describing the file that was written.
    """
    calibration = load_calibration(cfg.calibration)
    capture = cv2.VideoCapture(str(cfg.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {cfg.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise RuntimeError(f"{cfg.video}: unusable frame rate ({fps})")

    width, height = _output_size(source_w, source_h, cfg.resize)
    # Cropping trims the right and bottom edges only, so the principal point is
    # untouched; only the resize scales the intrinsics.
    linear_calibration = calibration.scaled(cfg.resize).without_distortion()

    if calibration.is_linear:
        print("[undistort] calibration has no distortion; resizing only")

    print(f"[undistort] {source_w}x{source_h} -> {width}x{height}, "
          f"{total} frames @ {fps:.3f} fps")

    writer = _FrameWriter(work.undistorted_video, width, height, fps, cfg.undistort_crf)
    K, dist = calibration.K, calibration.dist
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not calibration.is_linear:
                frame = cv2.undistort(frame, K, dist)
            if cfg.resize != 1.0:
                frame = cv2.resize(frame, None, fx=cfg.resize, fy=cfg.resize,
                                   interpolation=cv2.INTER_AREA)
            writer.write(frame[:height, :width])
            written += 1
            if written % 200 == 0:
                print(f"\r[undistort] {written}/{total} frames", end="", flush=True)
    finally:
        capture.release()
        writer.close()
    print(f"\r[undistort] {written} frames -> {work.undistorted_video}")

    if written == 0:
        raise RuntimeError(f"{cfg.video}: no frames could be read")

    _save_preview(cfg, work, calibration)

    return LinearVideo(
        path=work.undistorted_video,
        calibration=linear_calibration,
        fps=fps,
        n_frames=written,
        width=width,
        height=height,
    )


def _output_size(width: int, height: int, resize: float) -> tuple[int, int]:
    """Resized frame size, cropped down to a multiple of :data:`SIZE_MULTIPLE`."""
    w = int(width * resize)
    h = int(height * resize)
    w -= w % SIZE_MULTIPLE
    h -= h % SIZE_MULTIPLE
    if w <= 0 or h <= 0:
        raise ValueError(
            f"resize {resize} reduces {width}x{height} below one "
            f"{SIZE_MULTIPLE}-pixel block"
        )
    return w, h


def _save_preview(cfg: RunConfig, work: WorkDir, calibration: Calibration) -> None:
    """Write a side-by-side original/undistorted still next to the trajectory.

    Checking this image is the quickest way to confirm the right calibration file
    was used: a wrong one bends straight lines the wrong way.
    """
    capture = cv2.VideoCapture(str(cfg.video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return
    undistorted = cv2.undistort(frame, calibration.K, calibration.dist)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    destination = cfg.run_dir / "undistortion_preview.jpg"
    cv2.imwrite(str(destination), np.hstack((frame, undistorted)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"[undistort] preview -> {destination}")


class _FrameWriter:
    """Encodes BGR frames to h264, preferring ffmpeg for its CRF control.

    ``cv2.VideoWriter`` offers no quality knob, so frames are piped to ffmpeg
    when it is available and only fall back to OpenCV's ``mp4v`` encoder when it
    is not.
    """

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int) -> None:
        self.path = path
        self._process: Optional[subprocess.Popen] = None
        self._writer: Optional[cv2.VideoWriter] = None

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", str(crf), "-pix_fmt", "yuv420p",
                str(path),
            ]
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
            return

        print("[undistort] ffmpeg not found; falling back to OpenCV mp4v "
              "(--undistort-crf ignored)", file=sys.stderr)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open a video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            assert self._writer is not None
            self._writer.write(frame)

    def close(self) -> None:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            if self._process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed while writing {self.path}")
            self._process = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
