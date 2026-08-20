"""Camera intrinsics: the file format, and how to measure one from a video.

A calibration file is a single line of space-separated numbers::

    fx fy cx cy k1 k2 p1 p2 k3

that is, the four pinhole intrinsics followed by OpenCV's Brown-Conrady
distortion coefficients. Despite the ``fisheye``-flavoured names the cameras were
described with in the research pipeline, this is the *pinhole* model
(``cv2.calibrateCamera`` / ``cv2.undistort``), never ``cv2.fisheye``.

The distortion tail may be any length OpenCV accepts (4, 5, 8, 12 or 14). The
supplied ``calibration/tartan.txt`` has four zeros, since synthetic footage is
already rectilinear; the GoPro files have five coefficients each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

#: Distortion-vector lengths ``cv2.undistort`` accepts.
VALID_DIST_LENGTHS = (4, 5, 8, 12, 14)

#: Inner-corner count of the calibration board (a 10x7-square chessboard).
CHESSBOARD = (9, 6)


@dataclass(frozen=True)
class Calibration:
    """Pinhole intrinsics plus distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    """Distortion coefficients; all-zero means the camera is already rectilinear."""

    @property
    def K(self) -> np.ndarray:
        """The 3x3 camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def is_linear(self) -> bool:
        """True when every distortion coefficient is zero."""
        return bool(np.allclose(self.dist, 0.0))

    def scaled(self, factor: float) -> "Calibration":
        """Intrinsics for an image resized by *factor*.

        Distortion coefficients are dimensionless ratios of normalised image
        coordinates, so they are unaffected by a uniform resize; only the four
        pinhole terms scale.
        """
        return Calibration(
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            dist=self.dist.copy(),
        )

    def without_distortion(self) -> "Calibration":
        """The same pinhole intrinsics with distortion zeroed.

        This is what an image undistorted with :meth:`K` (rather than an optimal
        new camera matrix) is described by: ``cv2.undistort`` keeps the input
        camera matrix, so the focal lengths and principal point survive intact.
        """
        return Calibration(self.fx, self.fy, self.cx, self.cy, np.zeros(5))

    def as_row(self) -> str:
        """Serialise back to the one-line file format."""
        coefficients = " ".join(repr(float(c)) for c in self.dist)
        return f"{self.fx:.3f} {self.fy:.3f} {int(self.cx)} {int(self.cy)} {coefficients}"


def load_calibration(path: Path) -> Calibration:
    """Read a calibration file, rejecting malformed ones with a useful message."""
    path = Path(path)
    values = np.loadtxt(path, delimiter=" ", ndmin=1)
    if values.size < 8:
        raise ValueError(
            f"{path}: expected at least 8 values (fx fy cx cy + >=4 distortion "
            f"coefficients), found {values.size}"
        )
    dist = np.asarray(values[4:], dtype=np.float64)
    if dist.size not in VALID_DIST_LENGTHS:
        raise ValueError(
            f"{path}: {dist.size} distortion coefficients; OpenCV accepts "
            f"{VALID_DIST_LENGTHS}. Got: {dist.tolist()}"
        )
    return Calibration(float(values[0]), float(values[1]),
                       float(values[2]), float(values[3]), dist)


def save_calibration(path: Path, calibration: Calibration) -> None:
    """Write *calibration* to *path* in the one-line file format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibration.as_row() + "\n")


def calibrate_camera_from_video(
    video_path: Path,
    stride: int = 10,
    chessboard_dir: Optional[Path] = None,
    undistortion_dir: Optional[Path] = None,
    max_previews: int = 20,
) -> tuple[Calibration, float]:
    """Estimate intrinsics from a video of a moving chessboard.

    Every *stride*-th frame is searched for the :data:`CHESSBOARD` inner-corner
    grid; detections are refined to sub-pixel accuracy and accumulated, then
    ``cv2.calibrateCamera`` solves for the intrinsics and distortion.

    Object points are laid out in units of one square, so board size never
    enters: it would only scale the (discarded) extrinsic translations, leaving
    the camera matrix and distortion coefficients untouched.

    Args:
        video_path: Video of the board, filling as much of the frame as possible
            and covering the corners at a range of angles.
        stride: Search every N-th frame. Consecutive video frames are nearly
            identical, so a stride costs almost no information.
        chessboard_dir: If given, detections are drawn and saved here.
        undistortion_dir: If given, before/after previews are saved here.
        max_previews: Cap on the number of undistortion previews written.

    Returns:
        ``(calibration, rms_reprojection_error_px)``.

    Raises:
        RuntimeError: if the board was never found.
    """
    video_path = Path(video_path)
    for directory in (chessboard_dir, undistortion_dir):
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)

    cols, rows = CHESSBOARD
    board_points = np.zeros((rows * cols, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    previews: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    index = 0
    searched = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            index += 1
            searched += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
            if not found:
                continue

            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(board_points)
            image_points.append(refined)
            if len(previews) < max_previews:
                previews.append(frame.copy())
            if chessboard_dir is not None:
                drawn = frame.copy()
                cv2.drawChessboardCorners(drawn, CHESSBOARD, refined, found)
                cv2.imwrite(str(Path(chessboard_dir) / f"{index:06d}.png"), drawn)
            print(f"\r  board found in {len(object_points)} of "
                  f"{searched} frames searched", end="", flush=True)
    finally:
        capture.release()
    print()

    if not object_points:
        raise RuntimeError(
            f"chessboard never found in {video_path} "
            f"({searched} of {total} frames searched, stride {stride}).\n"
            f"Expected a {CHESSBOARD[0]}x{CHESSBOARD[1]} inner-corner grid "
            "(a 10x7-square board). Check the board size and that it is fully visible."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    calibration = Calibration(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=np.asarray(dist).ravel(),
    )

    reprojection_error = _mean_reprojection_error(
        object_points, image_points, rvecs, tvecs, K, dist
    )

    if undistortion_dir is not None:
        for i, frame in enumerate(previews):
            side_by_side = np.hstack((frame, cv2.undistort(frame, K, dist)))
            cv2.imwrite(str(Path(undistortion_dir) / f"{i:03d}.png"), side_by_side)

    return calibration, reprojection_error


def _mean_reprojection_error(object_points, image_points, rvecs, tvecs, K, dist) -> float:
    """RMS distance in pixels between detected and reprojected corners."""
    total = 0.0
    for i, points in enumerate(object_points):
        projected, _ = cv2.projectPoints(points, rvecs[i], tvecs[i], K, dist)
        total += cv2.norm(image_points[i], projected, cv2.NORM_L2SQR) / len(projected)
    return float(np.sqrt(total / len(object_points)))
