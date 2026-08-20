"""Paired RGB + depth frame reader for DPVO.

Because stage 1 already linearised the video and stage 2 stored exactly one
depth map per video frame, this reader is far simpler than its counterpart in the
research pipeline: there is no undistortion to apply, no separate "depth creation
stride" to reconcile, and no resolution mismatch to clamp. Dataset index ``i`` is
video frame ``i * stride`` is depth frame ``i * stride`` -- one index, no drift.

That indexing is worth being explicit about. The original applied the per-frame
depth corrections at ``correction[i]`` while reading depth frame
``i * depth_stride``, so any stride above 1 silently slid the corrections out of
step with the depth they were correcting. Here both use the same index.

Frames are yielded in **BGR** order, which is what DPVO expects: its own
``stream.py`` feeds it ``cv2.imread`` output without a colour conversion.
"""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue
from typing import Iterator, Optional

import cv2
import h5py
import numpy as np
import torch

from .calibration import Calibration

#: Depth is floored here after correction. A sufficiently negative fitted shift
#: would otherwise drive near depths to zero or below, making the inverse depth
#: DPVO initialises its patches with infinite, which NaNs out the bundle
#: adjustment a few frames later.
MIN_VALID_DEPTH = 0.1

#: Depth maps read from HDF5 per slab. Reading one frame at a time from a
#: gzip-compressed dataset is dominated by chunk decompression.
DEPTH_SLAB = 256

#: Frames buffered ahead of the consumer by the reader thread.
QUEUE_DEPTH = 64


class RGBDStream:
    """Iterable over ``(image, depth, intrinsics)`` tensors for DPVO.

    Each iteration yields:

    * ``image``: ``uint8`` tensor of shape ``(1, 3, H, W)``, BGR,
    * ``depth``: ``float32`` tensor of shape ``(H, W)``, metres, corrected,
    * ``intrinsics``: ``float32`` tensor ``[fx, fy, cx, cy]``.

    Reading runs on a background thread so video decoding and HDF5 decompression
    overlap with the caller's GPU work.
    """

    def __init__(
        self,
        video_path: Path,
        depth_path: Path,
        calibration: Calibration,
        stride: int = 1,
        corrections_path: Optional[Path] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.depth_path = Path(depth_path)
        self.calibration = calibration
        self.stride = stride

        with h5py.File(self.depth_path, "r") as handle:
            depth_frames, self.depth_height, self.depth_width = handle["images"].shape

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.video_path}")
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        # The two must line up frame for frame; a mismatch means the depth file
        # belongs to a different video (or an interrupted run).
        self.n_frames = min(video_frames, depth_frames)
        if abs(video_frames - depth_frames) > 1:
            raise ValueError(
                f"video and depth disagree on length: {video_frames} video frames "
                f"vs {depth_frames} depth maps ({self.depth_path.name}). "
                "The depth file does not belong to this video."
            )

        self.scale, self.shift = self._load_corrections(corrections_path)
        # Depth may be stored smaller than the video; resize it up to match.
        self._depth_resize = (
            None if (self.depth_width, self.depth_height) == (self.width, self.height)
            else (self.width, self.height)
        )
        self._intrinsics = np.array(
            [calibration.fx, calibration.fy, calibration.cx, calibration.cy],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(range(0, self.n_frames, self.stride))

    def _load_corrections(
        self, corrections_path: Optional[Path]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Read the per-frame scale and shift, if stage 3 produced any."""
        if corrections_path is None:
            return None, None
        with h5py.File(corrections_path, "r") as handle:
            scale = np.asarray(handle["scale"], dtype=np.float32)
            shift = np.asarray(handle["shift"], dtype=np.float32)
        if len(scale) < self.n_frames:
            raise ValueError(
                f"{corrections_path.name} covers {len(scale)} frames but the "
                f"video has {self.n_frames}"
            )
        return scale, shift

    def correct(self, depth: np.ndarray, frame_index: int) -> np.ndarray:
        """Apply frame *frame_index*'s scale and shift, then floor the result."""
        if self.scale is None or self.shift is None:
            return depth
        index = min(frame_index, len(self.scale) - 1)
        corrected = self.scale[index] * depth + self.shift[index]
        return np.maximum(corrected, MIN_VALID_DEPTH)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        queue: Queue = Queue(maxsize=QUEUE_DEPTH)
        thread = threading.Thread(target=self._read_into, args=(queue,), daemon=True)
        thread.start()
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        thread.join()

    def _read_into(self, queue: Queue) -> None:
        """Reader thread: walk the video forwards, pulling depth in slabs."""
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            with h5py.File(self.depth_path, "r") as handle:
                dataset = handle["images"]
                slab: Optional[np.ndarray] = None
                slab_start = -1
                position = 0

                for frame_index in range(0, self.n_frames, self.stride):
                    # Sequential reads only -- seeking per frame is far slower.
                    while position < frame_index:
                        capture.read()
                        position += 1
                    ok, frame = capture.read()
                    position += 1
                    if not ok:
                        break

                    if slab is None or not (
                        slab_start <= frame_index < slab_start + len(slab)
                    ):
                        slab_start = frame_index
                        slab = dataset[frame_index:frame_index + DEPTH_SLAB]
                    depth = np.asarray(slab[frame_index - slab_start], dtype=np.float32)
                    depth = self.correct(depth, frame_index)
                    if self._depth_resize is not None:
                        depth = cv2.resize(depth, self._depth_resize,
                                           interpolation=cv2.INTER_LINEAR)

                    queue.put((
                        torch.as_tensor(frame).permute(2, 0, 1)[None],
                        torch.as_tensor(depth),
                        torch.as_tensor(self._intrinsics),
                    ))
        except Exception as exc:  # surfaced on the consumer's thread
            queue.put(exc)
        finally:
            capture.release()
            queue.put(None)
