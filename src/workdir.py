"""Scratch space for the pipeline's large intermediates.

Three files are produced along the way and none of them is a deliverable:

``undistorted.mp4``
    the linearised video (stage 1),
``depth.h5``
    one metric depth map per frame (stage 2) -- this is the big one; a gzip-9
    HDF5 of an 11-minute 1080p run is comfortably over 20 GB,
``sas_corrections.h5``
    the per-frame depth scale and shift (stage 3), a few kilobytes.

:class:`WorkDir` puts them in a fresh directory, hands out their paths, and
removes the lot on exit. It prefers node-local scratch (``$SLURM_TMPDIR`` or
``$TMPDIR``) over ``/tmp``, which matters on a cluster: writing tens of gigabytes
of gzip to a shared filesystem is enormously slower than to local disk, and it is
what the research pipeline's ``VITLOK_DEPTH_TMPDIR`` plumbing existed to avoid.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Optional

from .config import RunConfig


def default_scratch_root() -> Path:
    """Best available scratch root: SLURM node-local, then ``$TMPDIR``, then ``/tmp``."""
    for env_var in ("SLURM_TMPDIR", "TMPDIR"):
        value = os.environ.get(env_var)
        if value and Path(value).is_dir():
            return Path(value)
    return Path(tempfile.gettempdir())


class WorkDir:
    """Context manager owning one run's intermediate files.

    Usage::

        with WorkDir(cfg) as work:
            ...                       # work.undistorted_video, work.depth, ...
        # directory removed here, unless cfg.keep_intermediates

    The directory is kept on an exception regardless of ``keep_intermediates``,
    so a crashed run can be inspected; the path is printed in that case.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.keep = cfg.keep_intermediates
        if cfg.work_dir is not None:
            self.path = cfg.work_dir
            self._owned = False
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root = default_scratch_root() / "video_to_trajectory"
            self.path = root / f"{cfg.run_name}_{stamp}"
            self._owned = True

    # -- the intermediates -------------------------------------------------
    @property
    def undistorted_video(self) -> Path:
        """Stage 1 output: the linearised video."""
        return self.path / "undistorted.mp4"

    @property
    def depth(self) -> Path:
        """Stage 2 output: one metric depth map per frame."""
        return self.path / "depth.h5"

    @property
    def corrections(self) -> Path:
        """Stage 3 output: per-frame depth scale and shift."""
        return self.path / "sas_corrections.h5"

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "WorkDir":
        self.path.mkdir(parents=True, exist_ok=True)
        print(f"[work] intermediates in {self.path}", file=sys.stderr)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if exc_type is not None:
            print(f"[work] run failed; intermediates kept at {self.path}", file=sys.stderr)
            return
        if self.keep:
            print(f"[work] intermediates kept at {self.path}", file=sys.stderr)
            return
        if not self._owned:
            # The user chose this directory; leave their files alone and remove
            # only what we put there.
            for path in (self.undistorted_video, self.depth, self.corrections):
                path.unlink(missing_ok=True)
            print(f"[work] removed intermediates from {self.path}", file=sys.stderr)
            return
        shutil.rmtree(self.path, ignore_errors=True)
        print(f"[work] removed {self.path}", file=sys.stderr)

    def free_bytes(self) -> int:
        """Free space on the filesystem holding this directory."""
        usage = shutil.disk_usage(self.path)
        return usage.free
