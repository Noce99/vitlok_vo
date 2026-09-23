#!/usr/bin/env python3
"""Import a zipped GoPro clip (video + GPX) into ``videos/<name>/``.

    python import_zip.py GH010050.zip -h 1.8
    python import_zip.py GH010050.zip -h 1.8 --compress

The zip must contain exactly one video and one ``.gpx`` file, and may also
contain a ``.txt`` file whose *last line* is an ffmpeg command that cuts the
video down to the section the GPX covers (as exported by some GoPro tools).
When that file is present, its command is run first; either way, the final
clip's duration is checked against the GPX track's time span before anything
is copied.

Pass ``--compress``/``-c`` to re-encode the final clip with
``libx264``/``aac`` (``crf 23``, ``preset medium``) before it lands in
``videos/<name>/``, trading CPU time for disk space.

The result is ``videos/<name>/<name>.<ext>`` for the video and GPX, plus:

* ``videos/<name>/<name>.yaml`` -- a run config (video, calibration, camera
  height, output dir, GPX) that can be launched directly with
  ``python video_to_trajectory.py --config videos/<name>/<name>.yaml`` and
  shared with ``gpx_evaluation.py --config``;
* ``videos/<name>/<name>.sbatch`` -- a job generated with :mod:`src.sbatch`,
  the same machinery ``make_sbatch.py`` uses, ready to ``sbatch`` on a
  cluster. It runs that same config, overriding only the output to
  ``videos/<name>/results/<timestamp>/``.

Either way results land alongside the source clip rather than in the shared
``./output`` directory. ``videos/`` is gitignored: these are
imported source clips, not something to track in the repo.

All intermediate files (the unzipped clip, the cut, the compressed output)
are written under ``/tmp`` and removed once the import succeeds; they are
left behind if something fails, so a broken import can be inspected.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from make_sbatch import _guess_venv
from src.gpx import read_gpx
from src.sbatch import (CLUSTERS, DEFAULT_CLUSTER, STAMP_FORMAT, JobSpec,
                         build_command, default_output_dir, now_stamp, write)

REPO_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = REPO_ROOT / "videos"
DEFAULT_CALIBRATION = REPO_ROOT / "calibration" / "gopro_L.txt"

#: How far apart the video and GPX durations may be before the import is refused.
DURATION_TOLERANCE_S = 3.0

COMPRESS_ARGS = [
    "-c:v", "libx264", "-crf", "23", "-preset", "medium",
    "-c:a", "aac", "-b:a", "128k",
]


def main() -> int:
    args = parse_args()

    zip_path = args.zip_path.expanduser().resolve()
    if not zip_path.is_file():
        sys.exit(f"[error] not a file: {zip_path}")

    calibration = args.calibration.expanduser().resolve()
    if not calibration.is_file():
        sys.exit(f"[error] calibration file not found: {calibration}")

    name = zip_path.stem
    dest_dir = VIDEOS_DIR / name
    if dest_dir.exists():
        sys.exit(f"[error] {dest_dir} already exists")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        sys.exit("[error] ffmpeg/ffprobe not found on PATH")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"import_zip_{name}_", dir="/tmp"))
    try:
        video_path, gpx_path, txt_path = _extract(zip_path, tmp_dir)
        if txt_path is not None:
            video_path = _run_cut_instructions(txt_path, tmp_dir, ffmpeg)
        _check_durations(video_path, gpx_path, ffprobe)
        if args.compress:
            video_path = _compress(video_path, tmp_dir, ffmpeg)

        dest_dir.mkdir(parents=True)
        final_video = dest_dir / f"{name}{video_path.suffix}"
        final_gpx = dest_dir / f"{name}{gpx_path.suffix}"
        shutil.copy2(video_path, final_video)
        shutil.copy2(gpx_path, final_gpx)
        print(f"[copied] {final_video}")
        print(f"[copied] {final_gpx}")
    except Exception as exc:
        sys.exit(f"[error] {exc}\n[error] import failed; intermediates kept in {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    config_path = _write_config(args, name, final_video, final_gpx, calibration, dest_dir)
    print(f"[written] {config_path}")
    sbatch_path = _write_sbatch(args, name, final_video, config_path, dest_dir)
    print(f"[written] {sbatch_path}")
    print(f"\nrun locally with:  python video_to_trajectory.py --config {config_path}")
    print(f"submit with:       sbatch {sbatch_path}")
    return 0


def _extract(zip_path: Path, tmp_dir: Path) -> tuple[Path, Path, Optional[Path]]:
    """Unzip the video, GPX and (optional) cut-instructions file into *tmp_dir*."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for n in names:
            if n.startswith("/") or ".." in Path(n).parts:
                raise ValueError(f"unsafe path in zip: {n}")

        videos = [n for n in names if Path(n).suffix.lower() in (".mp4", ".mov")]
        gpxs = [n for n in names if Path(n).suffix.lower() == ".gpx"]
        txts = [n for n in names if Path(n).suffix.lower() == ".txt"]
        if len(videos) != 1:
            raise ValueError(f"expected exactly one video in the zip, found {len(videos)}")
        if len(gpxs) != 1:
            raise ValueError(f"expected exactly one .gpx in the zip, found {len(gpxs)}")
        if len(txts) > 1:
            raise ValueError(f"expected at most one .txt in the zip, found {len(txts)}")

        zf.extractall(tmp_dir, members=videos + gpxs + txts)

    txt_path = tmp_dir / txts[0] if txts else None
    return tmp_dir / videos[0], tmp_dir / gpxs[0], txt_path


def _run_cut_instructions(txt_path: Path, tmp_dir: Path, ffmpeg: str) -> Path:
    """Run the ffmpeg command on *txt_path*'s last line; return the cut clip's path."""
    lines = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{txt_path.name} is empty")
    command_line = lines[-1]
    if not command_line.startswith("ffmpeg"):
        raise ValueError(f"last line of {txt_path.name} is not an ffmpeg command: {command_line!r}")

    command = shlex.split(command_line)
    command[0] = ffmpeg
    output_path = tmp_dir / command[-1]

    print(f"[cut] {command_line}")
    subprocess.run(command, cwd=tmp_dir, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"expected cut output {output_path} was not created")
    return output_path


def _video_duration_s(video_path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _gpx_duration_s(gpx_path: Path) -> float:
    _, _, times = read_gpx(gpx_path)
    times = times[~np.isnan(times)]
    if times.size < 2:
        raise ValueError(f"{gpx_path.name}: fewer than two timestamped track points")
    return float(times.max() - times.min())


def _check_durations(video_path: Path, gpx_path: Path, ffprobe: str) -> None:
    video_s = _video_duration_s(video_path, ffprobe)
    gpx_s = _gpx_duration_s(gpx_path)
    if abs(video_s - gpx_s) > DURATION_TOLERANCE_S:
        raise ValueError(
            f"video/GPX length mismatch: video is {video_s:.1f}s, GPX track "
            f"spans {gpx_s:.1f}s (tolerance {DURATION_TOLERANCE_S:.0f}s)"
        )
    print(f"[ok] video {video_s:.1f}s ~ GPX {gpx_s:.1f}s")


def _compress(video_path: Path, tmp_dir: Path, ffmpeg: str) -> Path:
    output_path = tmp_dir / f"{video_path.stem}-compressed{video_path.suffix}"
    command = [ffmpeg, "-y", "-i", str(video_path), *COMPRESS_ARGS, str(output_path)]
    print(f"[compress] {' '.join(command)}")
    subprocess.run(command, check=True)
    return output_path


def _write_config(args: argparse.Namespace, name: str, video_path: Path,
                  gpx_path: Path, calibration: Path, dest_dir: Path) -> Path:
    """Write a ``--config`` YAML describing this clip's run.

    Strings are written as JSON literals, which are valid YAML and safely
    quote any path.
    """
    config_path = dest_dir / f"{name}.yaml"
    config_path.write_text(
        f"# {name} -- imported by import_zip.py from {args.zip_path.name}\n"
        f"#\n"
        f"#   python video_to_trajectory.py --config {config_path}\n"
        f"#   python gpx_evaluation.py {dest_dir / 'results' / '<timestamp>' / 'trajectory.txt'} "
        f"--config {config_path}\n"
        f"\n"
        f"# --- inputs -----------------------------------------------------------------\n"
        f"video: {json.dumps(str(video_path))}\n"
        f"calibration: {json.dumps(str(calibration))}\n"
        f"\n"
        f"# The pipeline's only metric reference -- an error here scales the whole\n"
        f"# trajectory (see CLAUDE.md).\n"
        f"camera_height: {args.height}\n"
        f"\n"
        f"# --- outputs ----------------------------------------------------------------\n"
        f"output: {json.dumps(str(dest_dir / 'results'))}\n"
        f"run_name: {json.dumps(STAMP_FORMAT)}   # results/YYYY_MM_DD_hh_mm_ss/, one per run\n"
        f"\n"
        f"# --- evaluation (read by gpx_evaluation.py, ignored by video_to_trajectory.py)\n"
        f"gpx: {json.dumps(str(gpx_path))}\n"
    )
    return config_path


def _write_sbatch(args: argparse.Namespace, name: str, video_path: Path,
                  config_path: Path, dest_dir: Path) -> Path:
    stamp = now_stamp()
    spec = JobSpec(
        name=name,
        email=args.email,
        cluster=args.cluster,
        repo_root=REPO_ROOT,
        venv=_guess_venv(),
        command=build_command(
            video=None,
            config=config_path,
            calibration=None,
            camera_height=None,
            output=default_output_dir(video_path, VIDEOS_DIR),
            depth_model=None,
            run_name=stamp,
        ),
    )
    return write(spec, dest_dir / f"{name}.sbatch", stamp=stamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--help", action="help", default=argparse.SUPPRESS,
                        help="show this help message and exit")
    parser.add_argument("zip_path", type=Path,
                        help="Exported zip (video + GPX, optionally + cut instructions).")
    parser.add_argument("-h", "--height", type=float, required=True,
                        help="Camera height above ground, in metres.")
    parser.add_argument("-c", "--compress", action="store_true",
                        help="Re-encode the final clip (libx264/aac) before copying it in.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help="Calibration file for the generated config and sbatch job.")
    parser.add_argument("--email", default=_default_email(),
                        help="Address for SLURM notifications in the generated sbatch job.")
    parser.add_argument("--cluster", choices=tuple(CLUSTERS), default=DEFAULT_CLUSTER,
                        help="Cluster profile for the generated sbatch job.")
    args = parser.parse_args()
    if not args.email:
        parser.error("--email is required (no git user.email configured either)")
    return args


def _default_email() -> Optional[str]:
    try:
        result = subprocess.run(["git", "config", "user.email"],
                                capture_output=True, text=True, cwd=REPO_ROOT)
    except FileNotFoundError:
        return None
    email = result.stdout.strip()
    return email or None


if __name__ == "__main__":
    sys.exit(main())
