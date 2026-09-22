#!/usr/bin/env python3
"""
Generate SLURM sbatch files and folders for GoPro video processing.

Usage:
    python make_sbatch.py ELLEN1 GH010050 --camera-height 1.5
    python make_sbatch.py ELLEN1 GH010050 --camera-height 1.5 --radius 5
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------- Configuration ----------
VIDEO_DIR = Path("/nobackup/proj/disk/naiss2025-22-413/shared/data-bike/miei")
SBATCH_BASE = Path("/nobackup/proj/disk/naiss2025-22-413/shared/vitlok_mono_rgb/sbatches")
PROJECT_DIR = Path("/nobackup/proj/disk/naiss2025-22-413/shared/vitlok_mono_rgb")
CALIBRATION = "calibration/gopro_L.txt"
MAIL_USER = "enrico.mannocci99@gmail.com"

# GoPro filename pattern: GH<XX><YYYY>.MP4  -> chapter, file number
GOPRO_RE = re.compile(r"^(G[Hh])(\d{2})(\d{4})\.(MP4|mp4)$")


def parse_gopro_name(name: str):
    """Return (prefix, chapter:int, filenum:str) or None."""
    m = GOPRO_RE.match(Path(name).name)
    if not m:
        return None
    prefix, chapter, filenum, _ = m.groups()
    return prefix.upper(), int(chapter), filenum


def find_consecutive_files(base_name: str, radius: int):
    """
    Find GoPro files whose chapter number is within `radius` of the given
    file's chapter, sharing the same file number (e.g., GH010050, GH020050...).
    """
    parsed = parse_gopro_name(base_name)
    if not parsed:
        print(f"[warn] '{base_name}' does not match GoPro format GH##NNNN.MP4",
              file=sys.stderr)
        return []

    prefix, base_chapter, filenum = parsed
    matches = []

    for f in VIDEO_DIR.iterdir():
        if not f.is_file():
            continue
        p = parse_gopro_name(f.name)
        if not p:
            continue
        f_prefix, f_chapter, f_filenum = p
        if f_prefix == prefix and f_filenum == filenum and \
           abs(f_chapter - base_chapter) <= radius:
            matches.append((f_chapter, f))

    matches.sort(key=lambda x: x[0])
    return [m[1] for m in matches]


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={mail_user}
#SBATCH --nodes=1
#SBATCH -t 04:00:00
#SBATCH --partition=gpu
#SBATCH -A naiss2025-22-1304-gpu
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --chdir={sbatch_dir}
#SBATCH -o {sbatch_dir}/{file_stem}_{timestamp}.%j.log # STDOUT
#SBATCH -e {sbatch_dir}/{file_stem}_{timestamp}.%j.log # STDERR

set -euo pipefail

cd {project_dir}
source "{project_dir}/venv/bin/activate"

# Keep the multi-gigabyte intermediates on node-local scratch: writing
# gzip HDF5 to a shared filesystem is far slower and hurts other users.
export WORK_DIR="${{TMPDIR:-/tmp}}/video_to_trajectory_${{SLURM_JOB_ID:-$$}}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

python video_to_trajectory.py \\
    "{video_path}" \\
    --calibration "{calibration}" \\
    --camera-height {camera_height} \\
    --work-dir "$WORK_DIR" \\
    --dpvo-opts BUFFER_SIZE 16384
"""


def write_sbatch_for(video_file: Path, job_name: str, camera_height: float,
                     sbatch_base: Path, project_dir: Path,
                     calibration: str, mail_user: str):
    stem = video_file.stem  # e.g. GH010050
    sbatch_dir = sbatch_base / stem
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    sbatch_file = sbatch_base / f"{stem}.sbatch"
    if sbatch_file.exists():
        print(f"[skip] {sbatch_file} already exists")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        content = SBATCH_TEMPLATE.format(
            job_name=job_name,
            mail_user=mail_user,
            sbatch_dir=str(sbatch_dir),
            file_stem=stem,
            timestamp=timestamp,
            project_dir=str(project_dir),
            video_path=str(video_file),
            calibration=calibration,
            camera_height=camera_height,
        )
        sbatch_file.write_text(content)
        sbatch_file.chmod(0o755)
        print(f"[new]  {sbatch_file}")
        print(f"[new]  {sbatch_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM sbatch files and folders for GoPro videos."
    )
    parser.add_argument("job_name", help="SLURM job name (e.g. ELLEN1)")
    parser.add_argument("file_name",
                        help="Base GoPro file name (e.g. GH010050)")
    parser.add_argument("--camera-height", type=float, required=True,
                        help="Camera height in meters (e.g. 1.5)")
    parser.add_argument("--radius", type=int, default=10,
                        help="How many consecutive chapters to include "
                             "(default: 5)")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR,
                        help=f"Directory with MP4 files (default: {VIDEO_DIR})")
    parser.add_argument("--sbatch-base", type=Path, default=SBATCH_BASE,
                        help=f"Directory for sbatch files/folders "
                             f"(default: {SBATCH_BASE})")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR,
                        help=f"Project root (default: {PROJECT_DIR})")
    parser.add_argument("--calibration", default=CALIBRATION,
                        help=f"Calibration file (default: {CALIBRATION})")
    parser.add_argument("--mail-user", default=MAIL_USER,
                        help=f"Email for SLURM notifications "
                             f"(default: {MAIL_USER})")

    args = parser.parse_args()

    # Normalize the input: accept both "GH010050" and "GH010050.MP4"
    base = args.file_name
    if not base.upper().endswith(".MP4"):
        base = base + ".MP4"

    # Verify the file exists
    base_path = args.video_dir / base
    if not base_path.exists():
        print(f"[error] file not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    files = find_consecutive_files(base, args.radius)
    if not files:
        print(f"[error] no files matched '{base}' in {args.video_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  - {f.name}")

    for f in files:
        write_sbatch_for(
            video_file=f,
            job_name=args.job_name,
            camera_height=args.camera_height,
            sbatch_base=args.sbatch_base,
            project_dir=args.project_dir,
            calibration=args.calibration,
            mail_user=args.mail_user,
        )


if __name__ == "__main__":
    main()
