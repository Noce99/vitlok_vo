#!/usr/bin/env python3
"""Generate a SLURM job that runs the pipeline on a cluster.

Run it with no arguments and it asks for what it needs:

    $ python make_sbatch.py
    Email for job notifications: you@example.org
    Cluster [arrhenius]:
    Video to process: /data/walk.mp4
    ...
    written sbatches/walk.sbatch

Every prompt can also be supplied as a flag, so the same script serves scripted
use:

    python make_sbatch.py --email you@example.org --cluster disi \
        --video /data/walk.mp4 --calibration calibration/gopro_SW.txt \
        --camera-height 1.8 --submit

The generated job puts its intermediates on node-local scratch (``$TMPDIR``) and
deletes them on exit, so the multi-gigabyte depth file never touches the shared
filesystem. Add a new cluster by extending ``CLUSTERS`` in :mod:`src.sbatch`.

Copyright (C) 2026 the video_to_trajectory authors.
Licensed under the GNU General Public License v3.0 -- see LICENSE.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.sbatch import (CLUSTERS, DEFAULT_CLUSTER, JobSpec, build_command,
                         default_output_dir, now_stamp, write)

REPO_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = REPO_ROOT / "videos"


def main() -> int:
    args = parse_args()

    email = args.email or prompt("Email for job notifications", required=True)
    cluster = args.cluster or prompt("Cluster", default=DEFAULT_CLUSTER,
                                     choices=tuple(CLUSTERS))

    config = args.config
    video = args.video
    if config is None and video is None:
        answer = prompt("Video to process (or a --config YAML)", required=True)
        path = Path(answer).expanduser()
        if path.suffix.lower() in (".yaml", ".yml"):
            config = path
        else:
            video = path

    calibration = args.calibration
    camera_height = args.camera_height
    if config is None:
        if calibration is None:
            calibration = Path(prompt("Calibration file", required=True)).expanduser()
        if camera_height is None:
            camera_height = float(prompt("Camera height above ground (m)",
                                         default="1.8"))

    name = args.name or (video.stem if video else config.stem)
    venv = args.venv or _guess_venv()

    stamp = now_stamp()
    output = args.out or default_output_dir(video, VIDEOS_DIR)
    # Results beside the clip go in a folder named like the job's log file.
    run_name = stamp if output is not None and args.out is None else None

    spec = JobSpec(
        name=name,
        email=email,
        cluster=cluster,
        repo_root=REPO_ROOT,
        venv=venv,
        gpus=args.gpus,
        cpus=args.cpus,
        memory=args.memory,
        time_limit=args.time,
        account=args.account,
        command=build_command(
            video=video,
            config=config,
            calibration=calibration,
            camera_height=camera_height,
            output=output,
            depth_model=args.depth_model,
            extra=args.extra or (),
            run_name=run_name,
        ),
    )

    destination = args.sbatch_dir / f"{name}.sbatch"
    written = write(spec, destination, stamp=stamp)
    print(f"\nwritten {written}")
    print(f"logs will go to {written.parent / written.stem}/")

    if venv is None:
        print("\nNOTE: no virtualenv was found, so the job does not activate one. "
              "Pass --venv if the cluster needs it.", file=sys.stderr)

    if args.submit:
        print(f"\nsubmitting {written}")
        subprocess.run(["sbatch", str(written)], check=True)
    else:
        print(f"\nsubmit with:  sbatch {written}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--email", default=None,
                        help="Address for SLURM notifications (prompted if omitted).")
    parser.add_argument("--cluster", choices=tuple(CLUSTERS), default=None,
                        help=f"Cluster profile (prompted if omitted; "
                             f"default {DEFAULT_CLUSTER}).")

    job = parser.add_argument_group("what to run")
    job.add_argument("--video", type=Path, default=None)
    job.add_argument("--config", type=Path, default=None,
                     help="Run config YAML, instead of the flags below.")
    job.add_argument("--calibration", type=Path, default=None)
    job.add_argument("--camera-height", dest="camera_height", type=float,
                     default=None)
    job.add_argument("--depth-model", dest="depth_model", default=None,
                     choices=("metric3d", "depthpro"))
    job.add_argument("--out", type=Path, default=None,
                     help="Output directory for the job's trajectory (default: "
                          "videos/<name>/results/<timestamp>/ for a video under "
                          "videos/, otherwise ./output/).")
    job.add_argument("--extra", nargs="+", default=None,
                     help="Extra flags appended verbatim to the command.")

    resources = parser.add_argument_group("resources")
    resources.add_argument("--gpus", type=int, default=1)
    resources.add_argument("--cpus", type=int, default=16)
    resources.add_argument("--memory", default="300G")
    resources.add_argument("--time", default="03:00:00",
                           help="Wall-clock limit, HH:MM:SS.")
    resources.add_argument("--account", default=None,
                           help="Override the cluster profile's SLURM account.")
    resources.add_argument("--venv", type=Path, default=None,
                           help="Virtualenv to activate (auto-detected if omitted).")

    output = parser.add_argument_group("output")
    output.add_argument("--name", default=None,
                        help="Job name and sbatch filename stem.")
    output.add_argument("--sbatch-dir", dest="sbatch_dir", type=Path,
                        default=REPO_ROOT / "sbatches",
                        help="Where to write the .sbatch file.")
    output.add_argument("--submit", action="store_true",
                        help="Run sbatch on the generated file straight away.")
    return parser.parse_args()


def prompt(question: str, default: Optional[str] = None,
           required: bool = False, choices: Optional[tuple] = None) -> str:
    """Ask on the terminal, looping until the answer is acceptable."""
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"{question}{suffix}: ").strip()
        except EOFError:
            raise SystemExit(
                f"\n{question} is needed. Provide it as a flag when stdin is "
                "not a terminal."
            )
        if not answer and default is not None:
            answer = default
        if not answer and required:
            print("  (required)")
            continue
        if choices and answer not in choices:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        return answer


def _guess_venv() -> Optional[Path]:
    """Find a virtualenv to activate: the active one, or one beside the repo."""
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)
    for name in ("venv", ".venv", "venv_3_10"):
        candidate = REPO_ROOT / name
        if (candidate / "bin" / "activate").is_file():
            return candidate
    return None


if __name__ == "__main__":
    sys.exit(main())
