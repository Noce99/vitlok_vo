"""SLURM job files for running the pipeline on a cluster.

One run of ``video_to_trajectory.py`` is one GPU job: stages 2 and 4 want a GPU,
stages 1 and 3 are CPU-bound but far too short to be worth splitting out.

Two things in the generated script matter more than the rest:

``--work-dir "$TMPDIR"``
    puts the multi-gigabyte depth file on **node-local** scratch. Writing
    gzip-compressed HDF5 to a shared filesystem is dramatically slower and
    unkind to everyone else on it.

``--mem``
    is generous by default. Depth inference holds a frame batch in RAM while the
    HDF5 layer buffers compressed chunks, and a job that is killed for memory
    three hours in has wasted three hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

#: Cluster profiles. ``account`` may contain ``{kind}``, replaced by ``gpu``/``cpu``.
CLUSTERS: dict[str, dict] = {
    "arrhenius": {
        "partition": "gpu",
        "account": "naiss2025-22-1304-{kind}",
        "gpu_directive": "--gpus {n}",
        "modules": [],
    },
    "disi": {
        "partition": "sbuild,mi210,l40s,l40",
        "account": None,
        "gpu_directive": "--gres=gpu:{n}",
        "modules": [],
    },
}

DEFAULT_CLUSTER = "arrhenius"

#: Shared between the log filename and the default output directory, so both
#: name a job by the same moment even though SLURM has no date pattern of its own.
STAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"


def now_stamp() -> str:
    """The timestamp tagging a generated job's log file and default output dir."""
    return datetime.now().strftime(STAMP_FORMAT)


def default_output_dir(video: Optional[Path], videos_root: Path) -> Optional[Path]:
    """Where a job's trajectory should land when *video* lives under *videos_root*.

    Keeps results next to the source clip -- ``videos/<name>/results/``
    -- instead of the shared ``./output`` directory that direct CLI runs default
    to, so a video's outputs stay grouped with its inputs. Returns ``None`` (and
    lets the pipeline fall back to ``./output``) for videos outside *videos_root*.
    """
    if video is None:
        return None
    video = Path(video).resolve()
    try:
        video.relative_to(Path(videos_root).resolve())
    except ValueError:
        return None
    return video.parent / "results"


@dataclass
class JobSpec:
    """Everything that varies between one generated job and the next."""

    name: str
    email: str
    cluster: str = DEFAULT_CLUSTER
    command: str = ""
    repo_root: Path = Path(__file__).resolve().parent.parent
    venv: Optional[Path] = None
    gpus: int = 1
    cpus: int = 16
    memory: str = "300G"
    time_limit: str = "03:00:00"
    account: Optional[str] = None
    modules: Sequence[str] = field(default_factory=list)

    def resolved_account(self) -> Optional[str]:
        """The ``-A`` value, from the explicit override or the cluster profile."""
        if self.account:
            return self.account
        template = CLUSTERS[self.cluster]["account"]
        return template.format(kind="gpu") if template else None


def render(spec: JobSpec, log_dir: Path, stamp: Optional[str] = None) -> str:
    """Render *spec* into the text of an sbatch script.

    *stamp* defaults to now; pass one explicitly to share it with a
    :func:`default_output_dir` computed before the spec's command was built.
    """
    if spec.cluster not in CLUSTERS:
        raise ValueError(
            f"unknown cluster {spec.cluster!r}; known: {sorted(CLUSTERS)}"
        )
    profile = CLUSTERS[spec.cluster]

    # SLURM has no date pattern for log names, so the generation time is baked in
    # and %j (the job id) keeps repeated submissions distinct.
    if stamp is None:
        stamp = now_stamp()
    log_file = log_dir / f"{spec.name}_{stamp}.%j.log"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.name}",
        "#SBATCH --mail-type=ALL",
        f"#SBATCH --mail-user={spec.email}",
        "#SBATCH --nodes=1",
        f"#SBATCH -t {spec.time_limit}",
        f"#SBATCH --partition={profile['partition']}",
    ]
    account = spec.resolved_account()
    if account:
        lines.append(f"#SBATCH -A {account}")
    if spec.gpus:
        lines.append("#SBATCH " + profile["gpu_directive"].format(n=spec.gpus))
    lines += [
        f"#SBATCH --cpus-per-task={spec.cpus}",
        f"#SBATCH --mem={spec.memory}",
        f"#SBATCH --chdir={log_dir}",
        f"#SBATCH -o {log_file} # STDOUT",
        f"#SBATCH -e {log_file} # STDERR",
        "",
        "set -euo pipefail",
        "",
        f"cd {spec.repo_root}",
    ]
    for module in list(profile["modules"]) + list(spec.modules):
        lines.append(f"module load {module}")
    if spec.venv:
        lines.append(f'source "{spec.venv}/bin/activate"')
    lines += [
        "",
        "# Keep the multi-gigabyte intermediates on node-local scratch: writing",
        "# gzip HDF5 to a shared filesystem is far slower and hurts other users.",
        'export WORK_DIR="${TMPDIR:-/tmp}/video_to_trajectory_${SLURM_JOB_ID:-$$}"',
        'mkdir -p "$WORK_DIR"',
        'trap \'rm -rf "$WORK_DIR"\' EXIT',
        "",
        spec.command,
        "",
    ]
    return "\n".join(lines)


def write(spec: JobSpec, destination: Path, stamp: Optional[str] = None) -> Path:
    """Write the rendered job to *destination*, creating its log folder alongside.

    Returns:
        The path of the written ``.sbatch`` file.
    """
    destination = Path(destination)
    log_dir = destination.parent / destination.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(spec, log_dir, stamp=stamp))
    destination.chmod(0o755)
    return destination


def build_command(
    video: Optional[Path],
    config: Optional[Path],
    calibration: Optional[Path],
    camera_height: Optional[float],
    output: Optional[Path],
    depth_model: Optional[str],
    extra: Sequence[str] = (),
    run_name: Optional[str] = None,
) -> str:
    """Assemble the ``video_to_trajectory.py`` invocation for the job body."""
    parts = ["python video_to_trajectory.py"]
    if video is not None:
        parts.append(f'"{video}"')
    if config is not None:
        parts.append(f'--config "{config}"')
    if calibration is not None:
        parts.append(f'--calibration "{calibration}"')
    if camera_height is not None:
        parts.append(f"--camera-height {camera_height}")
    if depth_model is not None:
        parts.append(f"--depth-model {depth_model}")
    if output is not None:
        parts.append(f'--out "{output}"')
    if run_name is not None:
        parts.append(f'--run-name "{run_name}"')
    parts.append('--work-dir "$WORK_DIR"')
    # DPVO's default patch buffer (4096) is sized for short clips; cluster jobs
    # run long enough videos that it overflows ("buffer size is too small").
    parts.append("--dpvo-opts BUFFER_SIZE 16384")
    parts.extend(extra)
    return " \\\n    ".join(parts)
