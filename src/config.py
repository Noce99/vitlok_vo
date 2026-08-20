"""Run configuration: the single source of truth for one video-to-trajectory run.

Every parameter the pipeline needs lives in :class:`RunConfig`. Values are resolved
from three layers, each overriding the one below it:

1. command-line flags,
2. a YAML file given with ``--config``,
3. the defaults declared here.

That ordering means a config file can describe a run completely while a flag still
overrides any single field of it -- handy on the cluster, where one YAML per camera
is combined with a per-job ``--video``.

The research pipeline this was extracted from kept the same information in a
central ``data.yaml`` registry keyed by dataset name. Here there is no registry:
a run is described by its video plus the handful of parameters below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Fields coerced to absolute :class:`~pathlib.Path` objects on construction.
PATH_FIELDS = (
    "video", "calibration", "output", "work_dir",
    "depthpro_checkpoint", "dpvo_weights", "dpvo_config",
)

#: Depth backbones understood by ``--depth-model``.
DEPTH_MODELS = ("metric3d", "depthpro")

#: How the DPVO trajectory is brought to metric scale (``--scaling``).
SCALING_MODES = ("depth_ratio", "none")


@dataclass
class RunConfig:
    """Everything needed to turn one video into one trajectory.

    Attributes are grouped by the pipeline stage that consumes them. Paths are
    absolute by the time ``__post_init__`` has run.
    """

    # --- inputs -----------------------------------------------------------
    video: Path
    """Source video. May be distorted; stage 1 linearises it."""

    calibration: Path
    """Camera calibration file, ``fx fy cx cy k1 k2 p1 p2 k3`` on one line."""

    camera_height: float = 1.8
    """Distance in metres from the camera's optical centre to the ground.

    This is the *only* metric reference in the whole pipeline: the ground-plane
    fit in stage 3 recovers depth scale and shift by asserting that the fitted
    ground plane sits exactly this far below the camera. An error here propagates
    proportionally into the trajectory's scale.
    """

    # --- outputs ----------------------------------------------------------
    output: Path = REPO_ROOT / "output"
    """Directory that will hold ``<video stem>/trajectory.txt`` and its metadata."""

    work_dir: Optional[Path] = None
    """Where the large intermediates go. ``None`` selects a fresh directory under
    ``$TMPDIR`` (or ``/tmp``); see :mod:`src.workdir`."""

    keep_intermediates: bool = False
    """Keep the undistorted video, depth and corrections instead of deleting them."""

    # --- stage 1: undistortion -------------------------------------------
    resize: float = 0.5
    """Scale factor applied while writing the linear video. The default halves
    GoPro footage, which is what the research pipeline fed DPVO. Use ``1.0`` for
    already-small or synthetic footage."""

    undistort_crf: int = 14
    """x264 quality for the linear video. Lower is better; 14 is visually lossless."""

    # --- stage 2: depth ---------------------------------------------------
    depth_model: str = "metric3d"
    """Which metric depth backbone to run (see :data:`DEPTH_MODELS`)."""

    depthpro_checkpoint: Optional[Path] = None
    """Checkpoint for ``--depth-model depthpro``; unused by Metric3D."""

    depth_scale: float = 1.0
    """Extra downscale applied to the stored depth maps, relative to the linear
    video. ``1.0`` stores depth at the linear video's own resolution."""

    depth_batch_size: int = 64
    """Frames per HDF5 write. Also the chunk size of the stored dataset."""

    # --- stage 3: ground-plane scale & shift ------------------------------
    sas_stride: int = 20
    """Fit the ground plane every N-th depth frame; the result is interpolated
    back onto every frame. The fit is single-threaded and by far the slowest
    CPU stage, and scale/shift vary slowly, so a large stride costs little."""

    sas_smooth_sigma: float = 0.0
    """Gaussian smoothing (in frames) applied to the fitted scale/shift series.
    ``0`` disables it, matching the research pipeline's behaviour."""

    bottom_fraction: float = 0.6
    """Fraction of image rows, measured from the bottom, searched for ground."""

    max_ground_depth: float = 30.0
    """Points further than this are not considered when fitting the ground."""

    # --- stage 4: DPVO ----------------------------------------------------
    scaling: str = "depth_ratio"
    """How to scale DPVO's output (see :data:`SCALING_MODES`)."""

    stride: int = 1
    """Process every N-th frame of the linear video."""

    random_patch_ratio: Optional[float] = None
    """Overrides DPVO's ``CENTROID_SEL_RANDOM_RATIO``: the fraction of patches
    chosen uniformly at random rather than biased towards nearby depth."""

    dpvo_weights: Path = REPO_ROOT / "third_party" / "dpvo" / "dpvo.pth"
    dpvo_config: Path = REPO_ROOT / "third_party" / "dpvo" / "config" / "default.yaml"
    dpvo_opts: list[str] = field(default_factory=list)
    """Extra ``KEY value`` pairs merged into the DPVO config."""

    def __post_init__(self) -> None:
        for name in PATH_FIELDS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value).expanduser().resolve())
        self.validate()

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        """Fail fast, with a message naming the offending flag."""
        if not self.video.is_file():
            raise FileNotFoundError(f"--video: no such file: {self.video}")
        if not self.calibration.is_file():
            raise FileNotFoundError(f"--calibration: no such file: {self.calibration}")
        if self.depth_model not in DEPTH_MODELS:
            raise ValueError(
                f"--depth-model must be one of {DEPTH_MODELS}, got {self.depth_model!r}"
            )
        if self.scaling not in SCALING_MODES:
            raise ValueError(
                f"--scaling must be one of {SCALING_MODES}, got {self.scaling!r}"
            )
        if self.camera_height <= 0:
            raise ValueError(f"--camera-height must be positive, got {self.camera_height}")
        if not 0 < self.resize <= 1:
            raise ValueError(f"--resize must be in (0, 1], got {self.resize}")
        if not 0 < self.depth_scale <= 1:
            raise ValueError(f"--depth-scale must be in (0, 1], got {self.depth_scale}")
        if not 0 < self.bottom_fraction <= 1:
            raise ValueError(
                f"--bottom-fraction must be in (0, 1], got {self.bottom_fraction}"
            )
        if self.sas_stride < 1:
            raise ValueError(f"--sas-stride must be >= 1, got {self.sas_stride}")
        if self.stride < 1:
            raise ValueError(f"--stride must be >= 1, got {self.stride}")
        if self.random_patch_ratio is not None and not 0 <= self.random_patch_ratio <= 1:
            raise ValueError(
                f"--random-patch-ratio must be in [0, 1], got {self.random_patch_ratio}"
            )
        if self.depth_model == "depthpro":
            if self.depthpro_checkpoint is None:
                raise ValueError(
                    "--depth-model depthpro requires --depthpro-checkpoint "
                    "(see requirements-depthpro.txt)"
                )
            if not self.depthpro_checkpoint.is_file():
                raise FileNotFoundError(
                    f"--depthpro-checkpoint: no such file: {self.depthpro_checkpoint}"
                )

    def check_runtime_requirements(self) -> None:
        """Check what only the tracking stage needs, before the pipeline starts.

        Kept out of :meth:`validate` so a config can be built (and the evaluation
        script used) without a full DPVO install -- but called up front by
        ``video_to_trajectory.py``, because discovering missing weights *after*
        an hour of depth inference would be maddening.
        """
        if not self.dpvo_weights.is_file():
            raise FileNotFoundError(
                f"DPVO weights missing: {self.dpvo_weights}\n"
                "Run ./install.sh, which downloads them."
            )
        if not self.dpvo_config.is_file():
            raise FileNotFoundError(f"DPVO config missing: {self.dpvo_config}")

    # -- derived -----------------------------------------------------------
    @property
    def run_name(self) -> str:
        """Stem of the source video; names the output folder."""
        return self.video.stem

    @property
    def run_dir(self) -> Path:
        """``<output>/<run name>/`` -- where the trajectory is written."""
        return self.output / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serialisable view, used for the run metadata."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out

    # -- construction ------------------------------------------------------
    @classmethod
    def from_cli(cls, argv: Optional[list[str]] = None) -> "RunConfig":
        """Build a config from ``sys.argv`` (or *argv*), honouring ``--config``."""
        parser = build_parser()
        args = parser.parse_args(argv)
        return cls.from_args(args)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunConfig":
        """Merge parsed flags over a ``--config`` YAML over the dataclass defaults.

        Flags default to ``None`` in the parser precisely so that "not given" is
        distinguishable from "given a value that happens to equal the default".
        """
        settings: dict[str, Any] = {}

        if args.config is not None:
            config_path = Path(args.config).expanduser().resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"--config: no such file: {config_path}")
            loaded = yaml.safe_load(config_path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"--config must contain a YAML mapping: {config_path}")
            known = {f.name for f in fields(cls)}
            unknown = set(loaded) - known
            if unknown:
                raise ValueError(
                    f"--config has unknown key(s): {sorted(unknown)}. "
                    f"Known keys: {sorted(known)}"
                )
            settings.update(loaded)

        for f in fields(cls):
            value = getattr(args, f.name, None)
            if value is not None:
                settings[f.name] = value

        missing = [k for k in ("video", "calibration") if k not in settings]
        if missing:
            raise ValueError(
                "missing required setting(s): "
                + ", ".join(f"--{k}" for k in missing)
                + " (pass them as flags or in --config)"
            )

        for key in ("video", "calibration", "output", "work_dir",
                    "depthpro_checkpoint", "dpvo_weights", "dpvo_config"):
            if settings.get(key) is not None:
                settings[key] = Path(settings[key])

        return cls(**settings)


def build_parser() -> argparse.ArgumentParser:
    """The CLI shared by ``video_to_trajectory.py`` and ``make_sbatch.py``.

    Every flag defaults to ``None``; the real defaults live on :class:`RunConfig`.
    """
    p = argparse.ArgumentParser(
        prog="video_to_trajectory.py",
        description="Estimate a metric-scale camera trajectory from a single video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", nargs="?", default=None,
                   help="Video to process (or set 'video:' in --config).")
    p.add_argument("--config", default=None,
                   help="YAML file supplying any of the settings below.")

    g = p.add_argument_group("inputs")
    g.add_argument("--calibration", default=None,
                   help="Calibration file, e.g. calibration/gopro_SW.txt.")
    g.add_argument("--camera-height", dest="camera_height", type=float, default=None,
                   help="Camera-to-ground distance in metres (default: 1.8). "
                        "This sets the trajectory's metric scale.")

    g = p.add_argument_group("outputs")
    g.add_argument("--out", "--output", dest="output", default=None,
                   help="Output directory (default: ./output).")
    g.add_argument("--work-dir", dest="work_dir", default=None,
                   help="Directory for large intermediates "
                        "(default: a fresh folder under $TMPDIR or /tmp).")
    g.add_argument("--keep-intermediates", dest="keep_intermediates",
                   action="store_true", default=None,
                   help="Do not delete the work directory when the run finishes.")

    g = p.add_argument_group("stage 1: undistortion")
    g.add_argument("--resize", type=float, default=None,
                   help="Scale factor for the linear video (default: 0.5).")
    g.add_argument("--undistort-crf", dest="undistort_crf", type=int, default=None,
                   help="x264 CRF for the linear video (default: 14).")

    g = p.add_argument_group("stage 2: depth")
    g.add_argument("--depth-model", dest="depth_model", choices=DEPTH_MODELS,
                   default=None, help="Metric depth backbone (default: metric3d).")
    g.add_argument("--depthpro-checkpoint", dest="depthpro_checkpoint", default=None,
                   help="Checkpoint for --depth-model depthpro.")
    g.add_argument("--depth-scale", dest="depth_scale", type=float, default=None,
                   help="Downscale stored depth relative to the linear video "
                        "(default: 1.0).")

    g = p.add_argument_group("stage 3: ground-plane scale & shift")
    g.add_argument("--sas-stride", dest="sas_stride", type=int, default=None,
                   help="Fit the ground every N-th frame (default: 20).")
    g.add_argument("--sas-smooth-sigma", dest="sas_smooth_sigma", type=float,
                   default=None,
                   help="Gaussian smoothing of the scale/shift series, in frames "
                        "(default: 0, i.e. none).")
    g.add_argument("--bottom-fraction", dest="bottom_fraction", type=float, default=None,
                   help="Fraction of image rows from the bottom searched for "
                        "ground (default: 0.6).")

    g = p.add_argument_group("stage 4: DPVO")
    g.add_argument("--scaling", choices=SCALING_MODES, default=None,
                   help="How to scale the trajectory (default: depth_ratio).")
    g.add_argument("--stride", type=int, default=None,
                   help="Process every N-th frame (default: 1).")
    g.add_argument("--random-patch-ratio", dest="random_patch_ratio", type=float,
                   default=None,
                   help="Override DPVO's CENTROID_SEL_RANDOM_RATIO.")
    g.add_argument("--dpvo-opts", dest="dpvo_opts", nargs="+", default=None,
                   metavar="KEY VALUE",
                   help="Extra DPVO config overrides, e.g. --dpvo-opts PATCHES_PER_FRAME 128.")
    return p
