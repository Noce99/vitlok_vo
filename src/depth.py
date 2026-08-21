"""Stage 2 -- predict a metric depth map for every frame of the linear video.

Two backbones are supported:

``metric3d`` (default)
    `Metric3D <https://github.com/YvanYin/Metric3D>`_ ViT-small, pulled through
    ``torch.hub`` -- no checkpoint to manage, and small enough that many runs fit
    on one GPU. BSD-2-Clause.
``depthpro``
    Apple's `DepthPro <https://github.com/apple/ml-depth-pro>`_. Heavier and
    sharper, and it takes the focal length as an input. Its licence is not
    GPL-compatible, so it is neither vendored nor a hard dependency: it is
    imported lazily and only when actually selected. See
    ``requirements-depthpro.txt``.

Output is an HDF5 file with a single ``images`` dataset of shape
``(n_frames, H, W)``, float32, **in metres**, gzip-compressed. Depth is stored
one map per video frame, so a frame index is a depth index is
``time * fps`` -- the research pipeline's separate "depth creation stride" is
gone, and with it a whole class of off-by-N bugs.

A note on absolute accuracy: neither backbone is fed a canonical-camera
correction here, so the depths are metric only up to an unknown scale and shift.
Recovering exactly that scale and shift, from the known camera height, is what
stage 3 does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import h5py
import numpy as np

from .config import RunConfig
from .undistortion import LinearVideo
from .workdir import WorkDir

#: Callable turning one BGR frame into a float32 depth map in metres.
InferFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DepthMaps:
    """Handle on the depth file written by :func:`compute_depth`."""

    path: Path
    model: str
    n_frames: int
    height: int
    width: int
    scale: float
    """Size of a depth map relative to a linear-video frame."""

    def intrinsics_for(self, video: LinearVideo):
        """Intrinsics of the depth maps, i.e. the video's scaled by :attr:`scale`."""
        return video.calibration.scaled(self.scale)


def compute_depth(cfg: RunConfig, video: LinearVideo, work: WorkDir) -> DepthMaps:
    """Run the selected depth backbone over *video* and store the result.

    Args:
        cfg: Supplies ``depth_model``, ``depth_scale`` and the DepthPro checkpoint.
        video: The linear video from stage 1.
        work: Destination for ``depth.h5``.

    Returns:
        A :class:`DepthMaps` handle on the file that was written.
    """
    width = int(video.width * cfg.depth_scale)
    height = int(video.height * cfg.depth_scale)
    # The backbone sees the full-resolution linear frame; only storage is scaled.
    infer = _load_model(cfg, video)

    print(f"[depth] {cfg.depth_model}: {video.n_frames} frames -> "
          f"{width}x{height} maps")

    batch_size = min(cfg.depth_batch_size, video.n_frames)
    with h5py.File(work.depth, "w") as handle:
        dataset = handle.create_dataset(
            "images",
            shape=(video.n_frames, height, width),
            # Resizable so a short read can be truncated rather than leaving a
            # tail of zeros that later stages would treat as valid depth.
            maxshape=(None, height, width),
            dtype=np.float32,
            # One frame per chunk. Stage 3 reads every --sas-stride'th frame, and
            # a multi-frame chunk would make each of those reads decompress every
            # frame in the chunk to use one of them. Sequential readers are
            # unaffected: they decompress the same bytes either way.
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        handle.attrs["model"] = cfg.depth_model
        handle.attrs["model_type"] = "metric"
        handle.attrs["scale"] = cfg.depth_scale
        handle.attrs["fps"] = video.fps

        batch: list[np.ndarray] = []
        start = 0
        written = 0
        for frame in _iter_frames(video):
            depth = infer(frame).astype(np.float32)
            if depth.shape != (height, width):
                depth = cv2.resize(depth, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
            batch.append(depth)
            written += 1
            if len(batch) == batch_size:
                dataset[start:start + len(batch)] = np.stack(batch)
                start += len(batch)
                batch.clear()
                print(f"\r[depth] {written}/{video.n_frames} frames",
                      end="", flush=True)
        if batch:
            dataset[start:start + len(batch)] = np.stack(batch)
            start += len(batch)

        if start != video.n_frames:
            # Truncate rather than leave a tail of zeros that later stages would
            # silently treat as valid depth.
            dataset.resize((start, height, width))
    print(f"\r[depth] {start} maps -> {work.depth} "
          f"({work.depth.stat().st_size / 1e9:.2f} GB)")

    return DepthMaps(
        path=work.depth,
        model=cfg.depth_model,
        n_frames=start,
        height=height,
        width=width,
        scale=cfg.depth_scale,
    )


def _iter_frames(video: LinearVideo) -> Iterator[np.ndarray]:
    """Yield every BGR frame of the linear video, in order."""
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open linear video: {video.path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


# --- backbones ------------------------------------------------------------

def _load_model(cfg: RunConfig, video: LinearVideo) -> InferFn:
    """Instantiate the requested backbone and wrap it as an :data:`InferFn`."""
    if cfg.depth_model == "metric3d":
        return _load_metric3d()
    if cfg.depth_model == "depthpro":
        return _load_depthpro(cfg, video)
    raise ValueError(f"unknown depth model: {cfg.depth_model!r}")


def _shim_mmcv() -> None:
    """Satisfy Metric3D's stray ``mmcv`` import with an ``mmengine`` stand-in.

    ``mono/utils/comm.py`` opens with a bare ::

        from mmcv.utils import collect_env as collect_base_env

    which has no ``mmengine`` fallback -- unlike every other ``mmcv`` import on
    the inference path -- and whose only consumer is commented out a few lines
    below. Taken at face value it would make us build ``mmcv``'s CUDA extensions
    against torch 2.3.1 to import a name nothing calls.

    So we register a minimal ``mmcv.utils`` in :data:`sys.modules` first, backed
    by the real ``mmengine`` implementations. Nothing is stubbed out: these are
    the functions ``mmcv.utils`` would have re-exported anyway. If a genuine
    ``mmcv`` is installed, we leave it well alone.
    """
    import sys
    import types

    if "mmcv" in sys.modules:
        return
    try:
        import mmcv  # noqa: F401  -- a real one is already installed
        return
    except ImportError:
        pass

    from mmengine import Config, DictAction
    from mmengine.utils import get_git_hash
    from mmengine.utils.dl_utils import collect_env

    mmcv = types.ModuleType("mmcv")
    utils = types.ModuleType("mmcv.utils")
    for name, obj in (
        ("Config", Config),
        ("DictAction", DictAction),
        ("collect_env", collect_env),
        ("get_git_hash", get_git_hash),
    ):
        setattr(utils, name, obj)
    mmcv.utils = utils
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.utils"] = utils


def _load_metric3d() -> InferFn:
    """Metric3D ViT-small via ``torch.hub``; weights are fetched on first use."""
    import torch

    _shim_mmcv()
    print("[depth] loading Metric3D (torch.hub: yvanyin/metric3d)")
    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
    model = model.cuda().eval()

    @torch.no_grad()
    def infer(frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        ).cuda()
        predicted, _, _ = model.inference({"input": tensor})
        depth = predicted.squeeze().cpu().numpy()
        # Metric3D predicts at its own canonical resolution; bring it back to the
        # frame's geometry so depth pixels and image pixels correspond.
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        return depth

    return infer


def _load_depthpro(cfg: RunConfig, video: LinearVideo) -> InferFn:
    """Apple DepthPro. Imported lazily; see the module docstring on licensing."""
    import dataclasses

    import torch

    try:
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "--depth-model depthpro needs the 'depth_pro' package, which is not "
            "installed. It is an optional extra because Apple's licence is not "
            "GPL-3.0 compatible:\n"
            "    pip install -r requirements-depthpro.txt"
        ) from exc

    print(f"[depth] loading DepthPro ({cfg.depthpro_checkpoint})")
    config = dataclasses.replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(cfg.depthpro_checkpoint),
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config, device=torch.device("cuda")
    )
    model = model.eval()
    # DepthPro can estimate focal length itself, but we know it exactly.
    focal_px = torch.tensor(float(video.calibration.fx))

    @torch.no_grad()
    def infer(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prediction = model.infer(transform(rgb), f_px=focal_px)
        return prediction["depth"].squeeze().cpu().numpy()

    return infer
