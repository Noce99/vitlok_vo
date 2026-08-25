# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Estimates a **metric-scale camera trajectory** from a single video — no GPS, no IMU,
no stereo. Extracted and cleaned up from the [VitLok](https://git.chalmers.se/mannocci/vitlok)
research repository (`/home/enrico/vitlok` on this machine), which remains the
reference for behavioural parity.

## Commands

```bash
# install (Python 3.10–3.13)
python3.10 -m venv venv && source venv/bin/activate
CUDA_TAG=cu126 ./install.sh          # cu118 / cu121 / cu124 / ... to match the driver
                                      # (Python 3.13 needs cu126+: see install.sh)

# tests — MUST run from the repo root (see "Gotchas")
python -m pytest tests/ -q
python -m pytest tests/test_metrics.py -q                       # one file
python -m pytest tests/test_metrics.py::test_perfect_estimate_scores_zero -q

# the pipeline
python video_to_trajectory.py VIDEO --calibration calibration/gopro_SW.txt --camera-height 1.8
python gpx_evaluation.py output/<stem>/trajectory.txt --gpx TRACK.gpx --start-time <ISO8601>
python calibrate.py CHESSBOARD_VIDEO --name gopro_SW
python make_sbatch.py                # interactive; --email/--cluster to script it
```

There is no linter or formatter configured, and no packaging (`pyproject.toml` /
`setup.py`) for this project itself — `src` is importable only because scripts run
from the repo root.

Useful during development: `--keep-intermediates` retains the scratch directory,
`--resize 0.3` and a short clip make a full run take about a minute.

## Architecture

Four stages run back to back. `video_to_trajectory.py` is deliberately thin — it
wires them together and times them; all logic lives in `src/`.

```
RunConfig ─▶ undistortion ─▶ depth ─▶ ground_scale_shift ─▶ dpvo_runner ─▶ trajectory
             LinearVideo     DepthMaps  ScaleShiftSeries     TrajectoryResult
```

Each stage returns a frozen dataclass that the next consumes; those four types are
the contract between stages, and reading them is the fastest way to understand the
data flow. Large intermediates (linear video, depth HDF5, corrections HDF5) are
owned by `WorkDir` (`src/workdir.py`), which prefers node-local scratch
(`$SLURM_TMPDIR`/`$TMPDIR`) and deletes everything on success — but **keeps it on
exception**, so a crashed run can be inspected.

### The central idea

A monocular camera cannot see scale, and monocular depth networks return depth
that is correct only up to an unknown scale and shift. The pipeline resolves both
with the one distance that is actually known: **the camera's height above the
ground** (`--camera-height`). Stage 3 fits a plane to the ground visible in the
lower part of each frame and asserts it sits exactly that far below the camera,
which pins down `(s, t)` in `d_metric = s·d_pred + t`.

So `--camera-height` is not a tuning knob — it is the measurement everything rests
on, and an error in it propagates proportionally into distance travelled.

### Stage notes

**1 · `undistortion.py`** — `cv2.undistort` with the *input* camera matrix (not an
optimal new one), so `fx, fy, cx, cy` survive intact. Frames are cropped down to a
multiple of `SIZE_MULTIPLE = 8`, trimming right and bottom only so the principal
point is unaffected. Doing this once, up front, is what lets every later stage
carry four intrinsics and no distortion model.

**2 · `depth.py`** — `metric3d` (default, via `torch.hub`, no checkpoint) or
`depthpro` (optional). Writes `images` (float32, metres) to HDF5 with **one depth
map per video frame** and one frame per chunk.

**3 · `ground_scale_shift.py`** — unproject → CSF ground segmentation → averaged
local normals (`normals.py`) → RANSAC + Huber IRLS solve (`solve_scale_shift.py`).
Runs every `--sas-stride` frames (default 20) and interpolates between; failed fits
record NaN and are filled from neighbours. Ground segmentation is **purely
geometric** (Cloth Simulation Filter) — there is no learned segmentation model
anywhere in this pipeline, despite the research repo containing GANav/SegFormer/DFormer.

**4 · `dpvo_runner.py`** — depth reaches DPVO twice: inside the tracker (patch
placement and inverse-depth init, via the vendored patch) and afterwards, where
metric depth at each keyframe's patch centres is compared against DPVO's converged
inverse depth. That per-keyframe ratio is applied to *incremental displacements*
before re-integrating, which corrects scale **drift**, not just a constant factor.

### Evaluation

`gpx_evaluation.py` aligns with a rotation and a translation only, then reports
ATE / RTE / FPE / KITTI plus a map overprint, diagnostic panels and a GPX export.

## Gotchas

**Coordinate handedness is the single biggest trap.** Everything works in **ENU**
(x east, y north). Alignment applies rotation and translation but **never a
reflection**, so ground truth in the opposite handedness cannot be aligned at all —
metrics come out several times too large instead of failing. Simulator ground truth
(TartanAir and friends) is **NED**; those files need `--gt-axes ned`. The research
repo had two cancelling mirrors here (`(north, east)` ground truth *and* a
`txyz[:, [2,1,3]]` reorder inside `rotate_given_g_3d`); both were removed together.

**Alignment must never fit scale.** No Umeyama, no Sim(3). Scale is the quantity
being measured; fitting it away would hide exactly the error of interest.
`tests/test_alignment.py::test_scale_error_survives_alignment` guards this.

**DPVO is patched and vendored.** Upstream DPVO will not work — see
`third_party/dpvo/PATCHES.md` for the modified files: five for the depth-guidance
behaviour, plus four more (`dispatch.h`, `lietorch_gpu.cu`, `lietorch_cpu.cpp`,
`correlation_kernel.cu`) for a `Tensor::type()` → `.scalar_type()` build fix needed
by torch >= 2.9. `dpvo.pth` is gitignored and fetched by `install.sh`.

**DepthPro must stay at arm's length.** Its Apple licence is not GPL-3.0 compatible
for redistribution, so it is never vendored, never a hard dependency, and imported
lazily only when `--depth-model depthpro` is used.

**Frames are BGR**, not RGB. DPVO's own `stream.py` feeds it `cv2.imread` output
without conversion, so BGR is correct — do not "fix" this.

**Frame indexing is 1:1.** Dataset index `i` is video frame `i * stride` is depth
frame `i * stride`, and depth corrections are indexed by the same. The research
pipeline's separate "depth creation stride" is gone; keep it that way.

**Tests only run from the repo root.** There is no `conftest.py` or packaging, so
`import src.*` resolves via the current working directory. `pytest` from elsewhere
fails collection on all eight files.

**DPVO can OOM on the last stage**, after depth has already been computed. Its
memory grows with frame area and keyframe count; `--resize 0.5` does not fit 1080p
on an 11 GB card, `--resize 0.3` does.

## Verifying changes

The research repo is the parity baseline. Its
`35_trajectory_quantitative_evaluation.py` is the reference implementation of the
metrics, and these numbers currently reproduce **exactly**:

| Sequence | ATE RMSE | RTE RMSE |
| --- | --- | --- |
| `ForestEnv_0000` (`--gt-axes ned`) | 48.581 m | 4.045 m |
| `20_Skatas_video_6` (GPX + map) | 20.362 m | — |

If a change to `alignment.py`, `metrics.py` or `gpx.py` moves those, it is a
regression until proven otherwise. Note that changes to the *pipeline* stages are
expected to move results slightly — the README's "Differences from the research
pipeline" section lists the deliberate behavioural deviations.
