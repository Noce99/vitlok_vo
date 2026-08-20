# VitLok Mono RGB

Estimate a **metric-scale camera trajectory** from a single video — no GPS, no
IMU, no stereo rig. One camera, one calibration file, and one number: how high
the camera is above the ground.

```
video ──▶ undistort ──▶ metric depth ──▶ ground scale & shift ──▶ DPVO ──▶ trajectory
                                              ▲
                                      camera height (the only
                                      metric reference there is)
```

Then compare that trajectory against a GPS track:

```
trajectory + GPX + map ──▶ metrics · map overprint · diagnostics · GPX export
```

This is a cleaned-up extraction of the production path from the
[VitLok](https://git.chalmers.se/mannocci/vitlok) research project, which
investigates vision-based, GNSS-free localisation for pedestrians, bicycles and
trains in outdoor, infrastructure-poor environments.

---

## Why the camera height matters

A single moving camera cannot see scale. Every monocular method — DPVO included —
recovers the *shape* of a path but not its size: a small room and a large hall
produce identical images if you scale the motion to match. Monocular depth
networks have the same problem, returning depth that is correct only up to an
unknown scale and shift.

This pipeline resolves that with the one distance we actually know. The camera
sits a measured height above the ground; the ground is visible in the lower part
of nearly every frame; so fitting a plane to it and asserting that the plane is
exactly `--camera-height` metres below the camera pins down the depth map's scale
and shift. Metric depth then feeds DPVO, and the trajectory comes out in metres.

**So `--camera-height` is not a minor tuning knob — it is the measurement the
whole result rests on.** An error of 10 % in it is roughly a 10 % error in
distance travelled. Measure it; do not guess it.

---

## Install

Python **3.10** is required: DPVO's CUDA extensions build against torch 2.3.1,
which has no wheels for newer versions.

```bash
git clone <this repo> video_to_trajectory
cd video_to_trajectory
python3.10 -m venv venv
source venv/bin/activate

# set CUDA_TAG to match your driver: cu118, cu121, cu124, ...
CUDA_TAG=cu121 ./install.sh
```

`install.sh` installs torch, builds the vendored DPVO (three CUDA extensions,
a few minutes) and downloads its weights. Metric3D's weights download themselves
on first use. Verify with:

```bash
python -c "import dpvo, torch; print(torch.cuda.is_available())"
python video_to_trajectory.py --help
```

A GPU is needed for stages 2 and 4. `ffmpeg` is used for encoding the linearised
video; without it the pipeline falls back to OpenCV's encoder and ignores
`--undistort-crf`.

---

## 1. Calibrate the camera (once per camera and lens setting)

Film a **10×7-square chessboard** (9×6 inner corners) for 30–60 seconds, moving
it so it visits the frame's corners and is seen at a range of angles and
distances. Distortion is estimated from how straight lines bend near the edges,
so a board that never leaves the middle of the frame gives a poor fit.

```bash
python calibrate.py calibration_video/gopro_sw.mp4 --name gopro_SW
```

This writes `calibration/gopro_SW.txt`:

```
fx fy cx cy k1 k2 p1 p2 k3
```

and saves detection and before/after previews under `calibration_logs/`. Check
those previews — straight edges in the scene should come out straight. A
reprojection error above about 1 px means the board was not seen from enough
distinct viewpoints.

Calibrations for the project's GoPro modes ship in `calibration/`: `gopro_L`
(linear), `gopro_N` (normal), `gopro_W` (wide), `gopro_SW` (super-wide), plus
`tartan` for distortion-free synthetic footage.

## 2. Video → trajectory

```bash
python video_to_trajectory.py walk.mp4 \
    --calibration calibration/gopro_SW.txt \
    --camera-height 1.8
```

Or put the same settings in a file and use `--config`, which is easier to keep
around and to submit to a cluster:

```yaml
# run.yaml
video: /data/walk.mp4
calibration: calibration/gopro_SW.txt
camera_height: 1.8
depth_model: metric3d
resize: 0.5
```

```bash
python video_to_trajectory.py --config run.yaml --camera-height 1.75   # flags win
```

The result is:

```
output/walk/
├── trajectory.txt              time x y z metric_error  (metres, Z up)
├── metadata.json               how it was produced
└── undistortion_preview.jpg    original | undistorted — check this first
```

## 3. Evaluate against GPS

```bash
python gpx_evaluation.py output/walk/trajectory.txt \
    --gpx walk.gpx --start-time 2026-08-20T09:15:00Z \
    --map site.png --world site.pgw --epsg 3006
```

Writes into `output/walk/evaluation/`:

| File | What it shows |
| --- | --- |
| `metrics.json` + printed table | ATE, RTE, final-point error, KITTI drift |
| `map_overprint.png` | both tracks on the georeferenced map, with uncertainty discs |
| `diagnostics.png` | distance, heading and scale-drift panels |
| `estimate.gpx` | the estimate as a GPX track |

Without `--map` the overprint is drawn on plain axes in metres instead.

`--start-time` is when the video's **first frame** was recorded, in UTC, and is
what lines the GPX up with the video. Getting it wrong shifts the ground truth
along the track and inflates every metric.

For simulator ground truth in a file rather than a GPX:

```bash
python gpx_evaluation.py output/walk/trajectory.txt \
    --gt-trajectory scene.gt_trajectory --gt-axes ned
```

> **`--gt-axes` matters.** Alignment applies a rotation and a translation, never a
> reflection. TartanAir and most simulators export **NED** (x north, y east),
> which is the opposite handedness to the ENU (x east, y north) this tool works
> in. Pass the wrong one and the metrics come out several times too large instead
> of failing outright.

## 4. Run it on a cluster

```bash
python make_sbatch.py
```

It asks for an email address and a cluster (default `arrhenius`), then writes
`sbatches/<name>.sbatch`. Every prompt is also a flag, so it scripts too:

```bash
python make_sbatch.py --email you@example.org --cluster disi \
    --video /data/walk.mp4 --calibration calibration/gopro_SW.txt \
    --camera-height 1.8 --submit
```

The generated job puts its intermediates on **node-local scratch** (`$TMPDIR`)
and deletes them on exit — writing a multi-gigabyte gzip HDF5 to a shared
filesystem is far slower and unkind to everyone else on it. Add a cluster by
extending `CLUSTERS` in `src/sbatch.py`.

---

## The four stages

### 1 · Undistortion (`src/undistortion.py`)

Undistorts every frame with `cv2.undistort`, optionally resizes (`--resize`,
default `0.5`), and writes a linear video into the scratch directory. Everything
downstream then works with four intrinsics and no distortion model.

### 2 · Depth (`src/depth.py`)

Runs a metric depth network over the linear video and stores one depth map per
frame in a gzip HDF5, float32, in metres.

* `metric3d` (**default**) — Metric3D ViT-small via `torch.hub`; no checkpoint to
  manage, small enough to pack many runs onto one GPU.
* `depthpro` — Apple DepthPro; heavier, sharper, takes the focal length as input.
  Optional extra, see `requirements-depthpro.txt`.

### 3 · Ground scale & shift (`src/ground_scale_shift.py`)

Per frame: unproject the bottom `--bottom-fraction` of the depth map, segment
ground with **CSF** (Cloth Simulation Filter — a cloth is dropped onto the
inverted point cloud and the points it settles on are ground; no learned
segmentation is involved), average local surface normals for the plane
orientation, and solve for `(s, t)` in `d_metric = s·d_pred + t` by RANSAC and
Huber IRLS against the known camera height.

This runs every `--sas-stride` frames (default `20`) and interpolates in between:
it is the slowest CPU stage, and scale and shift drift slowly. Frames where the
fit fails record NaN and are filled from their neighbours, so a failure never
slides the rest of the series out of step with the video.

### 4 · DPVO (`src/dpvo_runner.py`)

Depth reaches DPVO twice. Inside the tracker, the vendored patch uses it to place
patches on nearby well-conditioned surfaces and to initialise their inverse
depths. Afterwards, `--scaling depth_ratio` (the default) compares metric depth
at each keyframe's patch centres against the inverse depth DPVO converged on; the
ratio is DPVO's scale error there, and applying it to the incremental
displacements before re-integrating corrects scale *drift*, not just a single
global scale factor. `--scaling none` leaves DPVO's own scale alone.

---

## Troubleshooting

**`torch.cuda.OutOfMemoryError` during the DPVO stage.** DPVO's memory grows with
frame area and with the number of keyframes it is holding, and it is the last
stage — so this can strike after the depth stage has already run. Lower
`--resize` (`0.3` comfortably fits 1080p footage on an 11 GB card where `0.5`
does not), or give the job a larger GPU. Re-running with `--keep-intermediates`
on the previous run's `--work-dir` is not yet supported: changing `--resize`
invalidates the linear video and the depth anyway.

**The trajectory drifts badly, or scale is obviously wrong.** Check
`metadata.json` → `ground_scale_shift`. A low `success_rate` means the ground was
rarely found: `--camera-height` may be wrong, the camera may not see enough
ground, or `--bottom-fraction` may be too small. A very high
`mean_condition_number` means scale and shift were weakly separable in that
scene, which happens when all the visible ground sits at a similar depth.

**Straight lines are bent in `undistortion_preview.jpg`.** Wrong calibration file
for the lens setting the video was shot at. The GoPro modes are genuinely
different cameras as far as calibration is concerned.

**Every metric is enormous but the map overprint looks roughly right.** Check
`--gt-axes`; see the note in step 3.

**`ffmpeg not found`.** The pipeline falls back to OpenCV's `mp4v` encoder and
ignores `--undistort-crf`. Installing ffmpeg is recommended — the fallback's
quality is not controllable.

---

## Output format

`trajectory.txt` is whitespace-separated with a one-line header, so
`np.loadtxt(path, skiprows=1)` reads it and so does a spreadsheet:

```
time x y z metric_error
0.000000 0.000000 0.000000 0.000000 0.004597
0.041708 -0.015569 -0.027884 -0.002754 0.006107
```

* `time` — seconds from the first frame
* `x y z` — metres in a local world frame, **Z up**, origin at the first pose
* `metric_error` — the tracker's own per-frame positional uncertainty, in metres,
  computed with no ground truth. `gpx_evaluation.py` draws its running sum as the
  uncertainty discs on the map.

`metadata.json` records the full configuration, per-stage timings, the depth
model, the ground-fit summary (median scale and shift, success rate, mean inlier
ratio) and this repo's git commit.

---

## Components and licences

This project is licensed under the **GNU General Public License v3.0**
(see `LICENSE`).

| Component | Licence | How it is used |
| --- | --- | --- |
| **video_to_trajectory** | GPL-3.0 | this repository |
| [DPVO](https://github.com/princeton-vl/DPVO) | MIT | vendored in `third_party/dpvo`, patched — see its `PATCHES.md` |
| [Metric3D](https://github.com/YvanYin/Metric3D) | BSD-2-Clause | fetched at runtime via `torch.hub` |
| [CSF](http://ramm.bnu.edu.cn/projects/CSF/) | Apache-2.0 | pip dependency |
| [Open3D](https://www.open3d.org/) | MIT | pip dependency |
| [DepthPro](https://github.com/apple/ml-depth-pro) | Apple ML research licence | **optional**, not vendored |

DepthPro is deliberately kept at arm's length: its licence is not GPL-3.0
compatible for redistribution, so it is neither shipped nor a hard dependency,
and is imported only when `--depth-model depthpro` is actually used.

---

## Differences from the research pipeline

For anyone coming from the original VitLok repository, the behaviour is not quite
identical:

* **Undistortion happens once, up front.** The original undistorted RGB frames on
  the fly but never the depth maps, so metric depth was sampled at *undistorted*
  pixel coordinates out of a *distorted* depth image. Predicting depth from
  already-linear frames removes the mismatch.
* **Depth is computed at the linear video's resolution**, rather than at full
  resolution and then downscaled on the way into storage.
* **Corrections are indexed by depth frame.** The original applied
  `correction[i]` while reading depth frame `i * depth_stride`, so any stride
  above 1 slid the corrections out of step with the depth they corrected.
* **Failed ground fits are interpolated over** instead of being dropped, which
  previously shortened the series and misaligned every frame after the failure.
* **One trajectory per run**, at a fixed path, instead of timestamped files inside
  one of seventeen `trajectories_dpvo_<variant>_<network>/` folders discovered by
  filename order.
* **No `data.yaml` registry.** A run is described by a video plus flags, or a
  small `--config` YAML.

Expect results close to, but not bit-identical with, the original.

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers the scale/shift estimator against synthetic ground with known `(s, t)` and
10 % gross outliers, the alignment (including that it does **not** absorb a scale
error), the metrics, calibration and trajectory I/O, the config layering, and the
sbatch generator.
>>>>>>> e591d32 (Initial commit: end-to-end video to metric trajectory)
