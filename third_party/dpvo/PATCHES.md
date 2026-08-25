# Local modifications to DPVO

This is [DPVO](https://github.com/princeton-vl/DPVO) (Deep Patch Visual Odometry,
Princeton Vision & Learning Lab, MIT licence -- see `LICENSE`) with a small patch
that lets an external **metric depth map** guide the tracker.

Stock DPVO is monocular and scale-free: it picks patch centroids at random and
initialises their inverse depths at random. This pipeline already has a metric
depth map for every frame, so both of those choices can be informed by it.

## What changed

| File | Change |
|---|---|
| `dpvo/config.py` | Adds two config keys: `CENTROID_SEL_STRAT` (default `'RANDOM'`, i.e. stock behaviour) and `CENTROID_SEL_RANDOM_RATIO` (default `0.5`). |
| `config/default.yaml` | Sets `CENTROID_SEL_STRAT: 'DEPTH_HALF_RANDOM'` and `CENTROID_SEL_RANDOM_RATIO: 0.5`. |
| `dpvo/net.py` | `VONet.patchify(...)` gains a `depth_map=` argument plus `centroid_sel_strat` / `centroid_sel_random_ratio`. Under `DEPTH_HALF_RANDOM` a fraction of the patches is still sampled uniformly at random and the rest is rejection-sampled from pixels with `0 < d < 10 m`; under `DEPTH_CLOSE` all patches are. |
| `dpvo/dpvo.py` | `DPVO.__call__(self, tstamp, image, intrinsics, depth=None)` -- the new `depth` argument is forwarded to `patchify`, and is then used to initialise each patch's inverse depth as `1/d` (clamped, with the median substituted where depth is invalid) instead of the stock random/median initialisation. |
| `dpvo/nut_depth.py` | New file: `save_depth_visualization`, a debugging helper for dumping the depth maps DPVO actually sees. |

Passing `depth=None` reproduces stock behaviour, and `CENTROID_SEL_STRAT: 'RANDOM'`
(as set in `config/fast.yaml`) restores stock centroid selection.

### Build compatibility with torch >= 2.9

Unrelated to the depth-guidance patch above: three more files needed a mechanical
fix to build against modern torch. `Tensor::type()` -- returning the long-deprecated
`at::DeprecatedTypeProperties` -- lost its implicit conversion to `c10::ScalarType`,
which broke every `AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), ...)` /
`DISPATCH_GROUP_AND_FLOATING_TYPES(group, x.type(), ...)` call with a compile error
("no suitable conversion function ... exists"). Fixed by switching every such call
site to `x.scalar_type()` (the non-deprecated equivalent) and simplifying
`DISPATCH_GROUP_AND_FLOATING_TYPES` in `dispatch.h` to consume a `ScalarType`
directly instead of round-tripping through `::detail::scalar_type()`.

| File | Change |
|---|---|
| `dpvo/lietorch/include/dispatch.h` | `DISPATCH_GROUP_AND_FLOATING_TYPES` takes `TYPE` as an `at::ScalarType` directly. |
| `dpvo/lietorch/src/lietorch_gpu.cu` | Every `a.type()` / `X.type()` dispatch argument → `.scalar_type()`. |
| `dpvo/lietorch/src/lietorch_cpu.cpp` | Same. |
| `dpvo/altcorr/correlation_kernel.cu` | `AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.type()/net.type(), ...)` → `.scalar_type()`. |

`a.device().type()` calls (a different `.type()`, on `torch::Device`) are untouched
-- that API was never deprecated.

## What was removed

Parts of upstream DPVO that this pipeline never reaches were dropped to keep the
vendored tree small:

- `DPRetrieval/`, `Pangolin/`, `DBoW2/` -- the image-retrieval backend and the
  Pangolin viewer. `dpvo.py` imports the viewer lazily (only under `viz=True`) and
  `loop_closure/retrieval` imports DPRetrieval lazily, so neither is reachable here.
- `dpvo/data_readers/` -- training-time dataset loaders; nothing in this pipeline
  references them.

`dpvo/loop_closure/` is kept: `patchgraph.py` imports `reduce_edges` from it at
module level, so it is not optional even with loop closure disabled.

## Rebasing onto a newer upstream

The patch is confined to the files in the two tables above and is small enough to
re-apply by hand. `git diff` against a fresh upstream checkout of the same commit
will show it in full.
