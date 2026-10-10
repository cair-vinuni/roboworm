# WorldCache in RoboWorM

**Method.** WorldCache (Nawaz et al., *WorldCache: Content-Aware Caching for Accelerated Video
World Models*, ECCV 2026; upstream [umair1221/WorldCache](https://github.com/umair1221/WorldCache))
uses DiCache's probe-and-extrapolate step skipping (see [../DiCache/README.md](../DiCache/README.md))
with a content-aware gate: the threshold is scaled by input motion, `thresh / (1 + α · Δx)`
(`α = motion_sensitivity`, `Δx` = relative L1 change of the denoiser input), so fast-moving
scenes are cached less. Optional guards: dynamic threshold decay, high-frequency drift,
saliency weighting, least-squares γ (OSI), flow-warped residuals, AdUC.

This directory is a **re-implementation** (no vendored upstream code). Guard table and pitfalls:
[docs/acceleration/caching.md](../../../docs/acceleration/caching.md#worldcache).

## How we apply it

| backbone | hook point | what is skipped | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT 2B/14B) | `runtime.py::apply_worldcache` replaces `dit.forward` with `worldcache_mini_train_dit_forward`; called from `Video2WorldPipeline._apply_cache_backend` (`models/dreamgen/cosmos_predict2/pipelines/video2world.py`) | all DiT blocks after the probe | sequential CFG → step budget `2 × num_steps`, one cache slot per branch; reset per chunk by `common.py::reset_cache_runtime`; all guards exposed on the CLI |
| **DreamDojo** (2B action-conditioned DiT) | same `apply_worldcache`, via `Video2WorldInference._apply_cache_backend` (`models/DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py`) | same | infer scripts expose only the 5 core knobs (thresh, ret, probe, motion, steps), no guards |
| **Ctrl-World** (SVD U-Net) | `adapter.py::enable_worldcache` attaches `ctrl_world_runtime.py::CtrlWorldWorldCacheRuntime` (subclass of `CtrlWorldDiCacheRuntime`) as `unet._ctrl_cache_runtime`; driven by `begin_step` / `observe_probe` / `finish_step` in `UNetSpatioTemporalConditionModel.forward` | remaining U-Net stages | CFG off; guards limited to HF drift, saliency, OSI, dynamic decay (no flow warp / AdUC / parallel CFG) |

Single GPU only; exclusive with every other backend.

## Flags and shipped defaults

| knob | DreamGen / DreamDojo | Ctrl-World | env |
|---|---|---|---|
| enable | `--worldcache-enabled` | `--use_worldcache` | — |
| threshold | `--worldcache-rel-l1-thresh` | `--worldcache_rel_l1_thresh` | `WORLDCACHE_REL_L1_THRESH` |
| warm-up | `--worldcache-ret-ratio` | `--worldcache_ret_ratio` | `WORLDCACHE_RET_RATIO` |
| probe | `--worldcache-probe-depth` | `--worldcache_probe_depth` | `WORLDCACHE_PROBE_DEPTH` |
| motion α | `--worldcache-motion-sensitivity` | `--worldcache_motion_sensitivity` | `WORLDCACHE_MOTION_SENSITIVITY` |
| decay guard | `--worldcache-dynamic-decay` (DreamGen) | `--worldcache_dynamic_decay` | — |

Shipped `thresh / ret / probe / motion` (full matrix in [configs.md](../../../docs/acceleration/configs.md)):

| backbone | cells |
|---|---|
| DreamGen | 0.20/0.2/2/5.0 bimanual MV, humanoid MV; 0.20/0.2/2/4.0 bimanual SV; 0.08/0.5/4/4.0 + decay single_arm MV + SV; 0.12/0.2/4/4.0 + decay humanoid SV |
| DreamDojo | 0.15/0.2/4/2.0 single_arm; 0.03/0.4/4/5.0 bimanual, humanoid |
| Ctrl-World | 0.08/0.30/2/2.5 single_arm 3-view; 0.04/0.0/1/2.5 bimanual 4-view; 0.03/0.0/1/2.5 humanoid 4-view |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_worldcache.sh`, output under
`sampling_dataset/worldcache/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_worldcache.sh
```

## Known limitations

- Same probe cost and `ret_ratio` cliff as DiCache (DreamGen humanoid SV collapses between
  `ret_ratio` 0.15 and 0.10).
- On the Ctrl-World 4-view grid only `ret_ratio = 0`, `probe_depth = 1` make it fast; the
  base-script defaults (`0.4 / 2–3`) were tuned on retired 3-view checkpoints.
- No context parallelism.
