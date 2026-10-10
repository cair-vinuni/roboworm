# DiCache in RoboWorM

**Method.** DiCache (Bu et al., *DiCache: Let Diffusion Model Determine Its Own Cache*,
[arXiv 2508.17356](https://arxiv.org/abs/2508.17356); upstream
[Bujiazi/DiCache](https://github.com/Bujiazi/DiCache)) skips whole denoiser forwards. After a
warm-up of `⌊ret_ratio · steps⌋` steps, each step first runs a shallow *probe* (the first
`probe_depth` blocks). If the accumulated relative L1 change of the probe output stays below
`rel_l1_thresh`, the rest of the network is skipped and the output is extrapolated from the
cached residuals, `x + r[-2] + γ (r[-1] − r[-2])`, with γ ∈ [1, 2] taken from the probe
residuals. Otherwise the forward is finished, reusing the probe output.

This directory is a **re-implementation** (no vendored upstream code). Algorithm details,
flag table and pitfalls: [docs/acceleration/caching.md](../../../docs/acceleration/caching.md#dicache).

## How we apply it

| backbone | hook point | what is skipped | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT 2B/14B) | `runtime.py::apply_dicache` replaces `dit.forward` with `dicache_mini_train_dit_forward`; called from `Video2WorldPipeline._apply_cache_backend` (`models/dreamgen/cosmos_predict2/pipelines/video2world.py`) when `cache_config` is a `DiCacheConfig` | all DiT blocks after the probe | CFG runs sequentially (cond, uncond), so the step budget is `2 × num_steps` and one cache slot per branch (`cnt % 2`); state reset per chunk by `common.py::reset_cache_runtime` |
| **DreamDojo** (2B action-conditioned DiT) | same `apply_dicache`, called from `Video2WorldInference._apply_cache_backend` (`models/DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py`); the infer scripts build `DiCacheConfig` from `--dicache-*` | same | same CFG/step accounting; effective step count from `_get_effective_num_steps` |
| **Ctrl-World** (SVD U-Net) | `adapter.py::enable_dicache` attaches `ctrl_world_runtime.py::CtrlWorldDiCacheRuntime` as `unet._ctrl_cache_runtime`; `UNetSpatioTemporalConditionModel.forward` (`models/Ctrl-World/models/unet_spatio_temporal_condition.py`) calls `begin_step` / `observe_probe` / `finish_step` | remaining down/mid/up stages | probe = output of the `probe_depth`-th attention-bearing stage; per-step (not accumulated) decision; CFG is off (one U-Net call per step) |

Single GPU only on every backbone; exclusive with every other backend
(`methods/cache_strategy/config.py::build_cache_runtime_config`,
`methods/cache_strategy/ctrl_world_utils.py::validate_ctrl_world_backend_args`).

## Flags and shipped defaults

| knob | DreamGen / DreamDojo | Ctrl-World | env |
|---|---|---|---|
| enable | `--dicache-enabled` | `--use_dicache` | — |
| threshold | `--dicache-rel-l1-thresh` | `--dicache_rel_l1_thresh` | `DICACHE_REL_L1_THRESH` |
| warm-up | `--dicache-ret-ratio` | `--dicache_ret_ratio` | `DICACHE_RET_RATIO` |
| probe | `--dicache-probe-depth` | `--dicache_probe_depth` | `DICACHE_PROBE_DEPTH` |
| steps | `--dicache-num-steps` (must equal sampler steps, 35) | from `--num_inference_steps` | `DICACHE_NUM_STEPS` |

Shipped `thresh / ret / probe` (full matrix in [configs.md](../../../docs/acceleration/configs.md)):

| backbone | cells |
|---|---|
| DreamGen | 0.20/0.2/2 bimanual MV, humanoid MV, bimanual SV; 0.15/0.2/2 humanoid SV; 0.08/0.2/2 single_arm MV + SV |
| DreamDojo | 0.08/0.2/2 on all three SV cells |
| Ctrl-World | 0.06/0.3/3 single_arm 3-view; 0.03/0.0/1 bimanual + humanoid 4-view grid |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_dicache.sh`, output under
`sampling_dataset/dicache/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_dicache.sh
```

## Known limitations

- The probe is paid on every gated step; a non-skipped gated step costs more than dense, so
  low thresholds can be net-negative (DreamGen humanoid MV `0.10` ≈ 0.8×).
- Lowering `ret_ratio` without tightening the threshold is where quality collapses.
- No context parallelism; launchers force one GPU even where dense uses 2–4.
