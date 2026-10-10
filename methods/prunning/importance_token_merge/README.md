# Importance-Based Token Merging (ITM) in RoboWorM

Our integration code; the upstream ToMeSD-based release is vendored in
[`../importance-token-merge/`](../importance-token-merge/README.md) (MIT) and is not imported at
runtime. Details and pitfalls: [docs/acceleration/pruning.md](../../../docs/acceleration/pruning.md).

**Method.** ITM (Wu et al., *Importance-Based Token Merging for Efficient Image and Video
Generation*, [arXiv 2411.16720](https://arxiv.org/abs/2411.16720)) builds an importance map from
classifier-free guidance, `|cond − uncond|`, keeps the most important tokens as merge
destinations and merges (ToMe-style) the unimportant ones into them before a layer, then unmerges
afterwards.

| file | role |
|---|---|
| `runtime.py` | `ITMTokenMerger`, `build_itm_parameters`, `compute_itm_heat_map`, `build_itm_attention_context` (DiTs) |
| `adapter.py` + `processor.py` | per-attention Ctrl-World path (`enable_itm` → `SVDITMAttnProcessor`) |
| `block_hold.py` | Ctrl-World block-hold (`enable_itm_block_hold`, feature-L2 importance) |
| `config.py` | `add_itm_args` (Ctrl-World), `add_dreamgen_itm_args` |

## How we apply it

| backbone | hook point | what is merged | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT) | `--use_itm` → per-block `config.net.itm_parameters` (`configs/base/config_itm.py`) → `ITMTokenMerger` held by `Block` in `models/dreamgen/cosmos_predict2/models/text2image_dit.py` (`prepare` / `merge` / `merge_rope` / `unmerge`); heat map from `compute_itm_heat_map(cond_x0, uncond_x0)` in `Video2WorldPipeline` (`pipelines/video2world.py`), passed via `build_itm_attention_context` | self-attention tokens of blocks `[start, N − keep_last_n_dense)` (optionally cross-attn / MLP) | uses the guidance-7 CFG pair that dense already computes; phase schedule dense → prune-only (`prune_from_step`) → merge (`merge_from_step`); `group_mode` fixed to `per_frame` |
| **DreamDojo** (2B DiT) | `--use-itm` in `models/DreamDojo/examples/action_conditioned.py` → `model.config.net.use_itm=true` → `ITMTokenMerger` in `Block` of `networks/minimal_v4_dit.py`; heat map in `action_conditioned_video2world_rectified_flow_model.py` | same | **different entrypoint from dense**; ITM needs `guidance > 0`, so `action_conditioned.py` raises guidance 0 → 1.5 (`FORCE_GUIDANCE` overrides) while dense runs at 0 |
| **Ctrl-World** (SVD U-Net) | `--use_itm --itm_spatiotemporal_hold` → `methods/prunning/SiTo/spatiotemporal_hold.py::enable_spatiotemporal_hold(similarity_recover=True)` | spatial tokens pruned once after `proj_in`, held through spatial + temporal blocks | CFG is off (guidance 1), so no heat map exists: keep = top-`keep_ratio` by feature L2 norm, restore = most similar kept token. **Same algorithm as Ctrl-World SiTo**, only the shipped keep ratio differs |

Implemented, not shipped on Ctrl-World: per-attention ITM (`enable_itm`, **forces
`guidance_scale` 1 → 2**, doubling the U-Net batch) and block-hold (`--itm_block_hold`,
`self_importance=True` so it runs at CFG = 1).

## Flags and shipped defaults

| knob | DreamGen | Ctrl-World | env |
|---|---|---|---|
| enable | `--use_itm` | `--use_itm --itm_spatiotemporal_hold` | — |
| amount | `--itm_compress_ratio` (< 0.75) | `--itm_st_keep_ratio` | `ITM_COMPRESS_RATIO` / `ITM_ST_KEEP_RATIO` |
| layers | `--itm_start_layer_idx`, `--itm_keep_last_n_dense` | `--itm_max_downsample_ratio` | `ITM_START_LAYER_IDX`, `ITM_KEEP_LAST_N_DENSE` / `ITM_MAX_DOWNSAMPLE_RATIO` |
| schedule | `--itm_prune_from_step`, `--itm_merge_from_step` | — | `ITM_PRUNE_FROM_STEP`, `ITM_MERGE_FROM_STEP` |

| backbone | shipped ([configs.md](../../../docs/acceleration/configs.md)) |
|---|---|
| DreamGen | compress 0.2, last_dense 4, start 11 (single_arm MV 14B: 14); schedule 1/2 or 0/0 per cell |
| DreamDojo | start 4, last_dense 2, schedule 1/2 (single_arm, humanoid; no bimanual launcher) |
| Ctrl-World | keep 0.8 / down 4 single_arm 3-view; keep 0.85 / down 4 bimanual + humanoid 4-view |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_itm.sh` → `sampling_dataset/itm/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_itm.sh
```

## Known limitations

- Net-negative on the DreamGen 2B DiTs (per-block prepare/merge cost exceeds the saved attention).
- On Ctrl-World it is not ITM's importance criterion (no CFG); compare it to SiTo as the same
  ST-hold with a different keep ratio.
- DreamDojo ITM runs at a different guidance than dense and has no released rollouts. Single GPU only.
