# SiTo in RoboWorM (integration notes)

`README.md` in this directory is the **upstream** SiTo README
([EvelynZhang-epiclab/SiTo](https://github.com/EvelynZhang-epiclab/SiTo)); `sito.py`, `patch.py`
and `utils.py` are the upstream (ToMeSD-style) code. Our integration is `runtime.py`
(DiTs), `spatiotemporal_hold.py`, `adapter.py` + `processor.py` (Ctrl-World) and `config.py`.
Details and pitfalls: [docs/acceleration/pruning.md](../../../docs/acceleration/pruning.md).

**Method.** SiTo (Zhang et al., *Training-free and Hardware-friendly Acceleration for Diffusion
Models via Similarity-based Token Pruning*, AAAI 2025) picks base tokens per image patch (highest
noisy similarity score), prunes the tokens most similar to them before a layer, and recovers the
pruned tokens afterwards by copying their most similar base token.

## How we apply it

| backbone | hook point | what is pruned | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT) | `--use_sito` → per-block `config.net.sito_parameters` (`runtime.py::build_sito_parameters`, via `configs/base/config_sito.py`) → `SiToTokenPruner` held by `Block` in `models/dreamgen/cosmos_predict2/models/text2image_dit.py`; `Block.forward` calls `prepare` / `prune` / `prune_rope` before self-attention and `recover` after | self-attention tokens of blocks `[start, N − keep_last_n_dense)` | recover is one `index_select`; RoPE pruned together with tokens; `temporal` grouping (prune tokens most similar to the previous frame) added on top of upstream `per_frame` / `full_2d`; `prune_ratio=None` → auto ratio |
| **DreamDojo** (2B DiT) | `--use-sito` in `models/DreamDojo/examples/action_conditioned.py` → `model.config.net.use_sito=true` (`cosmos_predict2/action_conditioned.py`) → same `SiToTokenPruner` in `Block` of `networks/minimal_v4_dit.py` | same | **different entrypoint from dense** (the dense infer scripts have no SiTo flags); disabled when a KV cache is active; runs at guidance 0 like dense |
| **Ctrl-World** (SVD U-Net) | `--use_sito --sito_spatiotemporal_hold` → `spatiotemporal_hold.py::enable_spatiotemporal_hold` monkeypatches every `TransformerSpatioTemporalModel.forward` | spatial tokens pruned **once** after `proj_in`, held through both the spatial and the temporal block, restored before `proj_out` | **prune-once-hold** (per-attention pruning cannot pay off: spatial attention ≈ 7 % of wall); keep = top-`keep_ratio` by feature L2 norm; restore = most cosine-similar kept token; plan computed once (`recompute_every` 999999); `max_downsample_ratio` 1 → largest stage only, 4 → all three |

The per-attention Ctrl-World path (`adapter.py::enable_sito` → `SVDSiToAttnProcessor`) is
implemented but not shipped.

## Flags and shipped defaults

| knob | DreamGen | Ctrl-World | env |
|---|---|---|---|
| enable | `--use_sito` | `--use_sito --sito_spatiotemporal_hold` | — |
| amount | `--sito_prune_ratio` | `--sito_st_keep_ratio` | `SITO_PRUNE_RATIO` / `SITO_ST_KEEP_RATIO` |
| layers | `--sito_start_layer_idx`, `--sito_keep_last_n_dense` | `--sito_max_downsample_ratio` | `SITO_START_LAYER_IDX`, `SITO_KEEP_LAST_N_DENSE` / `SITO_MAX_DOWNSAMPLE_RATIO` |
| grouping / noise | `--sito_group_mode`, `--sito_noise_alpha` | — | `SITO_GROUP_MODE`, `SITO_NOISE_ALPHA` |

DreamDojo uses the hyphenated `--use-sito` / `--sito-*` flags of `action_conditioned.py`.

| backbone | shipped ([configs.md](../../../docs/acceleration/configs.md)) |
|---|---|
| DreamGen | prune 0.20 (MV) / 0.30 (SV), start 6, last_dense 2, noise 0, group `temporal` |
| DreamDojo | start 4, last_dense 2, auto ratio (single_arm, humanoid; no bimanual launcher) |
| Ctrl-World | keep 0.9 / down 1 single_arm 3-view; keep 0.8 / down 4 bimanual + humanoid 4-view |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_sito.sh` → `sampling_dataset/sito/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_single_arm_singleview_dreamgen_sito.sh
```

## Known limitations

- `--sito_group_mode` is only forwarded by `examples/video2world_gr00t.py` (DreamGen single_arm
  MV); the other five DreamGen cells run `per_frame` despite `SITO_GROUP_MODE=temporal`.
- Ctrl-World: pruning before the temporal block hurts dynamics; `keep < 0.7` corrupts. On
  Ctrl-World SiTo and ITM run the **same** ST-hold algorithm (see
  [../importance_token_merge/README.md](../importance_token_merge/README.md)).
- DreamDojo SiTo has no released rollouts. Single GPU only.
