# Sparse VideoGen (SVG1, SVG2) in RoboWorM (integration notes)

`README.md` here is the older Ctrl-World adapter write-up; this file covers both variants on all
three backbones. Upstream code is vendored in
[`../Sparse-VideoGen/`](../Sparse-VideoGen/README.md)
([svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen), Apache-2.0); we
import its `svg.kmeans_utils`, `svg.kernels.triton.permute`, `svg.flashinfer_patch` and (SVG1 on
DiTs) `svg.models.cosmos.placement`. Knob tables and pitfalls:
[docs/acceleration/sparse_attention.md](../../../docs/acceleration/sparse_attention.md#svg1-sparse-videogen).

**Methods.**
- **SVG1** (Xi et al., *Sparse VideoGen*, [arXiv 2502.01776](https://arxiv.org/abs/2502.01776)):
  each head is routed to a spatial or a temporal sparse mask, chosen by online MSE profiling of
  both masks against dense attention on a few sampled query rows.
- **SVG2** (Yang et al., *Sparse VideoGen2*, [arXiv 2505.18875](https://arxiv.org/abs/2505.18875)):
  semantic-aware permutation — k-means clusters of queries/keys are made contiguous and only
  the key-centroid blocks covering the `top_p` attention mass are computed (FlashInfer
  variable block-sparse).

## How we apply it

| backbone | hook point | adaptations |
|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT) | `--use_svg --svg_variant {svg1,svg2}` → per-block `config.net.svg_parameters` (`configs/base/config_svg.py::build_svg_parameters`, `None` below `start_layer_idx`) → `SVGLocalAttnOp` in `models/dreamgen/cosmos_predict2/module/a2a_cp.py` → `dreamgen_runtime.py::DreamGenSVGRuntime` (dispatches to `DreamGenSVG1Runtime` / `DreamGenSVG2Runtime`) | dense warm-up for the first `dense_step_frac` of steps; SVG1 runs the selected mask as compiled `flex_attention`; SVG1 re-profiles **every call** (`profile_recompute_every=1`, not on the CLI); **SVG2 needs `models/dreamgen/.venv_svg`** (FlashInfer 0.3.1) |
| **DreamDojo** (2B DiT) | `--use-svg --svg-variant …` → Hydra `self_attention_backend=svg` → `config.py::build_dreamdojo_svg_parameters` (unset knobs from `_dreamdojo_svg_defaults`) → `SVGLocalAttnOp` in `networks/a2a_cp.py` → `dreamdojo_runtime.py::DreamDojoSVGRuntime` (subclass of the DreamGen runtime) | **profile caching**: `--svg-profile-recompute-every` (launchers 999999) holds the head assignment, removing ≈ 47 % of a dense call of overhead |
| **Ctrl-World** (SVD U-Net) | `--use_svg --svg_pattern {SVG1,SVG2}` → `adapter.py::enable_svg` assigns `processor_svg1.py::SVDSVGAttnProcessor` or `processor_svg2.py::SVDSAPAttnProcessor` to every attention module from `unet.named_modules()` | only `attn1` self-attention with T ≥ 128 after `first_layers_fp`; SVG1 uses U-Net-specific masks (`mask_utils.py`: diagonal band + view stripe) run as **masked SDPA**, with the profiled mask choice held per (layer, T); SVG2 re-runs k-means every call (labels change with the latents); canvas passed as one view |

## Flags and shipped defaults

| knob | DreamGen / Ctrl-World | DreamDojo | env |
|---|---|---|---|
| SVG1 sparsity (window width) | `--svg_sparsity` | `--svg-sparsity` | `SVG_SPARSITY` |
| SVG1 profiled rows | `--svg_num_sampled_rows` | `--svg-num-sampled-rows` | `SVG_NUM_SAMPLED_ROWS` |
| SVG2 centroids | `--svg_num_{q,k}_centroids` | `--svg-num-{q,k}-centroids` | `SVG_NUM_Q_CENTROIDS`, `SVG_NUM_K_CENTROIDS` |
| SVG2 top-p (speed lever) | `--svg_top_p_kmeans` | `--svg-top-p-kmeans` | `SVG_TOP_P_KMEANS` |
| first sparse layer | `--svg_start_layer_idx` (DiT) / `--svg_first_layers_fp` (CW) | `--svg-start-layer-idx` | `SVG_START_LAYER_IDX` / `SVG_FIRST_LAYERS_FP` |

Shipped values ([configs.md](../../../docs/acceleration/configs.md)):

| backbone | SVG1 | SVG2 (all q 32 / k 64) |
|---|---|---|
| DreamGen | sparsity 0.25, start 4 (single_arm SV 0.15) | top_p 0.7, min_kc 0.15 (bimanual MV 0.6; bimanual SV 0.9 / 0.10), start 1 |
| DreamDojo | sparsity 0.25, start 2 (humanoid 4), rows 16, profile_every 999999 | top_p 0.6, min_kc 0.15, start 4 |
| Ctrl-World | sparsity 0.25, rows 32, fp 2 | top_p 0.7, fp 2 |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_{svg1,svg2}.sh` →
`sampling_dataset/{svg1,svg2}/...`. DreamGen `*_svg2.sh` default `VENV` to `.venv_svg`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_svg2.sh
```

## Known limitations

- DreamGen SVG2 in `.venv` (FlashInfer 0.2.5) falls back to a dense-mask SDPA that OOMs (≈ 57 GiB)
  on multiview.
- On DreamDojo (S = 4800) both lose to dense flash: SVG1 ≈ 0.93×, SVG2's block-sparse kernel alone
  is 1.76× slower than dense (`scripts/tools/diag/_diag_dreamdojo_svg.py`).
- Ctrl-World SVG1 masked SDPA leaves the flash kernel (≈ 4× slower than dense at T = 3840).
- Fewer than q 32 / k 64 centroids corrupts outputs. No context parallelism.
