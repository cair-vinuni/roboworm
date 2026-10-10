# Radial Attention in RoboWorM

Our integration code; the upstream package is vendored in
[`../radial-attention/`](../radial-attention/README.md)
([mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention), Apache-2.0).
We reuse its `radial_attn/attn_mask.py` (mask construction + FlashInfer block-sparse call).
Knob table and pitfalls: [docs/acceleration/sparse_attention.md](../../../docs/acceleration/sparse_attention.md#radial).

**Method.** Radial Attention (Li et al., *Radial Attention: O(n log n) Sparse Attention with
Energy Decay for Long Video Generation*, [arXiv 2506.19852](https://arxiv.org/abs/2506.19852))
uses a static block mask whose band width shrinks with temporal distance between frames
(mimicking spatio-temporal energy decay), executed with FlashInfer block-sparse kernels.

| file | role |
|---|---|
| `dreamgen_runtime.py` | `DreamGenRadialRuntime` (DiT path, shared by DreamGen and DreamDojo); mask plan cached in `_RadialPlan` |
| `processor.py` | `SVDRadialAttnProcessor` (Ctrl-World diffusers processor); `MaskMap` + wrapper cached per padded length |
| `adapter.py` | `enable_radial_attention` / `disable_radial_attention` (Ctrl-World) |
| `config.py` | `add_radial_args` (Ctrl-World), `add_dreamgen_radial_args`, `build_dreamdojo_radial_parameters` |

## How we apply it

| backbone | hook point | what is sparsified | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT) | `--use_radial` → per-block `config.net.radial_parameters` (`configs/base/config_radial.py::build_radial_parameters`) → `RadialLocalAttnOp` in `models/dreamgen/cosmos_predict2/module/a2a_cp.py` → `DreamGenRadialRuntime.forward` | DiT self-attention of blocks `≥ start_layer_idx` | **fp16 → bf16 cast fix**: FlashInfer can return fp16 for bf16 inputs, so the runtime casts back to `query.dtype` (keep it); CP raises |
| **DreamDojo** (2B DiT) | `--use-radial` → Hydra `model.config.net.self_attention_backend=radial` → `build_dreamdojo_radial_parameters` → `RadialLocalAttnOp` in `networks/a2a_cp.py` (loads the same `DreamGenRadialRuntime`) | same | same cast fix; single GPU |
| **Ctrl-World** (SVD U-Net) | `adapter.py::enable_radial_attention` assigns `SVDRadialAttnProcessor` to every attention module found by `unet.named_modules()` | `attn1` self-attention with T ≥ 128 after the first `first_layers_fp` layers | latent canvas passed as one view (`num_views=1`, `H_per_view=24·rows`, `W=40·cols`, from `_enable_backend`); small layers padded to the block size |

## Flags and shipped defaults

| knob | DreamGen | DreamDojo | Ctrl-World | env |
|---|---|---|---|---|
| enable | `--use_radial` | `--use-radial` | `--use_radial` | — |
| decay | `--radial_decay_factor` | `--radial-decay-factor` | `--radial_decay_factor` | `RADIAL_DECAY_FACTOR` (Ctrl-World `RADIAL_DECAY`) |
| first dense | `--radial_start_layer_idx` | `--radial-start-layer-idx` | `--radial_first_layers_fp` | `RADIAL_START_LAYER_IDX` / `RADIAL_FIRST_LAYERS_FP` |
| mask profile | `--radial_model_type` | `--radial-model-type` | `--radial_model_type` | `RADIAL_MODEL_TYPE` |

Shipped `decay / start / type` (block 64 everywhere; [configs.md](../../../docs/acceleration/configs.md)):

| backbone | cells |
|---|---|
| DreamGen | 1.0/4/hunyuan single_arm MV + SV, humanoid MV + SV; 0.5/8/wan bimanual MV; 0.2/1/hunyuan bimanual SV |
| DreamDojo | 0.5/2/hunyuan single_arm, bimanual; 1.0/2/hunyuan humanoid |
| Ctrl-World | 1.0/fp 4/hunyuan single_arm 3-view; 1.0/fp 2/hunyuan bimanual + humanoid 4-view |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_radial.sh` → `sampling_dataset/radial/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_radial.sh
```

## Known limitations

- Needs FlashInfer (DreamGen `.venv` 0.2.5, DreamDojo/Ctrl-World 0.3.1). On Ctrl-World, if the
  vendored `RadialAttention` import fails the processor silently runs dense SDPA.
- Block size must be 64 or 128 on the DiT path.
- ≈ 1.0× on DreamDojo (S = 4800 too short for the kernel); `decay ≤ 0.5` breaks DreamDojo
  humanoid quality; weakest quality of the sparse methods on DreamGen multiview.
