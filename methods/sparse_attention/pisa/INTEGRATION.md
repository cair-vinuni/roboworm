# PISA in RoboWorM (integration notes)

`README.md` here is the older Ctrl-World adapter quick-start; this file covers all three
backbones. Upstream source is vendored in
[`../piecewise-sparse-attention/`](../piecewise-sparse-attention/README.md) (MIT); the Triton
kernel we actually call is the copy `piecewise_sparse_attn_tma.py::piecewise_sparse_attention`
in this directory, so the editable `piecewise_attn` package is **not** required.
Knob table and pitfalls: [docs/acceleration/sparse_attention.md](../../../docs/acceleration/sparse_attention.md#pisa).

**Method.** PISA (Li et al., *PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion
Transformers*, [arXiv 2602.01077](https://arxiv.org/abs/2602.01077)) is "exact-or-approximate"
rather than "keep-or-drop": per query block, a `density` fraction of critical key blocks is
computed exactly and the remaining blocks are approximated with a block-wise Taylor expansion,
so the full attention span is still covered.

## How we apply it

| backbone | hook point | what is sparsified | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT) | `--use_pisa` → `get_cosmos_predict2_video2world_pipeline()` sets per-block `config.net.pisa_parameters` (`models/dreamgen/cosmos_predict2/configs/base/config_pisa.py::build_pisa_parameters`; `None` below `start_layer_idx`) → `PISALocalAttnOp` in `models/dreamgen/cosmos_predict2/module/a2a_cp.py` | DiT self-attention of blocks `≥ start_layer_idx` | kernel loaded by file path (`_load_piecewise_sparse_attention`); no context parallelism |
| **DreamDojo** (2B DiT) | `--use-pisa` in the infer scripts → Hydra overrides `model.config.net.self_attention_backend=pisa` + knobs → `config.py::build_dreamdojo_pisa_parameters` (called in `networks/minimal_v4_dit.py`) → `PISALocalAttnOp` in `networks/a2a_cp.py` | same | same kernel; CP raises |
| **Ctrl-World** (SVD U-Net) | `adapter.py::enable_pisa` assigns `processor.py::SVDPISAAttnProcessor` to every module with `processor` + `to_q` (`unet.named_modules()`, not `set_attn_processor`); called from `_enable_backend` in `scripts/infer/infer_*_ctrlworld.py` | `attn1` self-attention with T ≥ 128 (`MIN_TOKENS_FOR_PISA`) at attention-layer index `≥ start_layer_idx` | cross-attention, temporal (T = 11), short stages and masked calls fall back to SDPA |

## Flags and shipped defaults

| knob | DreamGen / Ctrl-World | DreamDojo | env |
|---|---|---|---|
| enable | `--use_pisa` | `--use-pisa` | — |
| density | `--pisa_density` | `--pisa-density` | `PISA_DENSITY` |
| block size | `--pisa_block_size` | `--pisa-block-size` | `PISA_BLOCK_SIZE` |
| start layer | `--pisa_start_layer_idx` | `--pisa-start-layer-idx` | `PISA_START_LAYER_IDX` (Ctrl-World: `PISA_START_LAYER`) |

Shipped `density / block / start` ([configs.md](../../../docs/acceleration/configs.md)):

| backbone | cells |
|---|---|
| DreamGen | 0.2/64/2 everywhere except bimanual SV 0.1/64/0 |
| DreamDojo | 0.2/64/2 on all three SV cells |
| Ctrl-World | 0.5/64/2 single_arm 3-view; 0.2/64/**0** bimanual + humanoid 4-view |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_pisa.sh` → `sampling_dataset/pisa/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_pisa.sh
```

## Known limitations

- Bounded by the self-attention share: ≈ 1.3–1.5× on DreamGen 2B/14B, ≈ 1.0× on DreamDojo
  (self-attn ≈ 15–18 %, S = 4800), ≤ 1× on Ctrl-World.
- `density ≤ 0.1` melts DreamDojo outputs; speed saturates near 1.1× there.
- `adapter_dreamgen.py` (`PISAAttentionWrapper`, `enable_pisa_dreamgen`) is an older DreamGen
  wrapper that no entrypoint imports; DreamGen uses the `PISALocalAttnOp` path above.
