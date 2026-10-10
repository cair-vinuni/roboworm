# Acceleration methods

RoboWorM benchmarks 9 training-free acceleration methods against a `dense` baseline on three
backbones: **DreamGen** (Cosmos-Predict2 DiT, 2B / 14B), **DreamDojo** (2B action-conditioned
DiT) and **Ctrl-World** (SVD U-Net). Each method has one integration doc that says where it hooks
into each backbone, what we changed versus upstream, the flags, the shipped defaults and the
known limitations. Cross-method docs: [docs/acceleration/](../docs/acceleration/README.md)
(algorithms, knob tables, and the full cell × method config matrix in
[configs.md](../docs/acceleration/configs.md)).

| method | family | paper / upstream | our code | DreamGen | DreamDojo | Ctrl-World | doc |
|---|---|---|---|---|---|---|---|
| WorldCache | caching | ECCV 2026 / [umair1221/WorldCache](https://github.com/umair1221/WorldCache) | `cache_strategy/WorldCache/` (re-implemented) | yes | yes | yes | [README](cache_strategy/WorldCache/README.md) |
| DiCache | caching | [arXiv 2508.17356](https://arxiv.org/abs/2508.17356) / [Bujiazi/DiCache](https://github.com/Bujiazi/DiCache) | `cache_strategy/DiCache/` (re-implemented) | yes | yes | yes | [README](cache_strategy/DiCache/README.md) |
| FasterCache | caching | ICLR 2025 / [Vchitect/FasterCache](https://github.com/Vchitect/FasterCache) | `cache_strategy/FasterCache/{runtime,adapter,processor,config}.py` (upstream vendored alongside) | yes | yes | no-op (CFG off) | [INTEGRATION](cache_strategy/FasterCache/INTEGRATION.md) |
| PISA | sparse attention | [arXiv 2602.01077](https://arxiv.org/abs/2602.01077) / upstream in `sparse_attention/piecewise-sparse-attention/` | `sparse_attention/pisa/` | yes | yes | yes | [INTEGRATION](sparse_attention/pisa/INTEGRATION.md) |
| Radial | sparse attention | [arXiv 2506.19852](https://arxiv.org/abs/2506.19852) / [mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention) | `sparse_attention/radial/` | yes | yes | yes | [README](sparse_attention/radial/README.md) |
| SVG1 | sparse attention | [arXiv 2502.01776](https://arxiv.org/abs/2502.01776) / [svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) | `sparse_attention/svg/` | yes | yes | yes | [INTEGRATION](sparse_attention/svg/INTEGRATION.md) |
| SVG2 | sparse attention | [arXiv 2505.18875](https://arxiv.org/abs/2505.18875) / same upstream | `sparse_attention/svg/` | yes (`.venv_svg`) | yes | yes | [INTEGRATION](sparse_attention/svg/INTEGRATION.md) |
| SiTo | token reduction | AAAI 2025 / [EvelynZhang-epiclab/SiTo](https://github.com/EvelynZhang-epiclab/SiTo) | `prunning/SiTo/{runtime,spatiotemporal_hold,adapter,processor,config}.py` | yes | `action_conditioned` only | ST-hold | [INTEGRATION](prunning/SiTo/INTEGRATION.md) |
| ITM | token reduction | [arXiv 2411.16720](https://arxiv.org/abs/2411.16720) / upstream in `prunning/importance-token-merge/` | `prunning/importance_token_merge/` | yes | `action_conditioned` only | ST-hold (= SiTo ST-hold) | [README](prunning/importance_token_merge/README.md) |

## Hook point per backbone (one phrase each)

| method | DreamGen | DreamDojo | Ctrl-World |
|---|---|---|---|
| WorldCache | `apply_worldcache` replaces `dit.forward` | same, via `Video2WorldInference._apply_cache_backend` | `CtrlWorldWorldCacheRuntime` on `unet._ctrl_cache_runtime`, driven from the U-Net forward |
| DiCache | `apply_dicache` replaces `dit.forward` | same | `CtrlWorldDiCacheRuntime` on `unet._ctrl_cache_runtime` |
| FasterCache | `apply_fastercache` wraps `dit.forward` (CFG) + each `block.self_attn.forward` | same | `SVDFasterCacheAttnProcessor` via `set_attn_processor` |
| PISA | `PISALocalAttnOp` per DiT block (`config.net.pisa_parameters`) | `self_attention_backend=pisa` → `PISALocalAttnOp` | `SVDPISAAttnProcessor` on long `attn1` layers |
| Radial | `RadialLocalAttnOp` → `DreamGenRadialRuntime` | `self_attention_backend=radial` → same runtime | `SVDRadialAttnProcessor` on long `attn1` layers |
| SVG1 / SVG2 | `SVGLocalAttnOp` → `DreamGenSVGRuntime` | `self_attention_backend=svg` → `DreamDojoSVGRuntime` | `SVDSVGAttnProcessor` / `SVDSAPAttnProcessor` on long `attn1` layers |
| SiTo | `SiToTokenPruner` around self-attn in each DiT `Block` | same pruner, `action_conditioned.py` entrypoint | prune-once-hold over `TransformerSpatioTemporalModel.forward` |
| ITM | `ITMTokenMerger` in each DiT `Block`, CFG heat map | same, guidance forced to 1.5 | same prune-once-hold as SiTo (feature-L2 keep) |

All backends are single-GPU only and mutually exclusive. Launchers:
`scripts/run/<family>/run_<embodiment>_<view>_<model>[_4view_grid]_<method>.sh`, writing to
`sampling_dataset/<method>/...`.

## Other contents

- `cache_strategy/{common,config,stats,ctrl_world_utils}.py`: shared cache configs, DreamGen
  cache CLI + `build_cache_runtime_config`, block-work accounting, Ctrl-World exclusivity check.
  `config.py` also registers ScalingCache / BWCache / BSC (DreamGen bimanual MV only; not among
  the 9 audited methods).
- Vendored upstream trees (each with an `INTEGRATION.md` pointer):
  `sparse_attention/{Sparse-VideoGen,piecewise-sparse-attention,radial-attention}/`,
  `prunning/importance-token-merge/`, and the upstream parts of `cache_strategy/FasterCache/`
  and `prunning/SiTo/`.
- [`human_assessment/`](human_assessment/README.md): keypoint / optical-flow tools for human
  evaluation, not an acceleration method.
