# FasterCache in RoboWorM (integration notes)

`README.md` in this directory is the **upstream** FasterCache README
([Vchitect/FasterCache](https://github.com/Vchitect/FasterCache)); the upstream package
(`fastercache/`, `configs/`, `scripts/`) is vendored and is **not** used at runtime.
Our integration is `runtime.py` (DiTs), `adapter.py` + `processor.py` (Ctrl-World) and `config.py`
(Ctrl-World CLI). Details: [docs/acceleration/caching.md](../../../docs/acceleration/caching.md#fastercache).

**Method.** FasterCache (Lv et al., *FasterCache: Training-Free Video Diffusion Model
Acceleration with High Quality*, ICLR 2025) has two reuse mechanisms, both active from
`start_step` (`0` → `⌈0.3 · steps⌉`):
1. **CFG level**: on steps that are not multiples of `model_interval`, the unconditional forward
   is skipped and rebuilt from the conditional output plus cached low/high-frequency (FFT)
   deltas measured on the last anchor step.
2. **Block level**: on steps that are not multiples of `block_interval`, an eligible block's
   self-attention output is extrapolated, `y[-1] + 0.3 (y[-1] − y[-2])`.

## How we apply it

| backbone | hook point | what is reused | adaptations |
|---|---|---|---|
| **DreamGen** (Cosmos-Predict2 DiT) | `runtime.py::apply_fastercache(dit, cfg, cfg_mode="sequential")` wraps `dit.forward` (CFG shortcut) and each `block.self_attn.forward` (`_patch_dit_block_self_attention`); called from `Video2WorldPipeline._apply_cache_backend` | uncond forward + self-attn outputs | sequential CFG mode; eligible blocks `[4, N−2)` for 28 blocks (`[6, N−2)` for ≥ 36), seq ≥ 2048 (`block_seq_len_min`); history kept on CPU; reset per chunk |
| **DreamDojo** (2B DiT) | same `apply_fastercache`, via `Video2WorldInference._apply_cache_backend` | same | the rectified-flow model always evaluates cond and uncond (`cond + g · (cond − uncond)`), even at the shipped `guidance 0`, so both mechanisms are active |
| **Ctrl-World** (SVD U-Net) | `adapter.py::enable_fastercache` installs `processor.py::SVDFasterCacheAttnProcessor` via `unet.set_attn_processor`; CFG shortcut in `models/Ctrl-World/models/pipeline_ctrl_world.py` (`_prepare_fastercache_step`) | self-attn outputs only | layers after the first 2 and only at seq lengths `{180, 720, 2880}` (`ELIGIBLE_SEQ_LENS`) — **effectively a no-op** (see limitations) |

## Flags and shipped defaults

| knob | DreamGen / DreamDojo | Ctrl-World | env |
|---|---|---|---|
| enable | `--fastercache-enabled` (v2 path) | `--use_fastercache` | — |
| start | `--fastercache-start-step` | `--fastercache_start_step` | `FASTERCACHE_START_STEP` |
| model interval | `--fastercache-model-interval` | `--fastercache_model_interval` | `FASTERCACHE_MODEL_INTERVAL` |
| block interval | `--fastercache-block-interval` | `--fastercache_block_interval` | `FASTERCACHE_BLOCK_INTERVAL` |

Shipped `start / model_int / block_int` ([configs.md](../../../docs/acceleration/configs.md)):

| backbone | cells |
|---|---|
| DreamGen | 0/5/3 single_arm MV + SV, humanoid SV; 0/2/2 bimanual MV + SV; 0/5/5 humanoid MV |
| DreamDojo | 3/7/4 single_arm; 0/5/3 bimanual, humanoid |
| Ctrl-World | 0/5/3 single_arm 3-view, bimanual 4-view; humanoid 4-view: no launcher |

## Launchers

`scripts/run/<family>/run_<emb>_<view>_<model>[_4view_grid]_fastercache.sh`, output under
`sampling_dataset/fastercache/...`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_fastercache.sh
```

## Known limitations

- **Ctrl-World: N/A in practice.** CFG is off (`guidance_scale = 1.0`), so the CFG shortcut never
  fires; on the 4-view grid the self-attention lengths are 3840/960/240, none in
  `ELIGIBLE_SEQ_LENS`, so nothing is reused. Not in the released Ctrl-World coverage.
- Can be slower than dense when the FFT delta is recomputed too often (DreamGen bimanual
  `2/2` ≈ 0.94× MV / 0.76× SV) and corrupts when pushed (humanoid MV `block_int ≤ 3` or
  `model_int ≥ 8`).
- DreamGen also has a legacy v1 path (`--use_fastercache`, `config.net.fastercache_config`);
  launchers use v2. Single GPU only.
