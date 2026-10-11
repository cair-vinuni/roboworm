<div align="center">

<img src="docs/assets/figures/logo.png" alt="RoboWorM logo" width="96">

# RoboWorM: Evaluating Movement Realism, Consistency and Acceleration Effects in Embodied Video World Models

<!-- Paper badge hidden until the arXiv version is public:
<a href="https://roboworm.github.io/roboworm/"><img src="https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=white&labelColor=555555" alt="Paper"></a>
-->
<a href="https://roboworm.github.io/roboworm/"><img src="https://img.shields.io/badge/RoboWorM-Website-4c9a2a?logo=googlechrome&logoColor=white&labelColor=555555" alt="Project page"></a>
<a href="https://huggingface.co/datasets/doanh25032004/vgp_datasets"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-d4a017?logo=huggingface&logoColor=yellow&labelColor=555555" alt="Dataset on Hugging Face"></a>
<a href="https://huggingface.co/datasets/doanh25032004/video_gen_physics"><img src="https://img.shields.io/badge/Rollouts-Hugging%20Face-d4a017?logo=huggingface&logoColor=yellow&labelColor=555555" alt="Rollouts on Hugging Face"></a>
<a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Cite-1f3a5f?labelColor=555555" alt="BibTeX"></a>

<a href="https://roboworm.github.io/roboworm/#teaser"><img src="docs/assets/figures/teaser.jpg" alt="RoboWorM" width="90%"></a>

</div>

Code, data pointers and run scripts for **RoboWorM**, a physics audit for robot video
world models and the training-free methods that accelerate them. Interactive figures,
tables and rollout videos are on the **[project page](https://roboworm.github.io/roboworm/)**.

[![RoboWorM overview](docs/assets/figures/overview.png)](https://roboworm.github.io/roboworm/#abstract)

*Benchmark overview: (a) episodes collected from 5 corpora, 3 embodiments, 1–4 views and
two task horizons; (b) the physics audit with KIVA and RIGIS; (c) inference acceleration
across 3 world models and 9 methods, about 29k rollouts.*

- **Data**: 1,605 real episodes, 5 corpora, 3 embodiments (single-arm, bimanual,
  humanoid), 1–4 synced views, Markovian and non-Markovian tasks.
- **World models**: DreamGen, DreamDojo and Ctrl-World, 12 dense cells
  ([checkpoints/CHECKPOINTS.md](checkpoints/CHECKPOINTS.md)).
- **Acceleration**: 9 training-free methods (caching, sparse attention, token reduction),
  about 29k rollouts in total.
- **Metrics**: KIVA and RIGIS (ours), plus Visual Integrity, Text Alignment, FVD and
  PSNR / SSIM / LPIPS against the dense rollout.

[![Qualitative audit](docs/assets/figures/qualitative_audit.png)](https://roboworm.github.io/roboworm/#fig6)

*What the audit catches. (a) Hallucinated rollouts that VLM judges rate as good, while
KIVA (single-view) and RIGIS (multi-view) agree with human raters that they are bad.
(b) Dense vs. a well-tuned accelerator (2.1× faster, same score) vs. an over-accelerated
one (2.9× faster, kinematics and geometry break). (c) KIVA degradation grows with rollout
depth, from Markovian to non-Markovian tasks.*

See the [project page](https://roboworm.github.io/roboworm/#fig6) for the rollout videos behind these examples.

## Contents

- [Proposed metrics: KIVA and RIGIS](#proposed-metrics-kiva-and-rigis)
- [Quick start](#quick-start)
- [1. Installation](#1-installation)
- [2. Data and checkpoints](#2-data-and-checkpoints)
- [3. Training](#3-training)
- [4. Rollouts: dense and accelerated](#4-rollouts-dense-and-accelerated)
- [5. Metrics](#5-metrics)
- [6. Extending RoboWorM](#6-extending-roboworm)
- [Repository layout](#repository-layout)
- [Citation](#citation) · [License](#license) · [Acknowledgements](#acknowledgements)

Full documentation index: [docs/README.md](docs/README.md).

## Proposed metrics: KIVA and RIGIS

Two VLM-free physics metrics that check a rollout against what is already known about the
robot: its kinematic tree and its fixed camera rig.

| metric | what it checks | docs |
|---|---|---|
| **KIVA** (Kinematic Invariant Violation Assessment) | Single view: reads 3-D robot keypoints from the pixels, measures link-rigidity error and joint jerk, calibrated against real footage. | [README](metrics/kiva/README.md) · [setup](docs/metrics/kiva.md) · [#14](https://github.com/ltdoanhwork/RoboWorM/pull/14) |
| **RIGIS** (Robotic Inter-view Geometry & Intrinsic-Extrinsic Stability) | Multi view: VGGT-Ω lifts every view to depth + cameras and scores cross-view geometry (S_geo) and camera-rig stability (S_pose). | [README](metrics/rigis/README.md) · [setup](docs/metrics/rigis.md) · [#13](https://github.com/ltdoanhwork/RoboWorM/pull/13) |

## Quick start

| goal | what to do | guide |
|---|---|---|
| **Score released rollouts** (or reproduce the numbers) | download `sampling_dataset/`, run `python -m metrics.evaluate` | [docs/usage.md](docs/usage.md#a-score-released-rollouts) |
| **Evaluate your own world model** | roll out from the released inputs, save in the `sampling_dataset/` layout, score | [docs/usage.md](docs/usage.md#b-evaluate-your-own-world-model) |
| **Benchmark an acceleration method** | copy a dense launcher, add your flags, time + score against dense | [docs/usage.md](docs/usage.md#c-benchmark-an-acceleration-method) |
| **Use KIVA / RIGIS alone** | `kinescore score ...` / `run_video_metrics ...` on any robot video | [docs/usage.md](docs/usage.md#kiva-and-rigis-on-any-robot-video) |

```bash
PY=models/dreamgen/.venv/bin/python
CELL=single_arm/output/singleview/dreamgen/makovian
$PY -m metrics.evaluate --pred-root sampling_dataset/dense/$CELL \
    --metrics visual_integrity text_alignment --ref-template '' --out results/metrics/dense_sa
$PY -m metrics.evaluate --pred-root sampling_dataset/pisa/$CELL --ref-root sampling_dataset/dense/$CELL \
    --ref-template '{name}' --metrics fidelity --out results/metrics/pisa_sa
```

## 1. Installation

```bash
git clone --recursive https://github.com/ltdoanhwork/RoboWorM.git
# existing clone: git submodule update --init --recursive
```

Each model and metric has its own isolated environment (never mix them; launchers activate
the right one). Exact install commands, versions and a verification snippet:
[docs/setup/README.md](docs/setup/README.md).

| env | used for |
|---|---|
| `models/dreamgen/.venv` (py3.12, torch 2.6 cu124) | DreamGen, every method except SVG2, `metrics.evaluate` |
| `models/dreamgen/.venv_svg` (flashinfer 0.3.1) | DreamGen SVG2 |
| `models/DreamDojo/.venv` (py3.10, torch 2.7 cu128) | DreamDojo, all methods |
| `models/Ctrl-World/.venv` (py3.10, torch 2.7.1 cu126) | Ctrl-World, all methods |
| `metrics/rigis/.venv` (py3.13) | RIGIS |
| KIVA env (py3.10, `pip install -e "metrics/kiva[dino,video,bench]"`) | KIVA |

Also required: `ffmpeg` / `ffprobe` on `PATH`, and `export HF_HUB_DISABLE_XET=1` for
Hugging Face downloads on parallel filesystems.

## 2. Data and checkpoints

[![Benchmark composition](docs/assets/figures/dataset_composition.png)](https://roboworm.github.io/roboworm/#benchmark)

| download | HF repo | put it in |
|---|---|---|
| source / training data (LeRobot) | [`vgp_datasets`](https://huggingface.co/datasets/doanh25032004/vgp_datasets) | `datasets/` |
| benchmark rollouts (dense + 9 methods) | [`video_gen_physics`](https://huggingface.co/datasets/doanh25032004/video_gen_physics) | `sampling_dataset/` |
| base, metric and fine-tuned weights | `vgp_checkpoints_*`, `vgp_exp_logs` | `checkpoints/` |

```bash
export HF_HUB_DISABLE_XET=1
hf download doanh25032004/vgp_datasets      --repo-type dataset --local-dir datasets --include "humanoid/singleview/*"
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset --include "dense/*"
```

- All download commands, the expected tree and a sanity check: [docs/datasets/download.md](docs/datasets/download.md)
- Corpora, splits and layout: [docs/datasets/README.md](docs/datasets/README.md), [docs/datasets/sampling_dataset.md](docs/datasets/sampling_dataset.md)
- Checkpoint of every cell and how to restore it: [checkpoints/CHECKPOINTS.md](checkpoints/CHECKPOINTS.md)

## 3. Training

Optional: all fine-tuned checkpoints are released. Three cells use released base weights;
the other 9 were post-trained, each on one prepared input built from `datasets/`.
Checkpoint → prepare script → training input → launcher: [docs/training/README.md](docs/training/README.md).

```bash
NPROC=4 bash finetuning/dreamgen/train_bimanual_16fps.sh                      # DreamGen 2B
NPROC=2 bash finetuning/dreamdojo/launch.sh dreamdojo_2b_480_640_single_arm_sv  # DreamDojo 2B
NPROC=4 bash scripts/train/run_humanoid_multiview_ctrlworld_train_4view_grid.sh # Ctrl-World
```

## 4. Rollouts: dense and accelerated

Launchers live in `scripts/run/{dreamgen,dreamdojo,ctrlworld}/`, named
`run_<embodiment>_<view>_<model>[_4view_grid]_<method>.sh`. They are resumable, discover
episodes automatically and write to `sampling_dataset/<method>/...`. Each accelerated
launcher equals the dense launcher of its cell plus the method flags.

```bash
bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_full.sh              # DreamGen dense
bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_dicache.sh           # DreamGen + DiCache
bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_all_methods.sh    # DreamDojo, all 9 methods
bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid.sh      # Ctrl-World dense
```

- 12 dense launchers and knobs: [docs/inference/README.md](docs/inference/README.md)
- 9 methods, (cell × method) configs, speedup protocol: [docs/acceleration/README.md](docs/acceleration/README.md)
- How each method hooks into each backbone: [methods/README.md](methods/README.md)

[![Dense world-model results](docs/assets/figures/dense_results.png)](https://roboworm.github.io/roboworm/#models)

[![Speedup vs physical audit](docs/assets/figures/speedup_tradeoff.png)](https://roboworm.github.io/roboworm/#acceleration)

*Top: dense results for every cell (Markovian / non-Markovian). Bottom: speedup over dense vs
physical audit for every method on the bimanual cells. Interactive versions on the
[project page](https://roboworm.github.io/roboworm/#acceleration).*

## 5. Metrics

One CLI for all metrics: `python -m metrics.evaluate --metrics <name> ...`.

| metric | registry name | reference |
|---|---|---|
| KIVA, RIGIS (ours) | `kiva`, `rigis` | none (physics audit) |
| Visual Integrity (CLIP-IQA+) | `visual_integrity` | none |
| Text Alignment (CLIPScore) | `text_alignment` | prompt |
| FVD (I3D) | `fvd` | shared real pool |
| PSNR / SSIM / LPIPS | `fidelity` | dense rollout of the same episode (never GT) |

- Registry, options and citations: [metrics/README.md](metrics/README.md)
- Protocol and full-benchmark sweep commands: [docs/metrics/README.md](docs/metrics/README.md)

## 6. Extending RoboWorM

Add a world model (new row), an embodiment / dataset (new column), an acceleration method or
a metric. Metrics are stored per clip, so a new entrant is comparable with every existing
number without re-scoring.

[![Extending RoboWorM](docs/assets/figures/extending.png)](docs/extending/README.md)

| add | guide |
|---|---|
| a world model | [new_world_model.md](docs/extending/new_world_model.md) |
| an embodiment / dataset | [new_embodiment_dataset.md](docs/extending/new_embodiment_dataset.md) |
| an acceleration method | [new_acceleration_method.md](docs/extending/new_acceleration_method.md) |
| a metric | [new_metric.md](docs/extending/new_metric.md) |

## Repository layout

```
datasets/          source / training data (LeRobot) + DreamGen corpora      -> HF vgp_datasets
sampling_dataset/  rollouts {method}/{embodiment}/{input,output}/...         -> HF video_gen_physics
checkpoints/       base/, finetuned/, metrics/
models/            dreamgen/ (Cosmos-Predict2), DreamDojo/, Ctrl-World/   (own .venv each)
methods/           cache_strategy/, sparse_attention/, prunning/
finetuning/        training launchers + data prep per model
scripts/           prepare/ infer/ run/ train/ metrics/ tables/ tools/       (scripts/README.md)
metrics/           evaluate.py, registry.py, kiva/, rigis/, fvd/, ...
docs/              setup, datasets, training, inference, acceleration, metrics, extending
```

## Citation

```bibtex
@inproceedings{roboworm2027,
  title     = {RoboWorM: Evaluating Movement Realism, Consistency and Acceleration Effects in Embodied Video World Models},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under review},
  url       = {https://roboworm.github.io/roboworm/}
}
```

## License

RoboWorM's own code is released under the [Apache License 2.0](LICENSE) (see also
[NOTICE](NOTICE)). Third-party code under `models/`, `methods/` and `metrics/` keeps its
own license, listed below and in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md). VGGT-Omega
(FAIR Noncommercial), DINOv3 (DINOv3 License) and pyiqa (PolyForm Noncommercial) are
non-commercial only, so the full pipeline is for non-commercial research. FasterCache and
SiTo declare no upstream license. Datasets and rollouts on Hugging Face follow the
licenses of their source corpora.

## Acknowledgements

RoboWorM builds on the code of many projects. Each project keeps its own license, and
its LICENSE file stays in the vendored directory. Paths, commits, our changes and the
license-file locations are listed in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
Dataset licenses follow the source corpora: DROID, RoboChallenge, AIST Bimanual,
RoboCOIN and GR00T Teleop GR1.

| role | project | license | in this repo |
|---|---|---|---|
| world model | [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) / [GR00T Dreams](https://github.com/NVIDIA/GR00T-Dreams) | Apache-2.0 | `models/dreamgen/` |
| world model | [DreamDojo](https://github.com/NVIDIA/DreamDojo) | Apache-2.0 | `models/DreamDojo/` |
| world model | [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) (+ [diffusers](https://github.com/huggingface/diffusers) SVD pipeline) | MIT (Apache-2.0) | `models/Ctrl-World/` |
| policy | [openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0 | `models/Ctrl-World/openpi` (submodule) |
| caching | [FasterCache](https://github.com/Vchitect/FasterCache) | none declared | `methods/cache_strategy/FasterCache/` |
| caching | [DiCache](https://github.com/Bujiazi/DiCache), [WorldCache](https://github.com/umair1221/WorldCache) | re-implemented | `methods/cache_strategy/{DiCache,WorldCache}/` |
| sparse attention | [Sparse VideoGen 1/2](https://github.com/svg-project/Sparse-VideoGen) (+ CUTLASS, FlashInfer, pybind11) | Apache-2.0 | `methods/sparse_attention/Sparse-VideoGen/` |
| sparse attention | [Radial Attention](https://github.com/mit-han-lab/radial-attention) | Apache-2.0 | `methods/sparse_attention/radial-attention/` |
| sparse attention | [PISA](https://github.com/xie-lab-ml/piecewise-sparse-attention) | MIT | `methods/sparse_attention/piecewise-sparse-attention/` |
| token reduction | [SiTo](https://github.com/EvelynZhang-epiclab/SiTo) (on [ToMeSD](https://github.com/dbolya/tomesd), MIT) | none declared | `methods/prunning/SiTo/` |
| token reduction | [Importance-Based Token Merging](https://github.com/cvlab-stonybrook/importance-token-merge) | MIT | `methods/prunning/importance-token-merge/` |
| RIGIS | [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) | FAIR Noncommercial Research | `metrics/rigis/vggt_omega/` |
| KIVA | [DINOv3](https://github.com/facebookresearch/dinov3), [pytorch-kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics), [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) | DINOv3 License, MIT, Apache-2.0 | pip / HF weights |
| motion metrics | [WorldScore](https://github.com/haoyi-duan/WorldScore) with [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [VFIMamba](https://github.com/MCG-NJU/VFIMamba), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM 2](https://github.com/facebookresearch/sam2) | MIT, BSD-3, Apache-2.0 | `metrics/WorldScore` (submodule) |
| Visual Integrity | CLIP-IQA+ via [pyiqa](https://github.com/chaofengc/IQA-PyTorch) | PolyForm Noncommercial 1.0.0 | pip |
| Text Alignment | CLIPScore via [torchmetrics](https://github.com/Lightning-AI/torchmetrics) + [CLIP](https://github.com/openai/CLIP) | Apache-2.0, MIT | pip |
| fidelity, FVD | [piq](https://github.com/photosynthesis-team/piq) ([LPIPS](https://github.com/richzhang/PerceptualSimilarity)), [I3D](https://github.com/google-deepmind/kinetics-i3d) via [StyleGAN-V](https://github.com/universome/stylegan-v) export | Apache-2.0, BSD-2 | pip / `checkpoints/metrics/fvd/` |

Motion Amplitude (RBench) and FVD are re-implemented from their papers.
