# Downloading data, rollouts and checkpoints

Every launcher, prepare script and metric runner resolves paths relative to the repo root.
Run every download from the repo root, with exactly the `--local-dir` targets below. The
three trees are gitignored; only `checkpoints/CHECKPOINTS.md` is tracked.

```bash
pip install -U "huggingface_hub[cli]"
export HF_HUB_DISABLE_XET=1     # the Xet backend fails on parallel filesystems
```

`hf download` is the current name of `huggingface-cli download`; both take the same arguments.

## Where each download goes

| download | HF repo (type) | `--local-dir` | read by |
|---|---|---|---|
| source / training data | `doanh25032004/vgp_datasets` (dataset) | `datasets` | `scripts/run/*` (`--dataset-base datasets/<emb>/<view>`), `scripts/prepare/`, `finetuning/` |
| benchmark rollouts | `doanh25032004/video_gen_physics` (dataset) | `sampling_dataset` | `metrics.evaluate`, `scripts/metrics/`, `assets/scripts/`; launchers write new rollouts here |
| base weights | `doanh25032004/vgp_checkpoints_{nvidia,dreamdojo,google_t5}` (model) | `checkpoints/base/{nvidia,dreamdojo,google-t5}` | all model launchers |
| other base + metric weights | `doanh25032004/vgp_checkpoints_own` (model) | `checkpoints`, then re-layout | Ctrl-World, FVD I3D, WorldScore probes (KIVA / RIGIS weights: see CHECKPOINTS.md, Metrics) |
| our fine-tuned weights | `doanh25032004/vgp_exp_logs` (model) | `exp_logs`, then re-layout | every fine-tuned cell (`checkpoints/finetuned/...`) |

The re-layout snippet for the last two rows is in
[checkpoints/CHECKPOINTS.md](../../checkpoints/CHECKPOINTS.md#restore-from-huggingface).
The full backup / restore recipe is in [../setup/hf_backup_restore.md](../setup/hf_backup_restore.md).

## Source / training data

<https://huggingface.co/datasets/doanh25032004/vgp_datasets> (36.4 GB, LeRobot format):

```bash
# LeRobot sources + the DreamGen training corpora
hf download doanh25032004/vgp_datasets --repo-type dataset --local-dir datasets \
    --include "single_arm/*" "bimanual/*" "humanoid/*" \
              "single_arm_static_dreamgen_16fps/*" "bimanual_dreamgen_16fps/*" "humanoid_dreamgen_16fps/*" \
              "bimanual_multiview_grid_seg16_dreamgen/*" "humanoid_multiview_grid_seg16_dreamgen/*"
# one embodiment/view only
hf download doanh25032004/vgp_datasets --repo-type dataset --local-dir datasets --include "humanoid/singleview/*"
```

## Benchmark rollouts

<https://huggingface.co/datasets/doanh25032004/video_gen_physics> (178 GB, about 136k files).
Paths follow `sampling_dataset/{method}/{embodiment}/{input,output}/{view}/{model}/{split}/...`;
`input/` exists only under `dense/`.

```bash
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset            # all
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "dense/*" "pisa/*"                                                                        # two methods
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "*/bimanual/input/multiview/dreamgen/*" "*/bimanual/output/multiview/dreamgen/*"          # one cell
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "data_for_web/*.json" "data_for_web/*.csv"                                                # human labels
```

## Checkpoints

```bash
hf download doanh25032004/vgp_checkpoints_nvidia    --repo-type model --local-dir checkpoints/base/nvidia
hf download doanh25032004/vgp_checkpoints_dreamdojo --repo-type model --local-dir checkpoints/base/dreamdojo
hf download doanh25032004/vgp_checkpoints_google_t5 --repo-type model --local-dir checkpoints/base/google-t5
hf download doanh25032004/vgp_checkpoints_own       --repo-type model --local-dir checkpoints   # then re-layout, see CHECKPOINTS.md
```

## Expected tree

```
datasets/
  single_arm/{singleview,multiview}/{makovian,non_makovian}/        LeRobot: meta/ data/ videos/
  bimanual/{singleview,multiview}/{makovian,non_makovian}/<task>/    one dir per task
  humanoid/singleview/{makovian,non_makovian}/                       + captions/
  humanoid/multiview/{makovian,non_makovian}/<task>/                 RoboCOIN tasks
  single_arm_static_dreamgen_16fps/  bimanual_dreamgen_16fps/  humanoid_dreamgen_16fps/
  bimanual_multiview_grid_seg16_dreamgen/  humanoid_multiview_grid_seg16_dreamgen/
                                                                     DreamGen training corpora: videos/ metas/ t5_xxl/
sampling_dataset/
  {method}/{embodiment}/{input,output}/{view}/{model}/{split}/...    input/ only under dense/
  data_for_web/                                                      human labels + catalog
checkpoints/
  base/{nvidia,dreamdojo,google-t5,ctrl_world,stabilityai,openai,pi05_droid,meta-llama,droid,mjcf,...}
  finetuned/{ctrlworld,dreamdojo,dreamgen}/
  metrics/{fvd,worldscore,vggt_omega,kiva}/
```

Quick sanity check:

```bash
ls datasets/humanoid/singleview/makovian/meta            # info.json episodes.jsonl tasks.jsonl ...
ls datasets/bimanual/multiview/makovian | head -3          # close_cardboard_box close_toolbox ...
ls sampling_dataset/dense/humanoid/output/singleview/dreamgen   # makovian non_makovian
ls checkpoints/base checkpoints/finetuned checkpoints/metrics
```

Metric outputs and report tables are written to `results/` (gitignored, created on first run).
Corpora, Markovian labelling and dataset variants: [README.md](README.md). Rollout layout and
input builders per cell: [sampling_dataset.md](sampling_dataset.md).
