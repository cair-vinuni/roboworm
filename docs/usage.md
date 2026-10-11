# Using the benchmark

Three ways to use RoboWorM. None needs training: the fine-tuned checkpoints and every
rollout are released ([downloads](datasets/download.md)).

## A. Score released rollouts

Download a few cells and score them. This also reproduces the benchmark numbers.

```bash
export HF_HUB_DISABLE_XET=1
hf download doanh25032004/video_gen_physics --repo-type dataset --local-dir sampling_dataset \
    --include "dense/single_arm/*" "pisa/single_arm/*"
PY=models/dreamgen/.venv/bin/python
CELL=single_arm/output/singleview/dreamgen/makovian
# reference-free quality of a cell
$PY -m metrics.evaluate --pred-root sampling_dataset/dense/$CELL \
    --metrics visual_integrity text_alignment --ref-template '' --out results/metrics/dense_sa
# fidelity of an accelerated cell against dense
$PY -m metrics.evaluate --pred-root sampling_dataset/pisa/$CELL --ref-root sampling_dataset/dense/$CELL \
    --ref-template '{name}' --metrics fidelity --out results/metrics/pisa_sa
```

Full-benchmark sweeps (every metric on every rollout): [metrics/README.md](metrics/README.md).

## B. Evaluate your own world model

1. **Get the episodes:** download `vgp_datasets` into `datasets/`. The conditioning inputs of
   every episode (GT clips, prompts, actions) are under `sampling_dataset/dense/<embodiment>/input/`.
2. **Generate:** roll out from the same conditioning prefix to the full GT episode length.
3. **Save:** write the rollouts into `sampling_dataset/dense/<embodiment>/output/<view>/<your_model>/<split>/`.
   Single-view clips are named `episode_XXXXXX.mp4`, with `episode_XXXXXX_gt.mp4` and a
   `.txt` prompt next to each. Multi-view rollouts go in one directory per episode.
4. **Score:** Visual Integrity, Text Alignment, FVD, plus KIVA (single-view) or RIGIS
   (multi-view), see [metrics/README.md](metrics/README.md).

Your rows are then directly comparable with the released ones. Interface, launcher template
and checklist: [extending/new_world_model.md](extending/new_world_model.md).

## C. Benchmark an acceleration method

1. **Run it:** copy the dense launcher of a cell and add only your method flags
   ([extending/new_acceleration_method.md](extending/new_acceleration_method.md)).
2. **Time it:** against the dense run on the same GPU
   ([acceleration/README.md](acceleration/README.md), speedup protocol).
3. **Score it:** fidelity against dense, plus KIVA / RIGIS for physical integrity.

## KIVA and RIGIS on any robot video

```bash
# KIVA: one reader + thresholds per robot (metrics/kiva/README.md)
kinescore score --model ckpt/a1x_ee.a1x_sv.sv1_4x3.pt \
    --threshold metrics/kiva/thresholds/a1x_ee.a1x_sv.sv1_4x3.json --videos /path/to/clips
# RIGIS: start the VGGT-Ω service, then score episode dirs with one folder per view (metrics/rigis/README.md)
cd metrics/rigis && python -m multiview_consistency.run_video_metrics \
    --episodes_root /path/to/episodes --views "view1,view2" --out_csv rigis.csv --port 6111
```

Details: [../metrics/kiva/README.md](../metrics/kiva/README.md),
[../metrics/rigis/README.md](../metrics/rigis/README.md).
