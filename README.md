# Handshape classification: experiments A–D

EfficientNetB4 handshape classifiers for the SLAN-tool paper. The four
experiments follow the grouping strategies of the paper's Fig. 5 (HamNoSys 4
handshape chart, Hanke 2010). All of them use the same 34 classes and the
same leakage-free split, so their test results are directly comparable.

| Experiment | Fig. 5 | What is trained | Models |
|---|---|---|---|
| **A** | 5A | one model over all 34 classes | 1 |
| **B** | 5B | broad model over 10 HamNoSys **row** categories + one model per row | 10 |
| **C** | 5C | broad model over 9 HamNoSys **column** categories | 1 |
| **D** | 5D | one model per column category; with C = the original paper's full hierarchy | 8 |

Class `r_c` is the chart cell in row `r` (selected fingers) and column `c`
(finger shape); `7_r_c` is the thumb-opposition block. Groupings are derived
from these positions (`data/experiments.py`, table in
`data/splits/class_mapping.csv`). Classes 1_5 (4 images) and 7_6_2 (19
images) are excluded as too small, so A has 34 of the 36 Fig. 5A classes.
Category B8 (7_2_2) and C4 (5_4) contain a single class and need no model.

## Data split (`data/splits/`)

- `grouped_split.csv`: one 80/10/10 train/val/test assignment per image,
  shared by all 20 models, stratified per class. Frames of the same source
  video, byte-identical files and near-duplicate images (perceptual hash) are
  always in the same split. Built by `python -m data.make_grouped_split`
  (deterministic).
- `label_conflicts.csv`: 120 images excluded because identical copies carry
  two different labels (55 pairs 3_1/3_2, 5 pairs 3_1/5_1).
- `class_mapping.csv`: class → HamNoSys row/column and B/C category.

Checks run automatically before training:
- `python -m data.check_split`: no video, identical file or near-duplicate
  crosses splits;
- `python -m data.check_pipeline`: no double rescaling (raw [0, 255] input,
  exactly one 1/255 layer in the model), RandAugment (no flips) applied once
  and only to training data, single-stage fine-tuning set-up, identical
  preprocessing at evaluation, disjoint splits for every model.

## Running on the lab computers

```bash
git clone https://github.com/naziyazhanabayeva/handshape-hierarchy-training.git
cd handshape-hierarchy-training
python3 -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]>=2.18"   # Linux GPU
pip install -r requirements-train.txt
tmux new -s handshapes
export TEACHER_DATASET_DIR=/path/to/dataset  # contains 1_1/, 1_3/, ..., 7_6_4/
bash run_experiments.sh A                    # or B, or "C D", or nothing = all
```

The dataset is not in the repo; copy it separately. Detach with `Ctrl-b d`,
reattach with `tmux attach -t handshapes`. Re-running skips finished models.

Approximate GPU time (depends on the GPU and on when early stopping
triggers): A 4–10 h, B 8–20 h, C 4–10 h, D 4–10 h. Run C and D on
the same machine (D's full 34-class result needs C's broad model). For
example: machine 1 `bash run_experiments.sh A C D`, machine 2
`bash run_experiments.sh B`.

**Combining machines:** copy the finished experiment folder from
`models/experiments/` of one machine into `models/experiments/` of the other
(e.g. `B_rows_10/`), then run `python -m evaluation.evaluate_experiments` to
produce the comparison of all experiments.

## Restoring the trained models on another computer

The 20 trained models and the dataset are not in the repo; they are kept as
two archives (Google Drive):

- `handshape_ABCD_models_20261017.tar.gz` (3.9 GB): `experiments/` with every
  `<model>_best.keras`, `_history.csv`, `_test_metrics.json`, plus `SHA256SUMS.txt`
- `handshape_dataset_20261017.tar.gz` (2.2 GB): `dataset/` (all 36 class
  folders, 99,780 images) plus `dataset_SHA256SUMS.txt`

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions used for training (TF 2.21.0, Keras 3.15.1)
mkdir -p models && tar -xzf handshape_ABCD_models_20261017.tar.gz -C models/
(cd models/experiments && sha256sum -c --quiet SHA256SUMS.txt)
tar -xzf handshape_dataset_20261017.tar.gz -C ~/Desktop/
(cd ~/Desktop && sha256sum -c --quiet dataset_SHA256SUMS.txt)
export TEACHER_DATASET_DIR=~/Desktop/dataset
```

Then `python -m evaluation.make_figures` (needs only the models archive),
`python -m evaluation.evaluate_experiments` (models + dataset, re-predicts the
test set) or `python -m data.check_split` / `check_pipeline` (dataset). Some
image file names contain `*`, so extract the dataset on Linux/macOS, not Windows.

## Results for the paper (`results/experiments/`)

| Folder | Contents |
|---|---|
| `dataset/` | split tables per class (with HamNoSys row/column), per B row category, per C column category; `leakage_check.txt`, `pipeline_check.txt` |
| `A_flat_34/` | `table_A.tex`, per-class report, 34×34 confusion matrix, predictions |
| `B_rows_10/` | `table_B.tex` (broad + each row model + full hierarchy), 10×10 and 34×34 confusion matrices, per-class report, predictions |
| `C_columns_9/` | `table_C.tex`, 9×9 confusion matrix, predictions |
| `D_column_subcategories/` | `table_D.tex` (each column model + full hierarchy C→D), per-class report, 34×34 confusion matrix, predictions |
| `figures/` | `python -m evaluation.make_figures` (PNG 300 dpi + PDF): confusion matrices (A, B 10/34, C, D 34), per-sub-model accuracy/macro-F1 (B, D), training curves (A, B broad, C broad, B and D sub-models), `comparison_34class`, `comparison_broad_level`, `per_class_f1_comparison` |
| `comparison/` | `table_comparison.tex` (A vs B vs C+D: 34-class accuracy, macro-F1 with 95% CIs), `table_broad_level.tex` (B vs C broad), `summary.json` (all numbers + McNemar tests) |

Sub-model rows use the test images of the model's own category; the full
hierarchy rows route every test image through the broad model.

## Training recipe (all models): the original SLAN-tool paper's

As reported in the paper: ImageNet-pretrained EfficientNetB4 (380×380 input)
**fine-tuned in one stage** (no linear probing) for up to **50 epochs** with
**early stopping, patience 5** (validation accuracy, best weights kept),
**Adam**, **dropout 0.2** on the top layers, **RandAugment** on training
images (2 random operations per image from auto-contrast, equalize,
brightness, color, contrast, posterize, rotation, sharpness, shear,
translation, solarize; no flips).

Not reported in the paper, chosen here: learning rate 1e-4 (`LEARNING_RATE`
env var), batch size 32 (`BATCH_SIZE` env var), backbone BatchNorm layers
frozen (Keras' recommendation for fine-tuning EfficientNet), balanced class
weights, GAP + BatchNorm + Dropout + softmax head. See `data/const.py` and
`train/teacher_model.build_finetune_model`.

**GPU memory:** the whole backbone trains, so this needs much more memory than
training the head only. If training stops with an out-of-memory (OOM) error,
re-run with a smaller batch: `export BATCH_SIZE=16` (or 8). An interrupted
model resumes from its last finished epoch.
