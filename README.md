# classify_handshapes

Hierarchical handshape classifier: 1 broad model (7 families) + 7 family
models (34 fine-grained classes), EfficientNetB4.

## Leakage-free split

All 8 models share one train/val/test split, `data/splits/grouped_split.csv`
(80/10/10 per class). Frames of the same source video, byte-identical files
and near-duplicate images are always in the same split. Images whose identical
copies carry two different labels are excluded (`data/splits/label_conflicts.csv`:
55 pairs 3_1/3_2, 5 pairs 3_1/5_1). Classes 1_5 and 7_6_2 are excluded (too few
images).

```bash
python -m data.make_grouped_split   # rebuild the split (deterministic)
python -m data.check_split          # verify: no video / duplicate / near-duplicate crosses splits
```

## Training on a GPU server

Needed on the server: this repo (code + `data/splits/`) and the dataset
folder with one subfolder per class (`1_1/`, `1_3/`, ..., `7_6_4/`; not in
the repo, copy it separately).

```bash
git clone https://github.com/naziyazhanabayeva/handshape-hierarchy-training.git
cd handshape-hierarchy-training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt          # Linux GPU: pip install "tensorflow[and-cuda]>=2.16" first
tmux new -s handshapes
export TEACHER_DATASET_DIR=/path/to/dataset
bash run_training.sh
```

`run_training.sh` checks the split for leakage, trains the 8 models
(`models/teachers_grouped/`) and evaluates them (`results/paper/`). It is
resumable: re-running skips finished models. Detach with `Ctrl-b d`, reattach
with `tmux attach -t handshapes`.

## Results for the paper (`results/paper/`)

| File | Content |
|---|---|
| `table_dataset.tex` | Table 2: images per family and split |
| `table_models.tex` | Per-model test accuracy and macro-F1 |
| `table_hierarchy.tex` | Full pipeline: routing accuracy, 34-class accuracy, macro-F1, weighted-F1, 95% bootstrap CIs |
| `per_class_report.csv` | Precision / recall / F1 / support per class |
| `confusion_matrix_34.csv` | 34x34 confusion matrix |
| `predictions.csv` | Per-image predictions |
