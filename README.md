# classify_handshapes

Handshape classifier over 34 fine-grained classes (the Fig. 5A classes of the
SLAN-tool paper), EfficientNetB4. Three methods trained and compared on the
same split:
- **flat**: one model over all 34 classes (`flat_34`, 1 model),
- **row hierarchy**: broad model over 7 HamNoSys row families (selected
  fingers) + 7 family models (`broad_category`, `subcategory_1..7`, 8 models),
- **column hierarchy** (original paper, Fig. 5C/5D): broad model over 9 HamNoSys
  column categories (finger shape) + 8 category models; category 4 (Hooked)
  has one class and needs no model (`column_category`,
  `column_subcategory_*`, 9 models).

## Leakage-free split

All 18 models share one train/val/test split, `data/splits/grouped_split.csv`
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

`run_training.sh` checks the split for leakage, trains the 18 models of the three
methods (`models/teachers_grouped/`, about 25 GPU hours) and evaluates them (`results/paper/`). It is
resumable: re-running skips finished models. Detach with `Ctrl-b d`, reattach
with `tmux attach -t handshapes`.

### Splitting the training over two computers

Training is resumable per model, so the 18 models can be split, e.g.:

```bash
# computer 1 (~13 h)
python -m train.train_teachers flat_34 broad_category subcategory_1 subcategory_2 subcategory_3 subcategory_4 subcategory_5 subcategory_6 subcategory_7
# computer 2 (~12 h)
python -m train.train_teachers column_category column_subcategory_1 column_subcategory_2 column_subcategory_3 column_subcategory_5 column_subcategory_6 column_subcategory_7 column_subcategory_8 column_subcategory_9
```

Then copy all folders of `models/teachers_grouped/` to one computer and run
`python -m evaluation.evaluate_hierarchy`.

## Results for the paper (`results/paper/`)

| File | Content |
|---|---|
| `table_dataset.tex` | Table 2: images per family and split |
| `table_dataset_columns.tex` | The same per column category (Fig. 5C) |
| `table_comparison.tex` | The three methods: 34-class accuracy, macro-F1, weighted-F1, 95% bootstrap CIs |
| `table_models.tex` | Each model of both hierarchies: test accuracy, macro-F1, majority-class baseline |
| `summary.json` | All numbers above + routing accuracies + pairwise McNemar tests |
| `per_class_report.csv` | Precision / recall / F1 per class, each method |
| `confusion_matrix_34_{flat,row,column}.csv` | 34x34 confusion matrices |
| `predictions.csv` | Per-image predictions |
