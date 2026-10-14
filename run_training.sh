#!/usr/bin/env bash
# Retrain the 8 handshape models on the leakage-free split and produce the
# paper tables. Run from classify_handshapes/ inside tmux:
#
#   tmux new -s handshapes
#   export TEACHER_DATASET_DIR=/path/to/dataset     # folder with 1_1/, 1_3/, ...
#   bash run_training.sh
#
# Resumable: re-running skips models that already finished.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

: "${TEACHER_DATASET_DIR:?set TEACHER_DATASET_DIR to the dataset folder (contains 1_1/, 1_3/, ...)}"
export TEACHER_DATASET_DIR

python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))" \
    | tee logs/env.log

echo "== 1/3 checking split for leakage"
python -m data.check_split 2>&1 | tee logs/check_split.log

echo "== 2/3 training 8 models"
python -m train.train_teachers "$@" 2>&1 | tee -a logs/train.log

echo "== 3/3 evaluating"
python -m evaluation.evaluate_hierarchy 2>&1 | tee logs/evaluate.log

echo "Done. Results in results/paper/, models in models/teachers_grouped/"
