#!/usr/bin/env bash
# Run experiments A-D of the SLAN-tool paper (Fig. 5) on the leakage-free split.
# Run from the repo root inside tmux:
#
#   tmux new -s handshapes
#   export TEACHER_DATASET_DIR=/path/to/dataset     # folder with 1_1/, 1_3/, ...
#   bash run_experiments.sh A          # one machine: experiment A
#   bash run_experiments.sh B          # other machine: experiment B
#   bash run_experiments.sh C D        # C and D together (D's full pipeline needs C)
#   bash run_experiments.sh            # everything on one machine
#
# Resumable: re-running skips models that already finished.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

: "${TEACHER_DATASET_DIR:?set TEACHER_DATASET_DIR to the dataset folder (contains 1_1/, 1_3/, ...)}"
export TEACHER_DATASET_DIR
EXPERIMENTS=("${@:-A B C D}")
TAG=$(echo "${EXPERIMENTS[*]}" | tr -d ' ')

python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))" \
    | tee "logs/env_${TAG}.log"

echo "== 1/4 checking the split for leakage (videos, duplicates, near-duplicates)"
python -m data.check_split 2>&1 | tee "logs/check_split_${TAG}.log"

echo "== 2/4 checking the pipeline (rescaling, augmentation, per-model splits)"
python -m data.check_pipeline 2>&1 | tee "logs/check_pipeline_${TAG}.log"

echo "== 3/4 training experiment(s) ${EXPERIMENTS[*]}"
python -m train.train_experiment ${EXPERIMENTS[*]} 2>&1 | tee -a "logs/train_${TAG}.log"

echo "== 4/4 evaluating every experiment whose models are present"
python -m evaluation.evaluate_experiments 2>&1 | tee "logs/evaluate_${TAG}.log"
mkdir -p results/experiments/dataset
cp "logs/check_split_${TAG}.log" results/experiments/dataset/leakage_check.txt
cp "logs/check_pipeline_${TAG}.log" results/experiments/dataset/pipeline_check.txt

echo "Done. Models in models/experiments/, results in results/experiments/"
