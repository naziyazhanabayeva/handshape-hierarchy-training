#!/usr/bin/env bash
# Repeat runs of experiments A-D with other random seeds, strictly one
# experiment at a time (never two trainings on the GPUs at once). The first
# run (models/experiments/) counts as seed 1 and is never touched; seed N
# writes to models/experiments_seedN/ and results/experiments_seedN/.
#
# Run from the repo root inside tmux:
#   tmux new -d -s seeds "bash run_seeds.sh 2>&1 | tee -a logs/seeds.log"
#
# Resumable: re-running skips finished models and resumes an interrupted one.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source .venv/bin/activate
export TEACHER_DATASET_DIR="${TEACHER_DATASET_DIR:-$HOME/Desktop/dataset}"

SEEDS=("${@:-2 3}")
for seed in ${SEEDS[*]}; do
    for exp in A B C D; do
        echo "[seeds] $(date '+%F %T') seed $seed: starting experiment $exp"
        SEED=$seed \
        EXPERIMENT_MODELS_DIR="models/experiments_seed$seed" \
        RESULTS_DIR="results/experiments_seed$seed" \
            bash run_experiments.sh "$exp"
        status=$?
        echo "[seeds] $(date '+%F %T') seed $seed: experiment $exp finished (exit $status)"
        if [ $status -ne 0 ]; then
            echo "[seeds] stopping: fix the error, then re-run this script to resume"
            exit $status
        fi
    done
done
echo "[seeds] $(date '+%F %T') all seeds finished: ${SEEDS[*]}"
