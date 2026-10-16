#!/usr/bin/env bash
# Runs after C+D: waits for the 'handshapes' tmux session to finish, then
# starts A, waits, then starts B. Pure tmux/bash, independent of any
# other process - safe to let run after logging off anything else.
set -u
cd "$(dirname "$0")"
export TEACHER_DATASET_DIR=/home/nurobot427/Desktop/dataset

session_exists() {
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -qx "handshapes"
}

wait_for_session_end() {
  while session_exists; do
    sleep 15
  done
}

echo "[queue] waiting for current run (session 'handshapes') to finish..."
wait_for_session_end

echo "[queue] starting experiment A"
tmux new-session -d -s handshapes "source .venv/bin/activate && export TEACHER_DATASET_DIR=$TEACHER_DATASET_DIR && bash run_experiments.sh A"
wait_for_session_end

echo "[queue] starting experiment B"
tmux new-session -d -s handshapes "source .venv/bin/activate && export TEACHER_DATASET_DIR=$TEACHER_DATASET_DIR && bash run_experiments.sh B"
wait_for_session_end

echo "[queue] all experiments A, B, C, D finished"
