"""Train the models of experiments A-D (SLAN-tool paper, Fig. 5; see
data/experiments.py) on the shared leakage-free split.

Every model uses the original SLAN-tool paper's recipe (data/const.py,
train/teacher_model.build_finetune_model): ImageNet-pretrained EfficientNetB4
fine-tuned in ONE stage (no linear probing) for up to 50 epochs with early
stopping (patience 5, on validation accuracy, best weights restored), Adam,
dropout 0.2, RandAugment on training images only. Not reported by the paper
and chosen here: learning rate 1e-4, batch size 32, backbone BatchNorm
frozen, balanced class weights.

Output: models/experiments/<experiment dir>/<model>/<model>_best.keras, plus
<model>_history.csv and <model>_test_metrics.json. Resumable: a model whose
_test_metrics.json exists is skipped; an interrupted model resumes from its
last finished epoch.

Usage (from classify_handshapes/):
    python -m train.train_experiment A B C D     # all experiments
    python -m train.train_experiment D           # one experiment
"""
import json
import os
import sys

import keras
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import BackupAndRestore, CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from data.const import PAPER_EPOCHS, PAPER_LEARNING_RATE, PAPER_PATIENCE
from data.experiments import EXPERIMENTS
from data.teacher_dataset import load_experiment_datasets
from train.teacher_model import build_finetune_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get('EXPERIMENT_MODELS_DIR', os.path.join(ROOT, 'models', 'experiments'))
# Repeat runs: export SEED=2 (with EXPERIMENT_MODELS_DIR / RESULTS_DIR pointing
# to that seed's folders, see run_seeds.sh). Unset = the first, unseeded run.
SEED = int(os.environ['SEED']) if os.environ.get('SEED') else None


def _class_weight(train_label_indices):
    return dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(train_label_indices), y=train_label_indices)
    ))


def _callbacks(model_dir, file_prefix, patience):
    history = os.path.join(model_dir, file_prefix + '_history.csv')
    # After an interruption, keep the best validation accuracy reached before
    # it, so a worse epoch can't overwrite <model>_best.keras.
    best_so_far = None
    if os.path.exists(history) and os.path.getsize(history):
        best_so_far = float(pd.read_csv(history)['val_accuracy'].max())
    return [
        EarlyStopping(monitor='val_accuracy', patience=patience, mode='max',
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(model_dir, file_prefix + '_best.keras'),
                        monitor='val_accuracy', mode='max', save_best_only=True, verbose=1,
                        initial_value_threshold=best_so_far),
        CSVLogger(history, append=True),
        # Resume an interrupted model from its last finished epoch.
        BackupAndRestore(os.path.join(model_dir, 'backup')),
    ]


def train_model(model_dir, model_name, category_of, keep):
    if os.path.exists(os.path.join(model_dir, model_name + '_test_metrics.json')):
        print('[skip] {}: already trained'.format(model_dir), flush=True)
        return
    os.makedirs(model_dir, exist_ok=True)
    if SEED is not None:
        # Head initialization, dropout and RandAugment draws; GPU kernels can
        # still add small run-to-run differences.
        keras.utils.set_random_seed(SEED)
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_experiment_datasets(category_of, keep)
    print('[train] {}: {} classes, {} training images'.format(
        model_dir, num_classes, len(train_label_indices)), flush=True)
    class_weight = _class_weight(train_label_indices)

    model = build_finetune_model(model_name, num_classes, PAPER_LEARNING_RATE)
    model.fit(train_ds, epochs=PAPER_EPOCHS, validation_data=val_ds, class_weight=class_weight,
              callbacks=_callbacks(model_dir, model_name, PAPER_PATIENCE))
    model = load_model(os.path.join(model_dir, model_name + '_best.keras'))

    test_metrics = model.evaluate(test_ds)
    with open(os.path.join(model_dir, model_name + '_test_metrics.json'), 'w') as f:
        json.dump(dict(zip(model.metrics_names, test_metrics)), f, indent=2)


def train_experiment(key):
    experiment = EXPERIMENTS[key]
    print('==== Experiment {} ({} models)'.format(experiment['title'], len(experiment['models'])), flush=True)
    for model_name, category_of, keep in experiment['models']:
        train_model(os.path.join(MODELS_DIR, experiment['dir'], model_name), model_name, category_of, keep)


if __name__ == '__main__':
    keys = sys.argv[1:] or sorted(EXPERIMENTS)
    unknown = [k for k in keys if k not in EXPERIMENTS]
    if unknown:
        sys.exit('unknown experiment(s): {} (choose from {})'.format(unknown, sorted(EXPERIMENTS)))
    for key in keys:
        train_experiment(key)
