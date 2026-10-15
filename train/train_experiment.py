"""Train the models of experiments A-D (SLAN-tool paper, Fig. 5; see
data/experiments.py) on the shared leakage-free split.

Every model uses the same recipe (data/const.py, train/teacher_model.py):
EfficientNetB4 (ImageNet weights), stage 1 linear probe (frozen backbone,
Adam 1e-3, up to 50 epochs, early stopping patience 5), stage 2 fine-tuning
of block6/block7/top (Adam 1e-5, up to 30 epochs, patience 5), dropout 0.2,
balanced class weights, training-only augmentation, best checkpoint by
validation accuracy.

Output: models/experiments/<experiment dir>/<model>/<model>_best.keras, plus
training histories and <model>_test_metrics.json. Resumable: a model whose
_test_metrics.json exists is skipped; a model that finished stage 1 resumes
from its linear-probe checkpoint.

Usage (from classify_handshapes/):
    python -m train.train_experiment A B C D     # all experiments
    python -m train.train_experiment D           # one experiment
"""
import json
import os
import sys

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from data.const import FT_EPOCHS, FT_LEARNING_RATE, FT_PATIENCE, N_EPOCHS, PATIENCE
from data.experiments import EXPERIMENTS
from data.teacher_dataset import load_experiment_datasets
from train.teacher_model import build_teacher_model, unfreeze_for_finetuning

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get('EXPERIMENT_MODELS_DIR', os.path.join(ROOT, 'models', 'experiments'))


def _class_weight(train_label_indices):
    return dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(train_label_indices), y=train_label_indices)
    ))


def _callbacks(model_dir, file_prefix, patience):
    return [
        EarlyStopping(monitor='val_accuracy', patience=patience, mode='max',
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(model_dir, file_prefix + '_best.keras'),
                        monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        CSVLogger(os.path.join(model_dir, file_prefix + '_history.csv')),
    ]


def train_model(model_dir, model_name, category_of, keep):
    if os.path.exists(os.path.join(model_dir, model_name + '_test_metrics.json')):
        print('[skip] {}: already trained'.format(model_dir), flush=True)
        return
    os.makedirs(model_dir, exist_ok=True)
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_experiment_datasets(category_of, keep)
    print('[train] {}: {} classes, {} training images'.format(
        model_dir, num_classes, len(train_label_indices)), flush=True)
    class_weight = _class_weight(train_label_indices)

    # Stage 1: linear probe (backbone fully frozen, head only).
    linear_probe_best = os.path.join(model_dir, model_name + '_linear_probe_best.keras')
    if os.path.exists(linear_probe_best):
        print('[resume] {}: using existing linear-probe checkpoint'.format(model_dir), flush=True)
    else:
        model = build_teacher_model(model_name, num_classes)
        model.fit(train_ds, epochs=N_EPOCHS, validation_data=val_ds, class_weight=class_weight,
                  callbacks=_callbacks(model_dir, model_name + '_linear_probe', PATIENCE))

    # Stage 2: unfreeze the top blocks, fine-tune at a much lower LR.
    model = load_model(linear_probe_best)
    model = unfreeze_for_finetuning(model, FT_LEARNING_RATE)
    model.fit(train_ds, epochs=FT_EPOCHS, validation_data=val_ds, class_weight=class_weight,
              callbacks=_callbacks(model_dir, model_name, FT_PATIENCE))
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
