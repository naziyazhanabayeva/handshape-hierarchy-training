"""Train the 8 teacher models (1 broad-category + 7 subcategory models)
for the noisy-student pipeline, following the paper's reported training
regime: EfficientNetB4, Adam, dropout 0.2, early stopping (patience 5),
up to 50 epochs, with augmentation. See data/const.py for the reasonable
defaults chosen for parameters the paper did not report (learning rate,
batch size, which blocks to unfreeze, pseudo-label confidence threshold).

Training is two-stage per model, matching the recipe already proven on
this dataset in old/finetune_hierarchical.py:
  1. Linear probe: backbone fully frozen, only the new head trains.
  2. Fine-tune: unfreeze the top blocks, much lower LR, continue training
     from the best linear-probe checkpoint.
Both stages use class_weight='balanced' since the broad categories and
several subcategories are heavily imbalanced (e.g. broad category 5 has
~10x the images of broad category 3).

All 8 models use the shared leakage-free split in data/splits/grouped_split.csv.

Resumable: a model whose <model>_test_metrics.json exists is skipped, and a
model that finished stage 1 resumes from its linear-probe checkpoint.

Usage (from classify_handshapes/):
    python -m train.train_teachers                  # all 18 models of the 3 methods
    python -m train.train_teachers flat_34 column_category   # only these
"""
import json
import os
import sys

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model

from data.const import BROAD_CATEGORIES, COLUMN_CATEGORIES, N_EPOCHS, PATIENCE, FT_LEARNING_RATE, FT_EPOCHS, FT_PATIENCE
from data.teacher_dataset import (
    load_broad_category_datasets, load_column_category_datasets, load_column_subcategory_datasets,
    load_flat_datasets, load_subcategory_datasets,
)
from train.teacher_model import build_teacher_model, unfreeze_for_finetuning

MODELS_DIR = os.environ.get(
    'TEACHER_MODELS_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'teachers_grouped')
)


def _class_weight(train_label_indices):
    return dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(train_label_indices), y=train_label_indices)
    ))


def _callbacks(model_dir, file_prefix, patience):
    os.makedirs(model_dir, exist_ok=True)
    return [
        EarlyStopping(
            monitor='val_accuracy', patience=patience, mode='max',
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            os.path.join(model_dir, file_prefix + '_best.keras'),
            monitor='val_accuracy', mode='max', save_best_only=True, verbose=1,
        ),
        CSVLogger(os.path.join(model_dir, file_prefix + '_history.csv')),
    ]


def _train(model_name, num_classes, train_ds, val_ds, test_ds, train_label_indices):
    model_dir = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(os.path.join(model_dir, model_name + '_test_metrics.json')):
        print('[skip] {}: already trained in {}'.format(model_name, model_dir))
        return None
    os.makedirs(model_dir, exist_ok=True)
    class_weight = _class_weight(train_label_indices)

    # Stage 1: linear probe (backbone fully frozen, head only).
    linear_probe_best = os.path.join(model_dir, model_name + '_linear_probe_best.keras')
    if os.path.exists(linear_probe_best):
        print('[resume] {}: using existing linear-probe checkpoint'.format(model_name))
    else:
        model = build_teacher_model(model_name, num_classes)
        model.fit(
            train_ds, epochs=N_EPOCHS, validation_data=val_ds, class_weight=class_weight,
            callbacks=_callbacks(model_dir, model_name + '_linear_probe', PATIENCE),
        )

    # Stage 2: unfreeze top blocks, fine-tune at a much lower LR.
    model = load_model(linear_probe_best)
    model = unfreeze_for_finetuning(model, FT_LEARNING_RATE)
    model.fit(
        train_ds, epochs=FT_EPOCHS, validation_data=val_ds, class_weight=class_weight,
        callbacks=_callbacks(model_dir, model_name, FT_PATIENCE),
    )
    model = load_model(os.path.join(model_dir, model_name + '_best.keras'))

    test_metrics = model.evaluate(test_ds)
    with open(os.path.join(model_dir, model_name + '_test_metrics.json'), 'w') as f:
        json.dump(dict(zip(model.metrics_names, test_metrics)), f, indent=2)

    model.save(os.path.join(model_dir, model_name + '_final.keras'))
    return model


def train_broad_category_model():
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_broad_category_datasets()
    return _train('broad_category', num_classes, train_ds, val_ds, test_ds, train_label_indices)


def train_subcategory_model(broad_category):
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_subcategory_datasets(broad_category)
    model_name = 'subcategory_{}'.format(broad_category)
    return _train(model_name, num_classes, train_ds, val_ds, test_ds, train_label_indices)


def train_flat_model():
    """Baseline: one model over all 34 fine-grained classes, same recipe and
    split as the hierarchy, to test whether the hierarchy is needed."""
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_flat_datasets()
    return _train('flat_34', num_classes, train_ds, val_ds, test_ds, train_label_indices)


def train_column_category_model():
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_column_category_datasets()
    return _train('column_category', num_classes, train_ds, val_ds, test_ds, train_label_indices)


def train_column_subcategory_model(column):
    train_ds, val_ds, test_ds, num_classes, train_label_indices = load_column_subcategory_datasets(column)
    return _train('column_subcategory_' + column, num_classes, train_ds, val_ds, test_ds, train_label_indices)


def all_model_names():
    return (['flat_34', 'broad_category']
            + ['subcategory_' + b for b in BROAD_CATEGORIES]
            + ['column_category']
            + ['column_subcategory_' + c for c, subs in COLUMN_CATEGORIES.items() if len(subs) > 1])


def train_all_teacher_models(model_names=None):
    """Trains every model of the three methods (or only `model_names`):
      flat_34                                  single 34-class model
      broad_category + subcategory_1..7        hierarchy by HamNoSys row (7 families)
      column_category + column_subcategory_*   hierarchy by HamNoSys column (original
                                               paper, Fig. 5C/5D; column 4 has one
                                               class and needs no sub-model)
    """
    for name in model_names or all_model_names():
        if name == 'flat_34':
            train_flat_model()
        elif name == 'broad_category':
            train_broad_category_model()
        elif name == 'column_category':
            train_column_category_model()
        elif name.startswith('column_subcategory_'):
            train_column_subcategory_model(name.rsplit('_', 1)[1])
        elif name.startswith('subcategory_'):
            train_subcategory_model(name.rsplit('_', 1)[1])
        else:
            raise ValueError('unknown model name: ' + name)


if __name__ == '__main__':
    train_all_teacher_models(sys.argv[1:] or None)
