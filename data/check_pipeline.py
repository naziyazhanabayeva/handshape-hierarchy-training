"""Sanity checks of the input pipeline and model, run before training.
Guards against the bugs this project has had before:

  1. no double rescaling: the loader feeds raw [0, 255] pixels and the model
     contains exactly one 1/255 Rescaling layer (EfficientNetB4's own);
  2. augmentation (RandAugment, as in the original paper) is applied exactly
     once, only to training data, contains no flips, and the model itself has
     no augmentation layers; the model is set up for single-stage
     fine-tuning (backbone trainable, backbone BatchNorm frozen);
  3. validation/test pipelines are deterministic (no augmentation);
  4. evaluation preprocessing is identical to training preprocessing;
  5. for every model of every experiment, no image is in two of its splits
     and every image keeps the split it has in grouped_split.csv.
Exits with status 1 if any check fails.

Usage (from classify_handshapes/):  python -m data.check_pipeline
"""
import sys

import numpy as np
import tensorflow as tf

import data.teacher_dataset as td
from data.experiments import EXPERIMENTS, FINE_CLASSES
from data.const import PAPER_LEARNING_RATE
from train.teacher_model import build_finetune_model

failures = []


def check(ok, message):
    print('[{}] {}'.format('ok' if ok else 'FAIL', message))
    if not ok:
        failures.append(message)


def first_batch(ds):
    return next(iter(ds))[0].numpy()


def main():
    train_df, val_df, test_df = td.load_split_dataframes()
    sample = test_df.groupby('subcategory').head(1).head(8).assign(label=lambda d: d['subcategory'])
    classes = sorted(sample['label'].unique())

    # 1. Loader output range: raw pixels, not divided by 255.
    raw = first_batch(td._make_dataset(sample, classes, augment=False, shuffle=False))
    check(raw.max() > 1.5 and raw.max() <= 255.0 and raw.min() >= 0.0,
          'loader feeds raw [0, 255] pixels (min {:.1f}, max {:.1f})'.format(raw.min(), raw.max()))

    # 4. Evaluation preprocessing == training preprocessing.
    from evaluation.evaluate_experiments import _load_image as eval_load
    eval_img = eval_load(sample['path'].iloc[0]).numpy()
    check(np.allclose(eval_img, raw[0]), 'evaluation preprocessing identical to training preprocessing')

    # 3. Validation/test pipelines are deterministic.
    again = first_batch(td._make_dataset(sample, classes, augment=False, shuffle=False))
    check(np.array_equal(raw, again), 'validation/test pipeline is deterministic (no augmentation)')

    # 2. Augmentation (RandAugment) applied exactly once, only in training.
    calls = {'n': 0}
    original = td._augment_image

    def counting(image, label):
        calls['n'] += 1
        return original(image, label)

    td._augment_image = counting
    try:
        td._make_dataset(sample, classes, augment=False, shuffle=False)
        check(calls['n'] == 0, 'no augmentation in validation/test pipeline')
        aug = first_batch(td._make_dataset(sample, classes, augment=True, shuffle=False))
        check(calls['n'] == 1, 'augmentation applied exactly once in training pipeline ({} map)'.format(calls['n']))
    finally:
        td._augment_image = original
    check(not np.allclose(aug, raw), 'training images are actually augmented')
    check(aug.min() >= 0.0 and aug.max() <= 255.0 and aug.max() > 1.5,
          'augmented pixels stay in [0, 255] (min {:.1f}, max {:.1f})'.format(aug.min(), aug.max()))
    check(type(td._AUGMENTER).__name__ == 'RandAugment', 'augmentation is RandAugment (original paper)')
    ops = [type(l).__name__ for l in getattr(td._AUGMENTER, '_layers', [])]
    check(bool(ops) and not any('Flip' in o for o in ops),
          'RandAugment operations contain no flips: {}'.format(ops))

    # 1/2. Model: exactly one 1/255 rescaling, no augmentation layers inside,
    # single-stage fine-tuning (backbone trainable, backbone BatchNorm frozen).
    model = build_finetune_model('check', 3, PAPER_LEARNING_RATE)
    scales = [np.ravel(l.scale) for l in model.layers if type(l).__name__ == 'Rescaling']
    n_255 = sum(np.allclose(s, 1 / 255.0) for s in scales)
    check(n_255 == 1, 'model has exactly one 1/255 Rescaling layer (found {})'.format(n_255))
    random_layers = [l.name for l in model.layers if type(l).__name__.startswith('Random')]
    check(not random_layers, 'model contains no augmentation layers {}'.format(random_layers))
    convs = [l for l in model.layers if type(l).__name__ in ('Conv2D', 'DepthwiseConv2D')]
    bns = [l for l in model.layers if type(l).__name__ == 'BatchNormalization' and l.name != 'head_bn']
    check(all(l.trainable for l in convs) and not any(l.trainable for l in bns),
          'single-stage fine-tuning: all {} backbone conv layers trainable, {} backbone BatchNorm frozen'.format(
              len(convs), len(bns)))

    # 5. Per-model split integrity for every experiment.
    split_of = {p: s for df, s in ((train_df, 'train'), (val_df, 'val'), (test_df, 'test')) for p in df['path']}
    for key, experiment in sorted(EXPERIMENTS.items()):
        for model_name, category_of, keep in experiment['models']:
            parts = [df if keep is None else df[df['subcategory'].isin(keep)] for df in (train_df, val_df, test_df)]
            paths = [set(df['path']) for df in parts]
            overlap = (paths[0] & paths[1]) | (paths[0] & paths[2]) | (paths[1] & paths[2])
            moved = sum(split_of[p] != s for df, s in zip(parts, ('train', 'val', 'test')) for p in df['path'])
            present = [set(df['subcategory']) for df in parts]
            check(not overlap and not moved and present[0] == present[1] == present[2],
                  '{} {}: {} train / {} val / {} test images, disjoint, every class in all splits'.format(
                      key, model_name, *[len(df) for df in parts]))
    check(sorted(test_df['subcategory'].unique()) == FINE_CLASSES, 'test split covers all 34 classes')

    if failures:
        print('\n{} check(s) FAILED'.format(len(failures)))
        sys.exit(1)
    print('\nOK: all pipeline checks passed.')


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
