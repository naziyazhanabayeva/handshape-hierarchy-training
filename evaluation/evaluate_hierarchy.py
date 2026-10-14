"""Evaluate the 8 retrained models on the shared leakage-free test split and
write every number/table needed for the paper to results/paper/.

Outputs:
  table_dataset.tex / dataset_distribution.csv
      per broad family: fine classes, total, train, val, test images
  table_models.tex / per_model_accuracy.csv
      each model on its own test images (broad model: all test images;
      Category k model: test images whose true family is k):
      test images, correct, accuracy, macro-F1
  table_hierarchy.tex / summary.json
      full pipeline (broad model routes, family model predicts) on all test
      images: routing accuracy, 34-class accuracy, macro-F1, weighted-F1,
      with 95% bootstrap confidence intervals
  per_class_report.csv      precision / recall / F1 / support, 34 classes
  confusion_matrix_34.csv   rows = true class, columns = predicted class
  predictions.csv           per-image true and predicted labels

Usage (from classify_handshapes/, after train.train_teachers):
    python -m evaluation.evaluate_hierarchy
    TEACHER_MODELS_DIR=/path/to/models python -m evaluation.evaluate_hierarchy
"""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model

from data.const import BATCH_SIZE, BROAD_CATEGORIES, IMG_SIZE
from data.teacher_dataset import load_split_dataframes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get('TEACHER_MODELS_DIR', os.path.join(ROOT, 'models', 'teachers_grouped'))
OUT_DIR = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results', 'paper'))
BROAD_CLASSES = sorted(BROAD_CATEGORIES)
FINE_CLASSES = sorted(s for subs in BROAD_CATEGORIES.values() for s in subs)
N_BOOTSTRAP = 1000


def _load_image(path):
    # Identical to data/teacher_dataset._load_image: raw [0, 255] pixels,
    # EfficientNetB4 rescales internally.
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return tf.cast(image, tf.float32)


def predict(model_name, paths, classes):
    """Predicted class label and confidence for each path."""
    model = load_model(os.path.join(MODELS_DIR, model_name, model_name + '_best.keras'))
    call = tf.function(lambda x: model(x, training=False))
    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    probs = np.concatenate([call(batch).numpy() for batch in ds])
    return np.array(classes)[probs.argmax(axis=1)], probs.max(axis=1)


def bootstrap_ci(y_true, y_pred, metric, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true[idx], y_pred[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)


def dataset_distribution(train_df, val_df, test_df):
    rows = []
    for b in BROAD_CLASSES:
        n = [int((df['broad'] == b).sum()) for df in (train_df, val_df, test_df)]
        rows.append({'family': 'Category ' + b, 'fine_classes': len(BROAD_CATEGORIES[b]),
                     'total': sum(n), 'train': n[0], 'val': n[1], 'test': n[2]})
    df = pd.DataFrame(rows)
    total = df.drop(columns='family').sum()
    return pd.concat([df, pd.DataFrame([{'family': 'Total', **total.to_dict()}])], ignore_index=True)


def per_model_results(test_df):
    rows = []
    pred_broad, conf_broad = predict('broad_category', test_df['path'], BROAD_CLASSES)
    rows.append(dict(model='Broad classifier', classes=len(BROAD_CLASSES), test_images=len(test_df),
                     correct=int((pred_broad == test_df['broad']).sum()),
                     accuracy=accuracy_score(test_df['broad'], pred_broad),
                     macro_f1=macro_f1(test_df['broad'], pred_broad, BROAD_CLASSES)))
    family_preds = {}
    for b in BROAD_CLASSES:
        sub = test_df[test_df['broad'] == b]
        classes = sorted(BROAD_CATEGORIES[b])
        pred, _ = predict('subcategory_' + b, sub['path'], classes)
        family_preds[b] = pred
        rows.append(dict(model='Category {} classifier'.format(b), classes=len(classes),
                         test_images=len(sub), correct=int((pred == sub['subcategory']).sum()),
                         accuracy=accuracy_score(sub['subcategory'], pred),
                         macro_f1=macro_f1(sub['subcategory'], pred, classes)))
    return pd.DataFrame(rows), pred_broad, conf_broad


def run_hierarchy(test_df, pred_broad):
    """Route each image by the broad prediction, then classify with that
    family's model (a wrong route can only produce a wrong fine label)."""
    pred_fine = np.empty(len(test_df), dtype=object)
    conf_fine = np.zeros(len(test_df))
    for b in BROAD_CLASSES:
        idx = np.flatnonzero(pred_broad == b)
        if len(idx):
            pred_fine[idx], conf_fine[idx] = predict(
                'subcategory_' + b, test_df['path'].iloc[idx], sorted(BROAD_CATEGORIES[b]))
    return pred_fine.astype(str), conf_fine


def pct(x):
    return '{:.2f}\\%'.format(100 * x)


def write_tables(dist, models, summary):
    t2 = ['\\begin{tabular}{lrrrrr}', '\\toprule',
          '\\textbf{Broad family} & \\textbf{Fine classes} & \\textbf{Total images} & '
          '\\textbf{Training images} & \\textbf{Validation images} & \\textbf{Test images} \\\\', '\\midrule']
    for _, r in dist.iterrows():
        cells = [r['family'], r['fine_classes'], r['total'], r['train'], r['val'], r['test']]
        cells = [cells[0]] + ['{:,}'.format(int(c)) for c in cells[1:]]
        if r['family'] == 'Total':
            t2.append('\\midrule')
            cells = ['\\textbf{%s}' % c for c in cells]
        t2.append(' & '.join(cells) + ' \\\\')
    t2 += ['\\bottomrule', '\\end{tabular}']

    t3 = ['\\begin{tabular}{lrrcc}', '\\toprule',
          '\\textbf{Model} & \\textbf{Classes} & \\textbf{Test images} & '
          '\\textbf{Test acc.} & \\textbf{Macro-F1} \\\\', '\\midrule']
    for _, r in models.iterrows():
        t3.append('{} & {} & {:,} & {} & {:.4f} \\\\'.format(
            r['model'], r['classes'], r['test_images'], pct(r['accuracy']), r['macro_f1']))
    t3 += ['\\bottomrule', '\\end{tabular}']

    s = summary
    t4 = ['\\begin{tabular}{lcc}', '\\toprule',
          '\\textbf{Metric} & \\textbf{Value} & \\textbf{95\\% CI} \\\\', '\\midrule']
    for label, key in [('Broad routing accuracy (7 families)', 'routing_accuracy'),
                       ('34-class accuracy', 'accuracy_34'),
                       ('34-class macro-F1', 'macro_f1_34'),
                       ('34-class weighted-F1', 'weighted_f1_34')]:
        lo, hi = s[key + '_ci95']
        if 'f1' in key:
            t4.append('{} & {:.4f} & [{:.4f}, {:.4f}] \\\\'.format(label, s[key], lo, hi))
        else:
            t4.append('{} & {} & [{}, {}] \\\\'.format(label, pct(s[key]), pct(lo), pct(hi)))
    t4 += ['\\bottomrule', '\\end{tabular}']

    for name, lines in [('table_dataset.tex', t2), ('table_models.tex', t3), ('table_hierarchy.tex', t4)]:
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            f.write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df, test_df = load_split_dataframes()

    dist = dataset_distribution(train_df, val_df, test_df)
    dist.to_csv(os.path.join(OUT_DIR, 'dataset_distribution.csv'), index=False)

    models, pred_broad, conf_broad = per_model_results(test_df)
    models.to_csv(os.path.join(OUT_DIR, 'per_model_accuracy.csv'), index=False)
    print(models.to_string(index=False))

    pred_fine, conf_fine = run_hierarchy(test_df, pred_broad)
    y = test_df['subcategory'].to_numpy()
    summary = dict(
        test_images=len(test_df),
        routing_accuracy=accuracy_score(test_df['broad'], pred_broad),
        routing_accuracy_ci95=bootstrap_ci(test_df['broad'], pred_broad, accuracy_score),
        accuracy_34=accuracy_score(y, pred_fine),
        accuracy_34_ci95=bootstrap_ci(y, pred_fine, accuracy_score),
        macro_f1_34=macro_f1(y, pred_fine, FINE_CLASSES),
        macro_f1_34_ci95=bootstrap_ci(y, pred_fine, lambda a, b: macro_f1(a, b, FINE_CLASSES)),
        weighted_f1_34=f1_score(y, pred_fine, labels=FINE_CLASSES, average='weighted', zero_division=0),
        weighted_f1_34_ci95=bootstrap_ci(y, pred_fine, lambda a, b: f1_score(
            a, b, labels=FINE_CLASSES, average='weighted', zero_division=0)),
    )
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    p, r, f1, n = precision_recall_fscore_support(y, pred_fine, labels=FINE_CLASSES, zero_division=0)
    pd.DataFrame({'class': FINE_CLASSES, 'precision': p, 'recall': r, 'f1': f1, 'support': n}).to_csv(
        os.path.join(OUT_DIR, 'per_class_report.csv'), index=False)
    pd.DataFrame(confusion_matrix(y, pred_fine, labels=FINE_CLASSES),
                 index=FINE_CLASSES, columns=FINE_CLASSES).to_csv(os.path.join(OUT_DIR, 'confusion_matrix_34.csv'))
    pd.DataFrame({'path': test_df['path'].map(lambda q: '/'.join(q.split(os.sep)[-2:])),
                  'true_broad': test_df['broad'], 'pred_broad': pred_broad, 'broad_confidence': conf_broad,
                  'true_class': y, 'pred_class': pred_fine, 'class_confidence': conf_fine}).to_csv(
        os.path.join(OUT_DIR, 'predictions.csv'), index=False)

    write_tables(dist, models, summary)
    print('\nWrote results to', OUT_DIR)


if __name__ == '__main__':
    main()
