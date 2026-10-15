"""Evaluate both methods on the shared leakage-free test split and write
every number/table needed for the paper to results/paper/.

Methods (same split, same EfficientNetB4 recipe):
  flat       one model over all 34 fine-grained classes (flat_34)
  hierarchy  broad model routes to one of 7 families, that family's model
             predicts the fine class (broad_category + subcategory_1..7)

Outputs:
  table_dataset.tex / dataset_distribution.csv
      per broad family: fine classes, total, train, val, test images
  table_comparison.tex / summary.json
      flat vs hierarchy on all test images: 34-class accuracy, macro-F1,
      weighted-F1 (95% bootstrap CIs) and McNemar's test on the paired
      per-image correctness
  table_models.tex / per_model_accuracy.csv
      each hierarchy model on its own test images (broad model: all test
      images; Category k model: test images whose true family is k), with
      the majority-class baseline for reference
  per_class_report.csv      per class: support, precision/recall/F1 of both methods
  confusion_matrix_34_{flat,hierarchy}.csv   rows = true, columns = predicted
  predictions.csv           per-image true and predicted labels of both methods

Usage (from classify_handshapes/, after train.train_teachers):
    python -m evaluation.evaluate_hierarchy
    TEACHER_MODELS_DIR=/path/to/models python -m evaluation.evaluate_hierarchy
"""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import binomtest
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


def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)


def weighted_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)


def bootstrap_ci(y_true, y_pred, metric, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true[idx], y_pred[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def fine_metrics(y_true, y_pred):
    metrics = {
        'accuracy_34': accuracy_score,
        'macro_f1_34': lambda a, b: macro_f1(a, b, FINE_CLASSES),
        'weighted_f1_34': lambda a, b: weighted_f1(a, b, FINE_CLASSES),
    }
    out = {}
    for name, fn in metrics.items():
        out[name] = float(fn(y_true, y_pred))
        out[name + '_ci95'] = bootstrap_ci(y_true, y_pred, fn)
    return out


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on paired per-image correctness."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {'only_flat_correct': only_a, 'only_hierarchy_correct': only_b, 'p_value': float(p)}


def dataset_distribution(train_df, val_df, test_df):
    rows = []
    for b in BROAD_CLASSES:
        n = [int((df['broad'] == b).sum()) for df in (train_df, val_df, test_df)]
        rows.append({'family': 'Category ' + b, 'fine_classes': len(BROAD_CATEGORIES[b]),
                     'total': sum(n), 'train': n[0], 'val': n[1], 'test': n[2]})
    df = pd.DataFrame(rows)
    total = df.drop(columns='family').sum()
    return pd.concat([df, pd.DataFrame([{'family': 'Total', **total.to_dict()}])], ignore_index=True)


def per_model_results(test_df, pred_broad):
    """Each hierarchy model on its own test images (the family models are
    given the images of their true family, i.e. assuming correct routing)."""
    def row(name, classes, y, pred):
        majority = y.value_counts(normalize=True).iloc[0]
        return dict(model=name, classes=len(classes), test_images=len(y), correct=int((pred == y).sum()),
                    accuracy=accuracy_score(y, pred), macro_f1=macro_f1(y, pred, classes),
                    majority_baseline=majority)

    rows = [row('Broad classifier', BROAD_CLASSES, test_df['broad'], pred_broad)]
    for b in BROAD_CLASSES:
        sub = test_df[test_df['broad'] == b]
        classes = sorted(BROAD_CATEGORIES[b])
        pred, _ = predict('subcategory_' + b, sub['path'], classes)
        rows.append(row('Category {} classifier'.format(b), classes, sub['subcategory'], pred))
    return pd.DataFrame(rows)


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
    t_data = ['\\begin{tabular}{lrrrrr}', '\\toprule',
              '\\textbf{Broad family} & \\textbf{Fine classes} & \\textbf{Total images} & '
              '\\textbf{Training images} & \\textbf{Validation images} & \\textbf{Test images} \\\\', '\\midrule']
    for _, r in dist.iterrows():
        cells = [r['family']] + ['{:,}'.format(int(r[c])) for c in ('fine_classes', 'total', 'train', 'val', 'test')]
        if r['family'] == 'Total':
            t_data.append('\\midrule')
            cells = ['\\textbf{%s}' % c for c in cells]
        t_data.append(' & '.join(cells) + ' \\\\')
    t_data += ['\\bottomrule', '\\end{tabular}']

    t_cmp = ['\\begin{tabular}{lcccc}', '\\toprule',
             '\\textbf{Method} & \\textbf{Models} & \\textbf{34-class accuracy} & '
             '\\textbf{Macro-F1} & \\textbf{Weighted-F1} \\\\', '\\midrule']
    for key, label, n_models in [('flat', 'Single 34-class model', 1),
                                 ('hierarchy', 'Hierarchy (7 families)', 8)]:
        m = summary[key]
        acc_lo, acc_hi = m['accuracy_34_ci95']
        f1_lo, f1_hi = m['macro_f1_34_ci95']
        t_cmp.append('{} & {} & {} [{}, {}] & {:.4f} [{:.4f}, {:.4f}] & {:.4f} \\\\'.format(
            label, n_models, pct(m['accuracy_34']), pct(acc_lo), pct(acc_hi),
            m['macro_f1_34'], f1_lo, f1_hi, m['weighted_f1_34']))
    t_cmp += ['\\bottomrule', '\\end{tabular}']

    t_models = ['\\begin{tabular}{lrrccc}', '\\toprule',
                '\\textbf{Model} & \\textbf{Classes} & \\textbf{Test images} & '
                '\\textbf{Test acc.} & \\textbf{Macro-F1} & \\textbf{Majority baseline} \\\\', '\\midrule']
    for _, r in models.iterrows():
        t_models.append('{} & {} & {:,} & {} & {:.4f} & {} \\\\'.format(
            r['model'], r['classes'], r['test_images'], pct(r['accuracy']), r['macro_f1'],
            pct(r['majority_baseline'])))
    t_models += ['\\bottomrule', '\\end{tabular}']

    for name, lines in [('table_dataset.tex', t_data), ('table_comparison.tex', t_cmp),
                        ('table_models.tex', t_models)]:
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            f.write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df, test_df = load_split_dataframes()
    y = test_df['subcategory'].to_numpy()

    dist = dataset_distribution(train_df, val_df, test_df)
    dist.to_csv(os.path.join(OUT_DIR, 'dataset_distribution.csv'), index=False)

    pred_flat, conf_flat = predict('flat_34', test_df['path'], FINE_CLASSES)
    pred_broad, conf_broad = predict('broad_category', test_df['path'], BROAD_CLASSES)
    pred_hier, conf_hier = run_hierarchy(test_df, pred_broad)

    models = per_model_results(test_df, pred_broad)
    models.to_csv(os.path.join(OUT_DIR, 'per_model_accuracy.csv'), index=False)
    print(models.to_string(index=False))

    summary = {
        'test_images': len(test_df),
        'flat': fine_metrics(y, pred_flat),
        'hierarchy': dict(fine_metrics(y, pred_hier),
                          routing_accuracy=float(accuracy_score(test_df['broad'], pred_broad))),
        'mcnemar_flat_vs_hierarchy': mcnemar(y, pred_flat, pred_hier),
    }
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    report = {'class': FINE_CLASSES}
    for key, pred in [('flat', pred_flat), ('hierarchy', pred_hier)]:
        p, r, f1, n = precision_recall_fscore_support(y, pred, labels=FINE_CLASSES, zero_division=0)
        report['support'] = n
        report.update({key + '_precision': p, key + '_recall': r, key + '_f1': f1})
        pd.DataFrame(confusion_matrix(y, pred, labels=FINE_CLASSES), index=FINE_CLASSES,
                     columns=FINE_CLASSES).to_csv(os.path.join(OUT_DIR, 'confusion_matrix_34_{}.csv'.format(key)))
    pd.DataFrame(report).to_csv(os.path.join(OUT_DIR, 'per_class_report.csv'), index=False)

    pd.DataFrame({'path': test_df['path'].map(lambda q: '/'.join(q.split(os.sep)[-2:])),
                  'true_class': y, 'true_broad': test_df['broad'],
                  'flat_pred': pred_flat, 'flat_confidence': conf_flat,
                  'broad_pred': pred_broad, 'broad_confidence': conf_broad,
                  'hierarchy_pred': pred_hier, 'hierarchy_confidence': conf_hier}).to_csv(
        os.path.join(OUT_DIR, 'predictions.csv'), index=False)

    write_tables(dist, models, summary)
    print('\nWrote results to', OUT_DIR)


if __name__ == '__main__':
    main()
