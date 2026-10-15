"""Evaluate both methods on the shared leakage-free test split and write
every number/table needed for the paper to results/paper/.

Methods (same split, same EfficientNetB4 recipe):
  flat       one model over all 34 fine-grained classes (flat_34)
  row        hierarchy by HamNoSys row: broad model routes to one of 7
             families, that family's model predicts the fine class
             (broad_category + subcategory_1..7)
  column     hierarchy of the original paper (Fig. 5C/5D) by HamNoSys
             column: broad model routes to one of 9 column categories, that
             category's model predicts the fine class (column_category +
             column_subcategory_*; category 4 has a single class)

Outputs:
  table_dataset.tex / dataset_distribution.csv
      per broad family (HamNoSys row): fine classes, total, train, val, test images
  table_dataset_columns.tex / dataset_distribution_columns.csv
      the same per HamNoSys column category (original paper's Fig. 5C)
  table_comparison.tex / summary.json
      the three methods on all test images: 34-class accuracy, macro-F1,
      weighted-F1 (95% bootstrap CIs), routing accuracy, and pairwise
      McNemar tests on the paired per-image correctness
  table_models.tex / per_model_accuracy.csv
      each model of both hierarchies on its own test images (broad model:
      all test images; sub-model: test images whose true group is its
      group), with the majority-class baseline for reference
  per_class_report.csv      per class: support, precision/recall/F1 of each method
  confusion_matrix_34_{flat,row,column}.csv   rows = true, columns = predicted
  predictions.csv           per-image true and predicted labels of each method

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

from data.const import BATCH_SIZE, BROAD_CATEGORIES, COLUMN_CATEGORIES, IMG_SIZE
from data.teacher_dataset import load_split_dataframes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get('TEACHER_MODELS_DIR', os.path.join(ROOT, 'models', 'teachers_grouped'))
OUT_DIR = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results', 'paper'))
BROAD_CLASSES = sorted(BROAD_CATEGORIES)
# name, true-group column in the split, categories, broad model, sub-model prefix, row label
HIERARCHIES = [
    ('row', 'broad', BROAD_CATEGORIES, 'broad_category', 'subcategory_', 'Family'),
    ('column', 'column', COLUMN_CATEGORIES, 'column_category', 'column_subcategory_', 'Column category'),
]
FINE_CLASSES = sorted(s for subs in BROAD_CATEGORIES.values() for s in subs)
N_BOOTSTRAP = 1000
# key, table label, number of trained models
METHODS = [('flat', 'Single 34-class model', 1),
           ('row', 'Hierarchy by HamNoSys row (7 families)', 8),
           ('column', 'Hierarchy by HamNoSys column (9 categories, Fig. 5C/D)', 9)]


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
    return {'only_first_correct': only_a, 'only_second_correct': only_b, 'p_value': float(p)}


def dataset_distribution(train_df, val_df, test_df, group_col='broad', categories=BROAD_CATEGORIES,
                         label='Category'):
    rows = []
    for b in sorted(categories):
        n = [int((df[group_col] == b).sum()) for df in (train_df, val_df, test_df)]
        rows.append({'family': '{} {}'.format(label, b), 'fine_classes': len(categories[b]),
                     'total': sum(n), 'train': n[0], 'val': n[1], 'test': n[2]})
    df = pd.DataFrame(rows)
    total = df.drop(columns='family').sum()
    return pd.concat([df, pd.DataFrame([{'family': 'Total', **total.to_dict()}])], ignore_index=True)


def per_model_results(test_df, hierarchy, pred_group):
    """Each model of one hierarchy on its own test images (sub-models are
    given the images of their true group, i.e. assuming correct routing)."""
    name, group_col, categories, broad_model, prefix, label = hierarchy

    def row(model, classes, y, pred):
        return dict(hierarchy=name, model=model, classes=len(classes), test_images=len(y),
                    correct=int((pred == y).sum()), accuracy=accuracy_score(y, pred),
                    macro_f1=macro_f1(y, pred, classes),
                    majority_baseline=y.value_counts(normalize=True).iloc[0])

    groups = sorted(categories)
    broad_label = {'row': 'Broad classifier (7 families)', 'column': 'Broad classifier (9 column categories)'}[name]
    rows = [row(broad_label, groups, test_df[group_col], pred_group)]
    for g in groups:
        classes = sorted(categories[g])
        if len(classes) < 2:
            continue
        sub = test_df[test_df[group_col] == g]
        pred, _ = predict(prefix + g, sub['path'], classes)
        rows.append(row('{} {} classifier'.format(label, g), classes, sub['subcategory'], pred))
    return pd.DataFrame(rows)


def run_hierarchy(test_df, hierarchy, pred_group):
    """Route each image by the broad prediction, then classify with that
    group's model (a wrong route can only produce a wrong fine label).
    Single-class groups need no sub-model."""
    _, _, categories, _, prefix, _ = hierarchy
    pred_fine = np.empty(len(test_df), dtype=object)
    conf_fine = np.ones(len(test_df))
    for g in sorted(categories):
        idx = np.flatnonzero(pred_group == g)
        classes = sorted(categories[g])
        if not len(idx):
            continue
        if len(classes) == 1:
            pred_fine[idx] = classes[0]
        else:
            pred_fine[idx], conf_fine[idx] = predict(prefix + g, test_df['path'].iloc[idx], classes)
    return pred_fine.astype(str), conf_fine


def pct(x):
    return '{:.2f}\\%'.format(100 * x)


def dataset_table(dist, first_header):
    t_data = ['\\begin{tabular}{lrrrrr}', '\\toprule',
              '\\textbf{%s} & \\textbf{Fine classes}' % first_header + ' & \\textbf{Total images} & '
              '\\textbf{Training images} & \\textbf{Validation images} & \\textbf{Test images} \\\\', '\\midrule']
    for _, r in dist.iterrows():
        cells = [r['family']] + ['{:,}'.format(int(r[c])) for c in ('fine_classes', 'total', 'train', 'val', 'test')]
        if r['family'] == 'Total':
            t_data.append('\\midrule')
            cells = ['\\textbf{%s}' % c for c in cells]
        t_data.append(' & '.join(cells) + ' \\\\')
    t_data += ['\\bottomrule', '\\end{tabular}']
    return t_data


def write_tables(dist, dist_columns, models, summary):
    t_data = dataset_table(dist, 'Broad family')
    t_data_columns = dataset_table(dist_columns, 'Column category')

    t_cmp = ['\\begin{tabular}{lcccc}', '\\toprule',
             '\\textbf{Method} & \\textbf{Models} & \\textbf{34-class accuracy} & '
             '\\textbf{Macro-F1} & \\textbf{Weighted-F1} \\\\', '\\midrule']
    for key, label, n_models in METHODS:
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
    for i, (_, r) in enumerate(models.iterrows()):
        if i and r['model'].startswith('Broad'):
            t_models.append('\\midrule')
        t_models.append('{} & {} & {:,} & {} & {:.4f} & {} \\\\'.format(
            r['model'], r['classes'], r['test_images'], pct(r['accuracy']), r['macro_f1'],
            pct(r['majority_baseline'])))
    t_models += ['\\bottomrule', '\\end{tabular}']

    for name, lines in [('table_dataset.tex', t_data), ('table_dataset_columns.tex', t_data_columns),
                        ('table_comparison.tex', t_cmp),
                        ('table_models.tex', t_models)]:
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            f.write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df, test_df = load_split_dataframes()
    y = test_df['subcategory'].to_numpy()

    dist = dataset_distribution(train_df, val_df, test_df)
    dist.to_csv(os.path.join(OUT_DIR, 'dataset_distribution.csv'), index=False)
    dist_columns = dataset_distribution(train_df, val_df, test_df, 'column', COLUMN_CATEGORIES, 'Column')
    dist_columns.to_csv(os.path.join(OUT_DIR, 'dataset_distribution_columns.csv'), index=False)

    preds = {}
    preds['flat'] = predict('flat_34', test_df['path'], FINE_CLASSES)
    summary = {'test_images': len(test_df), 'flat': fine_metrics(y, preds['flat'][0])}
    model_tables = []
    for h in HIERARCHIES:
        name, group_col, categories, broad_model, _, _ = h
        pred_group, conf_group = predict(broad_model, test_df['path'], sorted(categories))
        preds[name] = run_hierarchy(test_df, h, pred_group)
        preds[name + '_group'] = (pred_group, conf_group)
        summary[name] = dict(fine_metrics(y, preds[name][0]),
                             routing_accuracy=float(accuracy_score(test_df[group_col], pred_group)))
        model_tables.append(per_model_results(test_df, h, pred_group))
    for a, b in [('flat', 'row'), ('flat', 'column'), ('row', 'column')]:
        summary['mcnemar_{}_vs_{}'.format(a, b)] = mcnemar(y, preds[a][0], preds[b][0])

    models = pd.concat(model_tables, ignore_index=True)
    models.to_csv(os.path.join(OUT_DIR, 'per_model_accuracy.csv'), index=False)
    print(models.to_string(index=False))
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    report = {'class': FINE_CLASSES}
    for key, _, _ in METHODS:
        p, r, f1, n = precision_recall_fscore_support(y, preds[key][0], labels=FINE_CLASSES, zero_division=0)
        report['support'] = n
        report.update({key + '_precision': p, key + '_recall': r, key + '_f1': f1})
        pd.DataFrame(confusion_matrix(y, preds[key][0], labels=FINE_CLASSES), index=FINE_CLASSES,
                     columns=FINE_CLASSES).to_csv(os.path.join(OUT_DIR, 'confusion_matrix_34_{}.csv'.format(key)))
    pd.DataFrame(report).to_csv(os.path.join(OUT_DIR, 'per_class_report.csv'), index=False)

    out = {'path': test_df['path'].map(lambda q: '/'.join(q.split(os.sep)[-2:])),
           'true_class': y, 'true_family': test_df['broad'], 'true_column': test_df['column']}
    for key in ('flat', 'row_group', 'row', 'column_group', 'column'):
        out[key + '_pred'], out[key + '_confidence'] = preds[key]
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, 'predictions.csv'), index=False)

    write_tables(dist, dist_columns, models, summary)
    print('\nWrote results to', OUT_DIR)


if __name__ == '__main__':
    main()
