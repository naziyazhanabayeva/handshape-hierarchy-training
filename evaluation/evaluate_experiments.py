"""Evaluate experiments A-D (data/experiments.py) on the shared leakage-free
test split and write every table/number for the paper, one folder per
experiment:

results/experiments/
  dataset/                 data split tables (whole dataset, per class, per B
                           row category, per C column category)
  A_flat_34/               single 34-class model
  B_rows_10/               broad model over 10 rows + row sub-models
  C_columns_9/             broad model over 9 columns
  D_column_subcategories/  column sub-models; full hierarchy C -> D
  comparison/              A vs B vs C->D on the same test images

Each experiment folder has metrics.json, a LaTeX table, a per-class report,
a confusion matrix and per-image predictions. 34-class metrics: accuracy,
macro-F1, weighted-F1 with 95% bootstrap confidence intervals. Sub-model
tables use the test images of the model's true group ("oracle routing");
the full-hierarchy numbers use the broad model's routing. comparison/ adds
pairwise exact McNemar tests on per-image correctness.

Usage (from classify_handshapes/, after train.train_experiment):
    python -m evaluation.evaluate_experiments
Experiments whose models are missing are skipped.
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

from data.const import BATCH_SIZE, IMG_SIZE
from data.experiments import (
    COLUMN_GROUPS, EXPERIMENTS, FINE_CLASSES, ROW_GROUPS, class_mapping_rows, column_category, row_category,
)
from data.teacher_dataset import load_split_dataframes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get('EXPERIMENT_MODELS_DIR', os.path.join(ROOT, 'models', 'experiments'))
OUT_DIR = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results', 'experiments'))
N_BOOTSTRAP = 1000


# ---------------------------------------------------------------- prediction

def _load_image(path):
    # Identical to data/teacher_dataset._load_image: raw [0, 255] pixels,
    # EfficientNetB4 rescales internally (checked by data/check_pipeline.py).
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return tf.cast(image, tf.float32)


def model_path(key, model_name):
    return os.path.join(MODELS_DIR, EXPERIMENTS[key]['dir'], model_name, model_name + '_best.keras')


def predict_probs(key, model_name, paths):
    """Class probabilities of one model on all test images (cached)."""
    cache = os.path.join(OUT_DIR, 'cache', '{}_{}.npy'.format(key, model_name))
    if os.path.exists(cache):
        return np.load(cache)
    print('  predicting with', model_path(key, model_name), flush=True)
    model = load_model(model_path(key, model_name))
    call = tf.function(lambda x: model(x, training=False))
    ds = tf.data.Dataset.from_tensor_slices(list(paths)).map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    probs = np.concatenate([call(b).numpy() for b in ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, probs)
    return probs


def available(key):
    return all(os.path.exists(model_path(key, m)) for m, _, _ in EXPERIMENTS[key]['models'])


# ---------------------------------------------------------------- metrics

def bootstrap_ci(y_true, y_pred, metric, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def classification_metrics(y_true, y_pred, labels):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fns = {
        'accuracy': accuracy_score,
        'macro_f1': lambda a, b: f1_score(a, b, labels=labels, average='macro', zero_division=0),
        'weighted_f1': lambda a, b: f1_score(a, b, labels=labels, average='weighted', zero_division=0),
    }
    out = {'n_images': int(len(y_true)), 'n_classes': len(labels),
           'majority_baseline': float(pd.Series(y_true).value_counts(normalize=True).iloc[0])}
    for name, fn in fns.items():
        out[name] = float(fn(y_true, y_pred))
        out[name + '_ci95'] = bootstrap_ci(y_true, y_pred, fn)
    return out


def mcnemar(y_true, pred_a, pred_b):
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {'only_first_correct': only_a, 'only_second_correct': only_b, 'p_value': float(p)}


# ---------------------------------------------------------------- writing

def pct(x):
    return '{:.2f}\\%'.format(100 * x)


def ci_pct(m, key):
    lo, hi = m[key + '_ci95']
    return '{} [{}, {}]'.format(pct(m[key]), pct(lo), pct(hi))


def ci_f1(m, key):
    lo, hi = m[key + '_ci95']
    return '{:.4f} [{:.4f}, {:.4f}]'.format(m[key], lo, hi)


def write_tex(path, header, rows, align):
    lines = ['\\begin{tabular}{%s}' % align, '\\toprule', ' & '.join('\\textbf{%s}' % h for h in header) + ' \\\\',
             '\\midrule']
    for r in rows:
        lines.append('\\midrule' if r == 'midrule' else ' & '.join(str(c) for c in r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_fine_outputs(folder, test_df, pred, confidence, extra_columns=None):
    """34-class per-class report, confusion matrix and predictions."""
    y = test_df['subcategory'].to_numpy()
    p, r, f1, n = precision_recall_fscore_support(y, pred, labels=FINE_CLASSES, zero_division=0)
    pd.DataFrame({'class': FINE_CLASSES, 'support': n, 'precision': p, 'recall': r, 'f1': f1}).to_csv(
        os.path.join(folder, 'per_class_report.csv'), index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=FINE_CLASSES), index=FINE_CLASSES,
                 columns=FINE_CLASSES).to_csv(os.path.join(folder, 'confusion_matrix_34.csv'))
    out = {'image': test_df['key'], 'true_class': y, 'pred_class': pred, 'confidence': confidence}
    out.update(extra_columns or {})
    pd.DataFrame(out).to_csv(os.path.join(folder, 'predictions.csv'), index=False)


def experiment_folder(key):
    folder = os.path.join(OUT_DIR, EXPERIMENTS[key]['dir'])
    os.makedirs(folder, exist_ok=True)
    return folder


# ---------------------------------------------------------------- dataset

def split_counts(dfs, group_of):
    rows = {}
    for name, df in zip(('train', 'val', 'test'), dfs):
        for g, n in df['subcategory'].map(group_of).value_counts().items():
            rows.setdefault(g, {'train': 0, 'val': 0, 'test': 0})[name] = int(n)
    return rows


def write_dataset_tables(dfs):
    folder = os.path.join(OUT_DIR, 'dataset')
    os.makedirs(folder, exist_ok=True)
    mapping = pd.DataFrame(class_mapping_rows())
    counts = split_counts(dfs, lambda c: c)
    per_class = mapping.assign(train=[counts[c]['train'] for c in mapping['class']],
                               val=[counts[c]['val'] for c in mapping['class']],
                               test=[counts[c]['test'] for c in mapping['class']])
    per_class['total'] = per_class[['train', 'val', 'test']].sum(axis=1)
    per_class.to_csv(os.path.join(folder, 'per_class_split.csv'), index=False)
    write_tex(os.path.join(folder, 'table_per_class.tex'),
              ['Class', 'HamNoSys row', 'HamNoSys column', 'B', 'C', 'Train', 'Val', 'Test'],
              [[r['class'].replace('_', '\\_'), r['hamnosys_row'], r['hamnosys_column'], r['B_row_category'],
                r['C_column_category'], '{:,}'.format(r['train']), '{:,}'.format(r['val']), '{:,}'.format(r['test'])]
               for _, r in per_class.iterrows()], 'lllccrrr')

    for name, groups, group_of, label in [('B_rows', ROW_GROUPS, row_category, 'Row category'),
                                          ('C_columns', COLUMN_GROUPS, column_category, 'Column category')]:
        counts = split_counts(dfs, group_of)
        rows, totals = [], np.zeros(4, dtype=int)
        for g in sorted(groups, key=int):
            n = [counts[g]['train'], counts[g]['val'], counts[g]['test']]
            totals += [len(groups[g])] + n
            rows.append([g, len(groups[g]), ', '.join(groups[g]).replace('_', '\\_'),
                         '{:,}'.format(sum(n))] + ['{:,}'.format(x) for x in n])
        rows += ['midrule', ['\\textbf{Total}', '\\textbf{%d}' % totals[0], '', '\\textbf{{{:,}}}'.format(totals[1:].sum())]
                 + ['\\textbf{{{:,}}}'.format(x) for x in totals[1:]]]
        write_tex(os.path.join(folder, 'table_{}.tex'.format(name)),
                  [label, 'Classes', 'Fine classes', 'Total', 'Train', 'Val', 'Test'], rows, 'cclrrrr')


# ---------------------------------------------------------------- experiments

def evaluate_flat(test_df):
    """A: single 34-class model."""
    probs = predict_probs('A', 'flat_34', test_df['path'])
    pred, conf = np.array(FINE_CLASSES)[probs.argmax(1)], probs.max(1)
    metrics = {'fine_34': classification_metrics(test_df['subcategory'], pred, FINE_CLASSES)}
    folder = experiment_folder('A')
    write_fine_outputs(folder, test_df, pred, conf)
    m = metrics['fine_34']
    write_tex(os.path.join(folder, 'table_A.tex'), ['Model', 'Classes', 'Test images', 'Accuracy', 'Macro-F1'],
              [['Single EfficientNetB4', 34, '{:,}'.format(m['n_images']), ci_pct(m, 'accuracy'),
                ci_f1(m, 'macro_f1')]], 'lcccc')
    return metrics, pred


def evaluate_hierarchy(test_df, broad_key, sub_key, groups, group_of):
    """Broad model of `broad_key` routes every test image to a group; the
    group's sub-model of `sub_key` predicts the fine class. Returns broad
    metrics, per-sub-model metrics (oracle routing) and full 34-class
    metrics."""
    group_names = sorted(groups)  # training label order (sorted strings)
    true_group = test_df['subcategory'].map(group_of).to_numpy()
    broad_probs = predict_probs(broad_key, 'broad', test_df['path'])
    pred_group = np.array(group_names)[broad_probs.argmax(1)]
    broad = classification_metrics(true_group, pred_group, group_names)
    broad_cm = pd.DataFrame(confusion_matrix(true_group, pred_group, labels=group_names),
                            index=group_names, columns=group_names)

    pred_fine = np.empty(len(test_df), dtype=object)
    conf_fine = np.ones(len(test_df))
    sub_rows = []
    for g in sorted(groups, key=int):
        classes = sorted(groups[g])
        routed = pred_group == g
        if len(classes) == 1:
            pred_fine[routed] = classes[0]
            continue
        probs = predict_probs(sub_key, 'sub_' + g, test_df['path'])
        sub_pred = np.array(classes)[probs.argmax(1)]
        pred_fine[routed], conf_fine[routed] = sub_pred[routed], probs.max(1)[routed]
        own = true_group == g  # oracle routing: the sub-model on its own group's images
        m = classification_metrics(test_df['subcategory'].to_numpy()[own], sub_pred[own], classes)
        sub_rows.append(dict(group=g, classes=', '.join(classes), **{k: v for k, v in m.items()}))
    pred_fine = pred_fine.astype(str)
    fine = classification_metrics(test_df['subcategory'], pred_fine, FINE_CLASSES)
    return broad, broad_cm, pd.DataFrame(sub_rows), fine, pred_group, broad_probs.max(1), pred_fine, conf_fine


def write_hierarchy_tables(folder, prefix, broad, sub_df, fine, group_label):
    rows = [['Broad classifier', broad['n_classes'], '{:,}'.format(broad['n_images']), ci_pct(broad, 'accuracy'),
             '{:.4f}'.format(broad['macro_f1']), pct(broad['majority_baseline'])], 'midrule']
    for _, r in sub_df.iterrows():
        rows.append(['{} {} classifier'.format(group_label, r['group']), r['n_classes'],
                     '{:,}'.format(r['n_images']), ci_pct(r, 'accuracy'), '{:.4f}'.format(r['macro_f1']),
                     pct(r['majority_baseline'])])
    rows += ['midrule', ['Full hierarchy (34 classes)', 34, '{:,}'.format(fine['n_images']),
                         ci_pct(fine, 'accuracy'), ci_f1(fine, 'macro_f1'), pct(fine['majority_baseline'])]]
    write_tex(os.path.join(folder, 'table_{}.tex'.format(prefix)),
              ['Model', 'Classes', 'Test images', 'Accuracy', 'Macro-F1', 'Majority baseline'], rows, 'lccccc')
    sub_df.to_csv(os.path.join(folder, 'sub_models.csv'), index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = load_split_dataframes()
    write_dataset_tables(dfs)
    test_df = dfs[2]
    test_df['key'] = test_df['path'].map(lambda q: '/'.join(q.split(os.sep)[-2:]))
    y = test_df['subcategory'].to_numpy()
    summary, preds = {}, {}

    if available('A'):
        summary['A'], preds['A'] = evaluate_flat(test_df)
    else:
        print('[skip] A: models missing')

    if available('B'):
        broad, cm, subs, fine, pg, pc, pf, cf = evaluate_hierarchy(test_df, 'B', 'B', ROW_GROUPS, row_category)
        folder = experiment_folder('B')
        cm.to_csv(os.path.join(folder, 'confusion_matrix_rows_10.csv'))
        write_hierarchy_tables(folder, 'B', broad, subs, fine, 'Row')
        write_fine_outputs(folder, test_df, pf, cf, {'pred_row': pg, 'row_confidence': pc,
                                                     'true_row': test_df['subcategory'].map(row_category)})
        summary['B'] = {'broad_rows_10': broad, 'fine_34': fine}
        preds['B'] = pf
    else:
        print('[skip] B: models missing')

    if available('C'):
        group_names = sorted(COLUMN_GROUPS)
        true_col = test_df['subcategory'].map(column_category).to_numpy()
        probs = predict_probs('C', 'broad', test_df['path'])
        pred_col = np.array(group_names)[probs.argmax(1)]
        broad = classification_metrics(true_col, pred_col, group_names)
        folder = experiment_folder('C')
        pd.DataFrame(confusion_matrix(true_col, pred_col, labels=group_names), index=group_names,
                     columns=group_names).to_csv(os.path.join(folder, 'confusion_matrix_columns_9.csv'))
        pd.DataFrame({'image': test_df['key'], 'true_column': true_col, 'pred_column': pred_col,
                      'confidence': probs.max(1)}).to_csv(os.path.join(folder, 'predictions.csv'), index=False)
        write_tex(os.path.join(folder, 'table_C.tex'),
                  ['Model', 'Classes', 'Test images', 'Accuracy', 'Macro-F1', 'Majority baseline'],
                  [['Broad classifier (HamNoSys columns)', 9, '{:,}'.format(broad['n_images']),
                    ci_pct(broad, 'accuracy'), ci_f1(broad, 'macro_f1'), pct(broad['majority_baseline'])]], 'lccccc')
        summary['C'] = {'broad_columns_9': broad}
    else:
        print('[skip] C: models missing')

    if available('C') and available('D'):
        broad, cm, subs, fine, pg, pc, pf, cf = evaluate_hierarchy(test_df, 'C', 'D', COLUMN_GROUPS, column_category)
        folder = experiment_folder('D')
        write_hierarchy_tables(folder, 'D', broad, subs, fine, 'Column category')
        write_fine_outputs(folder, test_df, pf, cf, {'pred_column': pg, 'column_confidence': pc,
                                                     'true_column': test_df['subcategory'].map(column_category)})
        summary['D'] = {'fine_34_with_C_routing': fine}
        preds['C+D'] = pf
    else:
        print('[skip] D: needs the models of C and D')

    folder = os.path.join(OUT_DIR, 'comparison')
    os.makedirs(folder, exist_ok=True)
    labels = {'A': ('A: single 34-class model', 1),
              'B': ('B: hierarchy by HamNoSys row (10 categories)', len(EXPERIMENTS['B']['models'])),
              'C+D': ('C+D: hierarchy by HamNoSys column (9 categories)',
                      len(EXPERIMENTS['C']['models']) + len(EXPERIMENTS['D']['models']))}
    fine_of = {'A': lambda: summary['A']['fine_34'], 'B': lambda: summary['B']['fine_34'],
               'C+D': lambda: summary['D']['fine_34_with_C_routing']}
    rows = []
    for k in ('A', 'B', 'C+D'):
        if k in preds:
            m = fine_of[k]()
            rows.append([labels[k][0], labels[k][1], ci_pct(m, 'accuracy'), ci_f1(m, 'macro_f1'),
                         '{:.4f}'.format(m['weighted_f1'])])
    if rows:
        write_tex(os.path.join(folder, 'table_comparison.tex'),
                  ['Method', 'Models', '34-class accuracy', 'Macro-F1', 'Weighted-F1'], rows, 'lcccc')
    broad_rows = []
    if 'B' in summary:
        b = summary['B']['broad_rows_10']
        broad_rows.append(['B: HamNoSys rows', 10, ci_pct(b, 'accuracy'), ci_f1(b, 'macro_f1'), pct(b['majority_baseline'])])
    if 'C' in summary:
        c = summary['C']['broad_columns_9']
        broad_rows.append(['C: HamNoSys columns', 9, ci_pct(c, 'accuracy'), ci_f1(c, 'macro_f1'), pct(c['majority_baseline'])])
    if broad_rows:
        write_tex(os.path.join(folder, 'table_broad_level.tex'),
                  ['Grouping', 'Categories', 'Accuracy', 'Macro-F1', 'Majority baseline'], broad_rows, 'lcccc')
    keys = [k for k in ('A', 'B', 'C+D') if k in preds]
    summary['mcnemar'] = {'{}_vs_{}'.format(a, b): mcnemar(y, preds[a], preds[b])
                          for i, a in enumerate(keys) for b in keys[i + 1:]}
    with open(os.path.join(folder, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print('\nWrote results to', OUT_DIR)


if __name__ == '__main__':
    main()
