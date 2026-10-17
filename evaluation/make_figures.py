"""Paper figures from results/experiments/ and models/experiments/:
confusion matrix heatmaps, per-submodel test accuracy / macro-F1 bars,
training/validation curves, and the A vs B vs C+D comparison figures.
Only draws figures whose input files already exist, so it's safe to re-run
as experiments finish.

Usage: python -m evaluation.make_figures
Output: results/experiments/figures/*.png and *.pdf (vector, for LaTeX)
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results', 'experiments')
MODELS_DIR = os.path.join(ROOT, 'models', 'experiments')
OUT_DIR = os.path.join(RESULTS_DIR, 'figures')

# Fixed method colours, used in every comparison figure.
METHOD_COLORS = {'A': '#2a78d6', 'B': '#eb6834', 'C+D': '#1baf7a', 'C': '#1baf7a'}
METHOD_LABELS = {'A': 'A: flat 34-class', 'B': 'B: row hierarchy', 'C+D': 'C+D: column hierarchy'}
INK = '#52514e'


def save(fig, out_name):
    base = os.path.splitext(out_name)[0]
    fig.savefig(os.path.join(OUT_DIR, base + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, base + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('wrote', base + '.png/.pdf')


def numeric_order(labels):
    return sorted(labels, key=lambda s: int(s))


def row_of(cls):
    p = cls.split('_')
    if p[0] != '7':
        return p[0]
    return {'1': '7', '2': '8', '3': '8', '4': '9', '5': '9', '6': '10'}[p[1]]


def confusion_matrix_heatmap(csv_path, title, out_name, figsize, row_separators=False):
    if not os.path.exists(csv_path):
        return
    cm = pd.read_csv(csv_path, index_col=0)
    cm.index = cm.index.astype(str)
    cm.columns = cm.columns.astype(str)
    if all(s.isdigit() for s in cm.index):
        order = numeric_order(cm.index)
        cm = cm.loc[order, order]
    cm_norm = cm.div(cm.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm.values, cmap='Blues', vmin=0, vmax=1)
    n = len(cm)
    small = n <= 12
    if small:
        for i in range(n):
            for j in range(n):
                v = cm_norm.values[i, j]
                if v >= 0.005:
                    ax.text(j, i, '{:.2f}'.format(v), ha='center', va='center', fontsize=7,
                            color='white' if v > 0.55 else '#0b0b0b')
    if row_separators:
        rows = [row_of(c) for c in cm.index]
        for k in range(1, n):
            if rows[k] != rows[k - 1]:
                ax.axhline(k - 0.5, color=INK, lw=0.6)
                ax.axvline(k - 0.5, color=INK, lw=0.6)
    fs = 9 if small else 7
    ax.set_xticks(range(n))
    ax.set_xticklabels(cm.columns, rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(cm.index, fontsize=fs)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Row-normalized fraction', shrink=0.8)
    fig.tight_layout()
    save(fig, out_name)


def _ci_err(values, ci_col):
    ci = ci_col.apply(json.loads).values
    lo = values - np.array([c[0] for c in ci])
    hi = np.array([c[1] for c in ci]) - values
    return [lo * 100, hi * 100]


def submodel_bars(csv_path, title, out_name, group_label):
    """Accuracy and macro-F1 per sub-model with 95% CIs. Macro-F1 matters here:
    a model that always predicts the majority class can still score high
    accuracy on an imbalanced category."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values('group', key=lambda s: s.astype(int)).reset_index(drop=True)
    acc = df['accuracy'].values
    f1 = df['macro_f1'].values
    baseline = df['majority_baseline'].values * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, acc * 100, w, yerr=_ci_err(acc, df['accuracy_ci95']), capsize=2,
           color='#2a78d6', label='Accuracy', error_kw={'lw': 0.8})
    ax.bar(x + w / 2, f1 * 100, w, yerr=_ci_err(f1, df['macro_f1_ci95']), capsize=2,
           color='#86b6ef', label='Macro-F1', error_kw={'lw': 0.8})
    ax.scatter(x - w / 2, baseline, color='#0b0b0b', marker='_', s=180, zorder=3,
               label='Majority baseline (accuracy)')
    labels = ['{} {}\n({} cl., n={})'.format(group_label, g, k, n)
              for g, k, n in zip(df['group'], df['n_classes'], df['n_images'])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Test score (%)')
    ax.set_ylim(0, 105)
    ax.set_title(title, pad=26)
    ax.grid(axis='y', color='#e5e5e2', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out_name)


def _read_history(history_csv):
    if not os.path.exists(history_csv) or os.path.getsize(history_csv) == 0:
        return None
    try:
        return pd.read_csv(history_csv)
    except pd.errors.ParserError:
        return None


def training_curves(history_csv, title, out_name):
    h = _read_history(history_csv)
    if h is None:
        return
    best = h['val_accuracy'].idxmax()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, name in [(axes[0], 'accuracy', 'Accuracy'), (axes[1], 'loss', 'Loss')]:
        ax.plot(h['epoch'], h[metric], color='#2a78d6', lw=2, label='train')
        ax.plot(h['epoch'], h['val_' + metric], color='#eb6834', lw=2, label='val')
        ax.axvline(h['epoch'][best], color=INK, ls=':', lw=1, label='best val acc (kept)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(color='#e5e5e2', lw=0.6)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, out_name)


def submodel_training_grid(exp_dir, title, out_name, group_label):
    """Small multiples: train/val accuracy of every sub-model of one experiment."""
    root = os.path.join(MODELS_DIR, exp_dir)
    if not os.path.isdir(root):
        return
    subs = sorted([d for d in os.listdir(root) if d.startswith('sub_')], key=lambda d: int(d[4:]))
    hists = [(d, _read_history(os.path.join(root, d, d + '_history.csv'))) for d in subs]
    hists = [(d, h) for d, h in hists if h is not None]
    if not hists:
        return
    cols = 3
    rows = int(np.ceil(len(hists) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows), sharey=True, squeeze=False)
    for ax, (d, h) in zip(axes.flat, hists):
        ax.plot(h['epoch'], h['accuracy'], color='#2a78d6', lw=1.6, label='train')
        ax.plot(h['epoch'], h['val_accuracy'], color='#eb6834', lw=1.6, label='val')
        ax.axvline(h['epoch'][h['val_accuracy'].idxmax()], color=INK, ls=':', lw=1)
        ax.set_title('{} {}'.format(group_label, d[4:]), fontsize=9)
        ax.grid(color='#e5e5e2', lw=0.6)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(hists):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('Accuracy', fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel('Epoch', fontsize=9)
    axes.flat[0].legend(fontsize=7, loc='lower right')
    fig.suptitle(title + '  (dotted: best val accuracy, the checkpoint kept)')
    fig.tight_layout()
    save(fig, out_name)


def _load_summary():
    path = os.path.join(RESULTS_DIR, 'comparison', 'summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def comparison_34(summary):
    """34-class accuracy and macro-F1 of A, B, C+D with 95% CIs."""
    entries = [('A', summary.get('A', {}).get('fine_34')),
               ('B', summary.get('B', {}).get('fine_34')),
               ('C+D', summary.get('D', {}).get('fine_34_with_C_routing'))]
    entries = [(k, e) for k, e in entries if e]
    if len(entries) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, name in [(axes[0], 'accuracy', '34-class accuracy (%)'),
                             (axes[1], 'macro_f1', '34-class macro-F1 (%)')]:
        for i, (k, e) in enumerate(entries):
            v = e[metric] * 100
            lo, hi = (c * 100 for c in e[metric + '_ci95'])
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt='o', ms=8, color=METHOD_COLORS[k],
                        capsize=4, lw=2)
            ax.annotate('{:.1f}'.format(v), (v, i), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=8, color='#0b0b0b')
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([METHOD_LABELS[k] for k, _ in entries])
        ax.set_ylim(-0.6, len(entries) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel(name)
        ax.grid(axis='x', color='#e5e5e2', lw=0.6)
    axes[1].set_yticklabels([])
    mc = summary.get('mcnemar', {})
    note = '; '.join('{} p={:.2g}'.format(k.replace('_vs_', ' vs '), v['p_value']) for k, v in mc.items())
    fig.suptitle('Full 34-class test set (n={}), 95% bootstrap CIs'.format(entries[0][1]['n_images']))
    if note:
        fig.text(0.5, -0.02, 'McNemar: ' + note, ha='center', fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, 'comparison_34class.png')


def comparison_broad(summary):
    """Broad level: B rows (10) vs C columns (9) with majority baselines."""
    b = summary.get('B', {}).get('broad_rows_10')
    c = summary.get('C', {}).get('broad_columns_9')
    if not (b and c):
        return
    entries = [('B', 'B: 10 row categories', b), ('C', 'C: 9 column categories', c)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(2)
    w = 0.36
    for i, (k, label, e) in enumerate(entries):
        for j, (metric, alpha, mname) in enumerate([('accuracy', 1.0, 'Accuracy'), ('macro_f1', 0.45, 'Macro-F1')]):
            v = e[metric] * 100
            lo, hi = (cv * 100 for cv in e[metric + '_ci95'])
            xpos = i + (j - 0.5) * w
            ax.bar(xpos, v, w, color=METHOD_COLORS[k], alpha=alpha, yerr=[[v - lo], [hi - v]], capsize=3,
                   error_kw={'lw': 0.8})
            ax.text(xpos, 3, mname, ha='center', va='bottom', rotation=90, fontsize=8,
                    color='white' if alpha == 1.0 else '#0b0b0b')
            ax.text(xpos, hi + 1.5, '{:.1f}'.format(v), ha='center', fontsize=8)
        ax.plot([i - w, i], [e['majority_baseline'] * 100] * 2, color='#0b0b0b', lw=2)
    ax.plot([], [], color='#0b0b0b', lw=2, label='Majority baseline (accuracy)')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in entries])
    ax.set_ylabel('Test score (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Broad (first-stage) classifiers, 95% bootstrap CIs')
    ax.grid(axis='y', color='#e5e5e2', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    save(fig, 'comparison_broad_level.png')


def per_class_f1_comparison():
    """Per-class F1 of A, B, C+D for all 34 classes, with test support."""
    paths = [('A', os.path.join(RESULTS_DIR, 'A_flat_34', 'per_class_report.csv')),
             ('B', os.path.join(RESULTS_DIR, 'B_rows_10', 'per_class_report.csv')),
             ('C+D', os.path.join(RESULTS_DIR, 'D_column_subcategories', 'per_class_report.csv'))]
    reps = [(k, pd.read_csv(p).set_index('class')) for k, p in paths if os.path.exists(p)]
    if len(reps) < 2:
        return
    classes = list(reps[0][1].index)
    support = reps[0][1]['support']
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(classes))
    offsets = np.linspace(-0.22, 0.22, len(reps))
    for (k, r), off in zip(reps, offsets):
        ax.scatter(x + off, r.loc[classes, 'f1'], s=26, color=METHOD_COLORS[k], label=METHOD_LABELS[k],
                   edgecolors='#fcfcfb', linewidths=0.8, zorder=3)
    rows = [row_of(c) for c in classes]
    for i in range(1, len(classes)):
        if rows[i] != rows[i - 1]:
            ax.axvline(i - 0.5, color='#d0d0cc', lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['{}\n{}'.format(c, int(support[c])) for c in classes], rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Test F1')
    ax.set_xlabel('Class (test support); vertical lines separate HamNoSys row categories')
    ax.set_title('Per-class F1 on the 34-class test set', pad=26)
    ax.grid(axis='y', color='#e5e5e2', lw=0.6)
    ax.legend(fontsize=8, ncol=len(reps), loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, 'per_class_f1_comparison.png')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'A_flat_34', 'confusion_matrix_34.csv'),
        'Experiment A: flat 34-class model', 'confusion_matrix_A_34.png', (11, 9), row_separators=True)
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'B_rows_10', 'confusion_matrix_rows_10.csv'),
        'Experiment B: broad row classifier (10 categories)', 'confusion_matrix_B_10.png', (7, 6))
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'B_rows_10', 'confusion_matrix_34.csv'),
        'Experiment B: full 34-class hierarchy (row routing)', 'confusion_matrix_B_34.png', (11, 9),
        row_separators=True)
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'C_columns_9', 'confusion_matrix_columns_9.csv'),
        'Experiment C: broad column classifier (9 categories)', 'confusion_matrix_C.png', (7, 6))
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'D_column_subcategories', 'confusion_matrix_34.csv'),
        'Experiment D: full 34-class hierarchy (C→D routing)', 'confusion_matrix_D_34.png', (11, 9),
        row_separators=True)

    submodel_bars(
        os.path.join(RESULTS_DIR, 'B_rows_10', 'sub_models.csv'),
        'Experiment B: test scores per row sub-model', 'accuracy_B_submodels.png', 'Row')
    submodel_bars(
        os.path.join(RESULTS_DIR, 'D_column_subcategories', 'sub_models.csv'),
        'Experiment D: test scores per column sub-model', 'accuracy_D_submodels.png', 'Column')

    training_curves(
        os.path.join(MODELS_DIR, 'A_flat_34', 'flat_34', 'flat_34_history.csv'),
        'Experiment A: flat 34-class model training curves', 'training_curves_A_flat34.png')
    training_curves(
        os.path.join(MODELS_DIR, 'B_rows_10', 'broad', 'broad_history.csv'),
        'Experiment B: broad row classifier training curves', 'training_curves_B_broad.png')
    training_curves(
        os.path.join(MODELS_DIR, 'C_columns_9', 'broad', 'broad_history.csv'),
        'Experiment C: broad column classifier training curves', 'training_curves_C_broad.png')
    submodel_training_grid('B_rows_10', 'Experiment B row sub-models', 'training_curves_B_submodels.png', 'Row')
    submodel_training_grid('D_column_subcategories', 'Experiment D column sub-models',
                           'training_curves_D_submodels.png', 'Column')

    summary = _load_summary()
    if summary:
        comparison_34(summary)
        comparison_broad(summary)
    per_class_f1_comparison()

    print('done, figures in', OUT_DIR)


if __name__ == '__main__':
    main()
