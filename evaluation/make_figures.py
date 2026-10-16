"""Paper figures from results/experiments/: confusion matrix heatmaps,
per-submodel test accuracy bar charts, and training/validation curves.
Only draws figures for experiments whose result files already exist, so
it's safe to re-run as A and B finish.

Usage: python -m evaluation.make_figures
Output: results/experiments/figures/*.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results', 'experiments')
MODELS_DIR = os.path.join(ROOT, 'models', 'experiments')
OUT_DIR = os.path.join(RESULTS_DIR, 'figures')


def confusion_matrix_heatmap(csv_path, title, out_name, figsize):
    if not os.path.exists(csv_path):
        return
    cm = pd.read_csv(csv_path, index_col=0)
    cm_norm = cm.div(cm.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm.values, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Row-normalized fraction', shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=200)
    plt.close(fig)
    print('wrote', out_name)


def submodel_accuracy_bars(csv_path, title, out_name, group_label='Column category'):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    acc = df['accuracy'].values * 100
    ci = df['accuracy_ci95'].apply(eval).values
    lo = acc - np.array([c[0] for c in ci]) * 100
    hi = np.array([c[1] for c in ci]) * 100 - acc
    baseline = df['majority_baseline'].values * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.bar(x, acc, yerr=[lo, hi], capsize=3, color='#4C72B0', label='Test accuracy')
    ax.scatter(x, baseline, color='#C44E52', marker='_', s=200, zorder=3, label='Majority baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(['{} {}'.format(group_label, g) for g in df['group']], rotation=30, ha='right')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=200)
    plt.close(fig)
    print('wrote', out_name)


def training_curves(history_csv, title, out_name):
    if not os.path.exists(history_csv) or os.path.getsize(history_csv) == 0:
        return
    try:
        h = pd.read_csv(history_csv)
    except pd.errors.ParserError:
        print('skip {}: still being written'.format(out_name))
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h['epoch'], h['accuracy'], label='train')
    axes[0].plot(h['epoch'], h['val_accuracy'], label='val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(h['epoch'], h['loss'], label='train')
    axes[1].plot(h['epoch'], h['val_loss'], label='val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=200)
    plt.close(fig)
    print('wrote', out_name)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'C_columns_9', 'confusion_matrix_columns_9.csv'),
        'Experiment C: broad column classifier (9 categories)', 'confusion_matrix_C.png', (6, 5))
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'D_column_subcategories', 'confusion_matrix_34.csv'),
        'Experiment D: full 34-class hierarchy (C→D routing)', 'confusion_matrix_D_34.png', (11, 9))
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'A_flat_34', 'confusion_matrix_34.csv'),
        'Experiment A: flat 34-class model', 'confusion_matrix_A_34.png', (11, 9))
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'B_rows_10', 'confusion_matrix_10.csv'),
        'Experiment B: broad row classifier (10 categories)', 'confusion_matrix_B_10.png', (6, 5))
    confusion_matrix_heatmap(
        os.path.join(RESULTS_DIR, 'B_rows_10', 'confusion_matrix_34.csv'),
        'Experiment B: full 34-class hierarchy (row routing)', 'confusion_matrix_B_34.png', (11, 9))

    submodel_accuracy_bars(
        os.path.join(RESULTS_DIR, 'D_column_subcategories', 'sub_models.csv'),
        'Experiment D: test accuracy per column sub-model', 'accuracy_D_submodels.png')
    submodel_accuracy_bars(
        os.path.join(RESULTS_DIR, 'B_rows_10', 'sub_models.csv'),
        'Experiment B: test accuracy per row sub-model', 'accuracy_B_submodels.png', group_label='Row category')

    training_curves(
        os.path.join(MODELS_DIR, 'C_columns_9', 'broad', 'broad_history.csv'),
        'Experiment C: broad classifier training curves', 'training_curves_C_broad.png')
    training_curves(
        os.path.join(MODELS_DIR, 'A_flat_34', 'flat_34', 'flat_34_history.csv'),
        'Experiment A: flat 34-class model training curves', 'training_curves_A_flat34.png')

    print('done, figures in', OUT_DIR)


if __name__ == '__main__':
    main()
