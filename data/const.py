import numpy as np
import os

from os import path
from datetime import datetime
from sys import platform
import errno

#
# | Base model | resolution|
# |----------------|-----|
# | EfficientNetB0 | 224 |
# | EfficientNetB1 | 240 |
# | EfficientNetB2 | 260 |
# | EfficientNetB3 | 300 |
# | EfficientNetB4 | 380 |
# | EfficientNetB5 | 456 |
# | EfficientNetB6 | 528 |
# | EfficientNetB7 | 600 |

NUM_CLASSES_TRAIN = 3
NUM_CLASSES_TEST = 3
IMG_SIZE = 380  # EfficientNetB4 native resolution
BATCH_SIZE = 32  # matches the recipe already proven on this dataset/GPU in old/train_hierarchical.py
SAVE_PERIOD = 1
LEARNING_RATE = 1e-3  # stage 1: linear probe, frozen backbone, head only ("Adam default from paper")
UNFREEZE_LEARNING_RATE = 1e-4
N_EPOCHS = 50  # paper: up to 50 epochs (stage 1 budget)

# Stage 2 (fine-tuning): unfreeze the top blocks of an already linear-probed
# model and continue training at a much lower LR. Matches the recipe already
# proven on this dataset in old/finetune_hierarchical.py.
FT_LEARNING_RATE = 1e-5
FT_EPOCHS = 30
FT_PATIENCE = 5
N_WORKERS=0 #0
TOP_DROPOUT_RATE = 0.2  # paper
LAST_TEACHER_MODEL_NAME='11_f333_b5_sgdDlrNE'
MODEL_NAME='11_f333_b5_sgdDlrNE'
HISTORY_NAME= MODEL_NAME+'HISTORY'
WEIGHTS = "imagenet"  # tf.keras.applications.EfficientNetB4 only supports 'imagenet' or None
PATIENCE = 5  # paper: early stopping patience 5
N_EPOCHS_UNFREEZE=100
ITERATION='1'
STUDENT_ANNOTATIONS='student_train_drop_if<0.7.csv'
NEW_DATASET_ANNOTATIONS='dataset_labels.csv'#'new_train2.csv'

# --- Teacher model training (paper params + reasonable defaults for unreported values) ---

# Which top blocks of EfficientNetB4 to unfreeze for fine-tuning. B4 has 7 top-level
# blocks (block1..block7); unfreezing the last two plus the final "top" conv/bn is a
# common, conservative choice that lets the model adapt high-level features to
# handshapes without disturbing low-level filters or destabilizing BatchNorm
# statistics on a relatively small dataset.
UNFREEZE_BLOCK_PREFIXES = ('block6', 'block7', 'top_')

# Confidence above which a pseudo-label from a teacher model is trusted for
# noisy-student style self-training. 0.7 mirrors the threshold already used
# elsewhere in this repo for filtering low-confidence student annotations.
PSEUDO_LABEL_CONFIDENCE_THRESHOLD = 0.7

# Dataset split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Root directory containing one flat folder of images per fine-grained
# subcategory (e.g. dataset/1_1/, dataset/7_6_4/, ...). Overridable via env
# var so the same code runs unchanged locally and on Kaggle/Colab, where the
# dataset is mounted under a different path (e.g. /kaggle/input/<name>).
TEACHER_DATASET_DIR = os.environ.get(
    'TEACHER_DATASET_DIR',
    os.path.join(path.expanduser('~'), 'Desktop', 'dataset')
)

# Shared leakage-free train/val/test assignment for all 8 models (frames of
# one source video, identical files and near-duplicates kept in one split).
# Built by data/make_grouped_split.py, verified by data/check_split.py.
SPLIT_FILE = os.environ.get(
    'SPLIT_FILE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits', 'grouped_split.csv')
)

# Broad category -> list of subcategory folder names used to train it.
# 1_5 and 7_6_2 are excluded: too few images (4 and 19 respectively) to be
# usable in an 80/10/10 stratified split.
BROAD_CATEGORIES = {
    '1': ['1_1', '1_3'],
    '2': ['2_1', '2_2', '2_3', '2_5'],
    '3': ['3_1', '3_2', '3_3', '3_5'],
    '4': ['4_1', '4_2', '4_3', '4_5'],
    '5': ['5_1', '5_2', '5_3', '5_4', '5_5'],
    '6': ['6_1', '6_2', '6_3', '6_5'],
    '7': ['7_1_1', '7_1_2', '7_1_3', '7_1_4', '7_2_2',
          '7_4_1', '7_4_2', '7_4_3', '7_6_1', '7_6_3', '7_6_4'],
}

# Repo root, computed dynamically so this doesn't break on machines other
# than the original authors' (previously hardcoded to '/home/kenny/...' and
# '/Users/madina/...').
SYS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if platform == "linux" or platform == "linux2":
    TRAIN_PATH = '/media/kenny/Extra/downloads/1mil/train_by_category/'#train_by_classes/' #nds-v2-training/'
    TEST_PATH = '/media/kenny/Extra/downloads/1mil/ph2014-dev-set-handshape-annotations/'
    LABELS_PATH = os.path.join(SYS_PATH, 'train', 'results', 'labeling')
elif platform == "darwin":
    TRAIN_PATH = '/Users/madina/Desktop/dataset/1miohands-v2-training/nds-v2-training/'
    TEST_PATH = '/Users/madina/Desktop/dataset/1miohands-v2-training/ph2014-dev-set-handshape-annotations/'
    LABELS_PATH = os.path.join(SYS_PATH, 'train', 'results', 'labeling')
elif platform == "win32":
    TRAIN_PATH = ''
    TEST_PATH = ''
    LABELS_PATH = os.path.join(SYS_PATH, 'train', 'results', 'labeling')

CLASSES = [str(x) for x in np.arange(1, NUM_CLASSES_TRAIN, 1).tolist()]

SAVE_DIR = os.path.join(SYS_PATH or '.', 'models', 'categories', '1', '1_1', MODEL_NAME)
RES_DIR = os.path.join(SYS_PATH or '.', 'results', 'categories', '1', '1_1', MODEL_NAME)
LABELING_DIR = "results/labeling/"
LOG_DIR = "logs/scalars/"
WORK_DIR = "model_architecture"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME + "_model.hdf5")#"_epoch-{epoch:02d}_val_loss-{val_loss:.2f}_val_acc-{val_accuracy:.2f}.hdf5")
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME + "_epoch-{epoch:02d}_val_loss-{val_loss:.2f}_val_acc-{val_accuracy:.2f}.hdf5")
MODEL_CSV_HIST_PATH = os.path.join(RES_DIR, HISTORY_NAME + "_log.csv")
HIST_PATH = os.path.join(RES_DIR, HISTORY_NAME + ".kerashist")
HIST_PLOT_PATH = os.path.join(RES_DIR, HISTORY_NAME)
LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
PLOT_PATH = os.path.join(WORK_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))