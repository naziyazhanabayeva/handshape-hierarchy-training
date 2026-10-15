"""Data loading for the 8 teacher models (1 broad-category + 7 subcategory models).

Expects TEACHER_DATASET_DIR to contain one flat folder of images per
fine-grained subcategory, e.g.:

    dataset/1_1/*.png
    dataset/1_3/*.png
    dataset/2_1/*.png
    ...

BROAD_CATEGORIES (data/const.py) maps each broad category id ('1'..'7') to
the list of subcategory folder names that belong to it. The broad-category
model is trained to predict the broad category id; each subcategory model is
trained to predict the subcategory id within one broad category.

Train/val/test membership comes from the shared, leakage-free split file
data/splits/grouped_split.csv (load_split_dataframes), not from a random
per-model split: the old split_dataframe() put frames of the same source
video, and near-duplicate images, into different splits, and gave each model
a different test set. split_dataframe() is kept only for reference.

Built on tf.data.Dataset rather than the legacy
tensorflow.keras.preprocessing.image.ImageDataGenerator: under Keras 3,
model.fit() doesn't correctly treat that legacy generator as an
infinitely-repeating iterator the way Keras 2 did, so training silently runs
out of data after one pass (or worse, fakes progress without doing real
work). tf.data.Dataset is the Keras-3-native, properly supported way to do
this, and also gives real parallel/prefetched image decoding instead of
single-threaded PIL calls.
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data.const import (
    BROAD_CATEGORIES, IMG_SIZE, BATCH_SIZE, TEACHER_DATASET_DIR, SPLIT_FILE, column_category,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
SHUFFLE_SEED = 1337


def _list_images(folder):
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]


def build_broad_category_dataframe(dataset_dir=TEACHER_DATASET_DIR,
                                    broad_categories=BROAD_CATEGORIES):
    """One row per image, labeled with its broad category id ('1'..'7')."""
    rows = []
    for broad_category, subcategories in broad_categories.items():
        for subcategory in subcategories:
            folder = os.path.join(dataset_dir, subcategory)
            for image_path in _list_images(folder):
                rows.append({'path': image_path, 'label': broad_category})
    return pd.DataFrame(rows, columns=['path', 'label'])


def build_subcategory_dataframe(broad_category, dataset_dir=TEACHER_DATASET_DIR,
                                 broad_categories=BROAD_CATEGORIES):
    """One row per image belonging to `broad_category`, labeled with its
    subcategory folder name (e.g. '7_4_1')."""
    rows = []
    for subcategory in broad_categories[broad_category]:
        folder = os.path.join(dataset_dir, subcategory)
        for image_path in _list_images(folder):
            rows.append({'path': image_path, 'label': subcategory})
    return pd.DataFrame(rows, columns=['path', 'label'])


def split_dataframe(df, label_col='label', train_size=TRAIN_SPLIT,
                     val_size=VAL_SPLIT, test_size=TEST_SPLIT, seed=SHUFFLE_SEED):
    """Stratified 80/10/10 (by default) split of `df` by `label_col`."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    train_df, rest_df = train_test_split(
        df, train_size=train_size, stratify=df[label_col], random_state=seed
    )
    test_fraction_of_rest = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df, test_size=test_fraction_of_rest,
        stratify=rest_df[label_col], random_state=seed
    )
    return train_df, val_df, test_df


def _load_image(path, label, num_classes):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # No /255 rescale here: tf.keras.applications.EfficientNetB4 has its own
    # built-in Rescaling layer and expects raw [0, 255] pixels. Rescaling
    # twice crushes the input signal to near-zero and prevents the frozen
    # backbone from learning (this exact bug already bit this project once --
    # see old/handshape-hierarchical/models_broken_double_rescale_bug).
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


# Built once at module load time, not inside _augment_batch: these layers
# hold internal tf.Variable state (a random seed generator), and creating a
# new one on every tf.data.Dataset.map() call is rejected by Keras 3 with
# "tf.function only supports singleton tf.Variables created on the first
# call". Geometric ops (rotation/translation/zoom) are scale-independent;
# RandomBrightness defaults to operating on a [0, 255] value range, which
# matches the raw pixel scale we now feed it (see _load_image above).
_AUGMENTER = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(5 / 360),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomBrightness(0.25),
])


def _augment_batch(images, labels):
    # No horizontal flip: for handshapes, mirroring can change/invalidate
    # the label (e.g. thumb position, crossing direction are
    # chirality-dependent). Each layer samples an independent random
    # transform per example within the batch.
    return _AUGMENTER(images, training=True), labels


def _make_dataset(df, classes, augment, shuffle):
    class_to_index = {c: i for i, c in enumerate(classes)}
    paths = df['path'].tolist()
    label_indices = [class_to_index[label] for label in df['label']]
    num_classes = len(classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, label_indices))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SHUFFLE_SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: _load_image(path, label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_datasets(train_df, val_df, test_df):
    """Build train/val/test tf.data.Datasets from path/label dataframes.

    Returns (train_ds, val_ds, test_ds, num_classes, train_label_indices).
    train_label_indices is the integer-encoded label for every row in
    train_df, in the same class-index space as the datasets -- used for
    class_weight computation (the broad categories and several subcategories
    are heavily imbalanced).

    Augmentation is only applied to the training dataset, matching the
    paper's reported use of augmentation during training. Datasets are
    finite (one full pass per epoch); model.fit()/evaluate() consume them in
    full each call, no steps_per_epoch/repeat() bookkeeping needed.
    """
    classes = sorted(train_df['label'].unique())
    class_to_index = {c: i for i, c in enumerate(classes)}

    train_ds = _make_dataset(train_df, classes, augment=True, shuffle=True)
    val_ds = _make_dataset(val_df, classes, augment=False, shuffle=False)
    test_ds = _make_dataset(test_df, classes, augment=False, shuffle=False)

    train_label_indices = np.array([class_to_index[label] for label in train_df['label']])
    return train_ds, val_ds, test_ds, len(classes), train_label_indices


def load_split_dataframes(split_file=SPLIT_FILE, dataset_dir=TEACHER_DATASET_DIR):
    """(train_df, val_df, test_df) from the shared leakage-free split file
    (see data/make_grouped_split.py). Columns: path (absolute), subcategory,
    broad (HamNoSys row family), column (HamNoSys column category, Fig. 5C). The same split is used by every model of the hierarchy."""
    df = pd.read_csv(split_file, dtype=str)
    df['path'] = [os.path.join(dataset_dir, p) for p in df['path']]
    df = df.rename(columns={'cls': 'subcategory'})
    df['column'] = df['subcategory'].map(column_category)
    return tuple(df[df['split'] == name].reset_index(drop=True) for name in ('train', 'val', 'test'))


def load_broad_category_datasets():
    splits = [df.assign(label=df['broad']) for df in load_split_dataframes()]
    return build_datasets(*splits)


def load_flat_datasets():
    """Single-model baseline: all 34 fine-grained classes at once."""
    splits = [df.assign(label=df['subcategory']) for df in load_split_dataframes()]
    return build_datasets(*splits)


def load_subcategory_datasets(broad_category):
    splits = [
        df[df['broad'] == broad_category].assign(label=lambda d: d['subcategory'])
        for df in load_split_dataframes()
    ]
    return build_datasets(*splits)


def load_column_category_datasets():
    """Original-paper hierarchy (Fig. 5C): broad model over 9 column categories."""
    splits = [df.assign(label=df['column']) for df in load_split_dataframes()]
    return build_datasets(*splits)


def load_column_subcategory_datasets(column):
    """Original-paper hierarchy (Fig. 5D): fine classes within one column category."""
    splits = [
        df[df['column'] == column].assign(label=lambda d: d['subcategory'])
        for df in load_split_dataframes()
    ]
    return build_datasets(*splits)
