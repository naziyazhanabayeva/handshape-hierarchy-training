"""Verify that data/splits/grouped_split.csv has no leakage between train,
val and test. Recomputes everything from the image files, so it can be run on
the training server against its own copy of the dataset before training.

Checks:
  1. every image in the dataset is either in the split file or a listed
     label conflict / excluded class, and every split-file path exists;
  2. no source video has frames in more than one split;
  3. no byte-identical (MD5) image appears in more than one split;
  4. no near-duplicate pair (dHash distance <= 10) crosses splits;
  5. every class is present in train, val and test.
Exits with status 1 if any check fails.

Usage (from classify_handshapes/):
    python -m data.check_split
"""
import os
import sys

import pandas as pd

from data.const import TEACHER_DATASET_DIR
from data.make_grouped_split import (
    SPLITS_DIR, hash_dataset, near_duplicate_pairs, video_id,
)


def main(dataset_dir=TEACHER_DATASET_DIR):
    split = pd.read_csv(os.path.join(SPLITS_DIR, 'grouped_split.csv'))
    conflicts = set(pd.read_csv(os.path.join(SPLITS_DIR, 'label_conflicts.csv'))['path'])
    hashes = hash_dataset(dataset_dir)
    failures = []

    on_disk = set(hashes['path'])
    missing = set(split['path']) - on_disk
    unassigned = on_disk - set(split['path']) - conflicts
    print('images on disk: {}, in split file: {}, label conflicts: {}'.format(
        len(on_disk), len(split), len(conflicts)))
    if missing:
        failures.append('{} split-file images missing on disk, e.g. {}'.format(len(missing), sorted(missing)[:3]))
    if unassigned:
        failures.append('{} images on disk not in split file, e.g. {}'.format(len(unassigned), sorted(unassigned)[:3]))

    df = hashes.merge(split[['path', 'split']], on='path')
    df['video'] = df['path'].map(video_id)

    n_videos = df.dropna(subset=['video']).groupby('video')['split'].nunique()
    n_md5 = df.groupby('md5')['split'].nunique()
    pairs = near_duplicate_pairs(df)
    cross = [(a, b) for a, b in pairs if df['split'][a] != df['split'][b]]
    print('source videos: {} ({} span splits)'.format(len(n_videos), int((n_videos > 1).sum())))
    print('MD5 groups spanning splits: {}'.format(int((n_md5 > 1).sum())))
    print('near-duplicate pairs: {} ({} cross splits)'.format(len(pairs), len(cross)))
    if (n_videos > 1).any():
        failures.append('videos in >1 split: {}'.format(list(n_videos[n_videos > 1].index[:5])))
    if (n_md5 > 1).any():
        failures.append('{} identical images in >1 split'.format(int((n_md5 > 1).sum())))
    if cross:
        failures.append('{} near-duplicate pairs cross splits, e.g. {}'.format(
            len(cross), [(df['path'][a], df['path'][b]) for a, b in cross[:3]]))

    counts = pd.crosstab(df['cls'], df['split']).reindex(columns=['train', 'val', 'test'], fill_value=0)
    empty = counts[(counts == 0).any(axis=1)]
    if len(empty):
        failures.append('classes missing from a split: {}'.format(list(empty.index)))

    if failures:
        print('\nFAILED:\n  ' + '\n  '.join(failures))
        sys.exit(1)
    print('\nOK: no leakage between train, val and test.')


if __name__ == '__main__':
    main()
