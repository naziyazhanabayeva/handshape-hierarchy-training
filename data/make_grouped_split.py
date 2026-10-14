"""Build the leakage-free 80/10/10 train/val/test split used by all 8
models (data/splits/grouped_split.csv).

Images are grouped so that everything that could leak between splits stays
in ONE split:
  - frames of the same source video (parsed from the filename, e.g.
    't_997.mp4_fn000032-0.png' -> 't_997.mp4', '01frame198.jpg' -> '01frame'),
  - byte-identical files (MD5),
  - near-duplicate images (difference hash, Hamming distance <= NEAR_DUP_DIST
    of 128 bits).
Groups are formed with union-find, then assigned to splits with
StratifiedGroupKFold (10 folds: fold 0 = test, fold 1 = val, rest = train),
stratified by fine-grained class. Among SEARCH_SEEDS seeds, the one whose
per-class val/test fractions are closest to 10% (with every class present
in val and test) is kept.

Byte-identical images that appear under two different class folders are
label conflicts; they are excluded and listed in label_conflicts.csv.

One split is shared by the broad model and all 7 subcategory models, so an
image is in the same split for every model of the hierarchy.

Usage (from classify_handshapes/):
    python -m data.make_grouped_split            # writes data/splits/*.csv
"""
import collections
import hashlib
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from data.const import BROAD_CATEGORIES, TEACHER_DATASET_DIR

SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
NEAR_DUP_DIST = 10
SEARCH_SEEDS = 40
KEPT_CLASSES = {s for subs in BROAD_CATEGORIES.values() for s in subs}


def video_id(rel_path):
    """Source-video id of an image, or None if the filename doesn't say."""
    folder, name = rel_path.split('/', 1)
    m = re.match(r'(.+\.mp4)_fn\d+', name)
    if m and not m.group(1).startswith('*'):
        return m.group(1)  # named video, can span class folders
    m = re.match(r'(\d+(?:_\d+)?)frame\d+', name)
    if m:
        return folder + '/' + m.group(1) + 'frame'  # numbered clip within a folder
    return None


def _hash_image(args):
    root, rel = args
    path = os.path.join(root, rel)
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    im = Image.open(path).convert('L')
    h = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    v = np.asarray(im.resize((8, 9), Image.LANCZOS), dtype=np.int16)
    bits = np.concatenate([(h[:, 1:] > h[:, :-1]).ravel(), (v[1:] > v[:-1]).ravel()])
    return rel, md5, np.packbits(bits).tobytes().hex()


def hash_dataset(root):
    rels = [
        cls + '/' + name
        for cls in sorted(os.listdir(root)) if cls in KEPT_CLASSES
        for name in sorted(os.listdir(os.path.join(root, cls)))
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]
    with Pool() as pool:
        rows = pool.map(_hash_image, [(root, r) for r in rels], chunksize=200)
    df = pd.DataFrame(rows, columns=['path', 'md5', 'phash'])
    df['cls'] = df['path'].str.split('/').str[0]
    return df


def _bits(df):
    raw = np.frombuffer(bytes.fromhex(''.join(df['phash'])), dtype=np.uint8)
    return np.unpackbits(raw.reshape(len(df), 16), axis=1).astype(bool)


def near_duplicate_pairs(df, max_dist=NEAR_DUP_DIST):
    """All (i, j) with Hamming distance <= max_dist. Candidate pairs come from
    8 exact 16-bit hash blocks (pigeonhole: any pair within 7 bits shares a
    block; pairs at 8-10 bits are found when they share one too)."""
    bits = _bits(df)
    blocks = np.packbits(bits, axis=1).view('>u2')  # (n, 8) 16-bit blocks
    pairs = set()
    for b in range(blocks.shape[1]):
        buckets = collections.defaultdict(list)
        for i, v in enumerate(blocks[:, b]):
            buckets[v].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            idx = np.array(idx)
            B = bits[idx]
            for k in range(0, len(idx), 500):
                d = (B[k:k + 500, None, :] != B[None, :, :]).sum(2)
                for a, c in zip(*np.nonzero(d <= max_dist)):
                    if a + k < c:
                        pairs.add((idx[a + k], idx[c]))
    return pairs


def leakage_groups(df, pairs):
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    first = {}
    for i, (path, md5) in enumerate(zip(df['path'], df['md5'])):
        for key in (('video', video_id(path)), ('md5', md5)):
            if key[1] is None:
                continue
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    for a, b in pairs:
        union(a, b)
    # Canonical id = smallest row index in the group, so the split does not
    # depend on the order in which unions happened.
    roots = np.array([find(i) for i in range(len(df))])
    return pd.Series(np.arange(len(df))).groupby(roots).transform('min').to_numpy()


def assign_splits(df):
    best = None
    for seed in range(SEARCH_SEEDS):
        folds = np.empty(len(df), dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        for k, (_, test_idx) in enumerate(sgkf.split(df['path'], df['cls'], df['group'])):
            folds[test_idx] = k
        split = np.where(folds == 0, 'test', np.where(folds == 1, 'val', 'train'))
        counts = pd.crosstab(df['cls'], split)
        if (counts[['val', 'test']] == 0).any(axis=None):
            continue
        frac = counts.div(counts.sum(axis=1), axis=0)
        err = ((frac[['val', 'test']] - 0.1) ** 2).sum(axis=None)
        if best is None or err < best[0]:
            best = (err, seed, split)
    print('chosen seed:', best[1])
    return best[2]


def main(dataset_dir=TEACHER_DATASET_DIR):
    os.makedirs(SPLITS_DIR, exist_ok=True)
    df = hash_dataset(dataset_dir)

    n_labels = df.groupby('md5')['cls'].nunique()
    conflict_md5 = set(n_labels[n_labels > 1].index)
    conflicts = df[df['md5'].isin(conflict_md5)]
    conflicts[['path', 'md5']].to_csv(os.path.join(SPLITS_DIR, 'label_conflicts.csv'), index=False)
    df = df[~df['md5'].isin(conflict_md5)].reset_index(drop=True)

    df['group'] = leakage_groups(df, near_duplicate_pairs(df))
    df['split'] = assign_splits(df)
    df['broad'] = df['cls'].str.split('_').str[0]
    df[['path', 'cls', 'broad', 'split']].to_csv(
        os.path.join(SPLITS_DIR, 'grouped_split.csv'), index=False)
    print('{} images ({} label-conflict images excluded), {} leakage groups'.format(
        len(df), len(conflicts), df['group'].nunique()))
    print(df['split'].value_counts().to_dict())


if __name__ == '__main__':
    main()
