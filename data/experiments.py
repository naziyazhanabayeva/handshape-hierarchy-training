"""Definitions of the four experiments of the SLAN-tool paper, Fig. 5 (A-D),
all on the same 34 fine-grained classes and the same leakage-free split
(data/splits/grouped_split.csv).

Class names encode the cell of the HamNoSys 4 handshape chart (Hanke 2010):
  '<row>_<col>'       top block      row = selected fingers, col = finger shape
  '7_<row>_<col>'     thumb-opposition block
Every grouping below is derived from that position, not hand-labelled.

  A  single model over all 34 classes (the Fig. 5A classes; 1_5 and 7_6_2
     have too few images and are excluded)
  B  grouping by HamNoSys ROW (Fig. 5B, 10 categories): broad model over the
     10 rows + one model per row to reach the 34 classes
  C  grouping by HamNoSys COLUMN (Fig. 5C, 9 categories): broad model over
     the 9 columns
  D  sub-models within each column category (Fig. 5D); with the C broad
     model this is the original paper's full hierarchy (34 classes)

Usage: python -m data.experiments    # writes data/splits/class_mapping.csv
"""
import csv
import os

from data.const import BROAD_CATEGORIES, COLUMN_CATEGORIES

FINE_CLASSES = sorted(s for subs in BROAD_CATEGORIES.values() for s in subs)

# HamNoSys 4 handshape chart (Hanke 2010) row/column names.
TOP_ROWS = {'1': 'Fist', '2': 'One Finger', '3': 'Two Fingers nonspread', '4': 'Two Fingers spread',
            '5': 'Flathand (Four Fingers nonspread)', '6': 'Four Fingers spread'}
TOP_COLUMNS = {'1': 'Selected Fingers Extended', '2': 'Selected Fingers Flattened',
               '3': 'Selected Fingers Bent', '4': 'Selected Fingers Hooked', '5': 'Derivation Examples'}
THUMB_ROWS = {'1': 'One Finger, others in fist position', '2': 'Two Fingers (nonspread), others in fist position',
              '3': 'Two Fingers (spread), others in fist position', '4': 'Four Fingers (nonspread)',
              '5': 'Four Fingers (spread)', '6': 'One Finger, others extended (spread)'}
THUMB_COLUMNS = {'1': 'Fingertip-Thumbtip Opposition w/ fingers rounded',
                 '2': 'Fingertip-Thumbtip Opposition w/ fingers flattened',
                 '3': 'Other thumb-opposition columns (fingers straight, hitchhiker\'s fingers, '
                      'thumb interphalangeal / metacarpophalangeal joint)',
                 '4': 'Derivation Examples'}


def row_category(cls):
    """Fig. 5B: rows 1-6 of the top block are categories 1-6; the thumb-
    opposition rows are 7 (row 1), 8 (rows 2-3), 9 (rows 4-5), 10 (row 6)."""
    p = cls.split('_')
    if p[0] != '7':
        return p[0]
    return {'1': '7', '2': '8', '3': '8', '4': '9', '5': '9', '6': '10'}[p[1]]


def column_category(cls):
    """Fig. 5C: columns 1-5 of the top block are categories 1-5; the thumb-
    opposition columns are 6 (rounded), 7 (flattened), 8 (narrow columns),
    9 (derivation examples)."""
    p = cls.split('_')
    return p[1] if p[0] != '7' else str(5 + int(p[2]))


def _groups(category_of):
    groups = {}
    for c in FINE_CLASSES:
        groups.setdefault(category_of(c), []).append(c)
    return groups


ROW_GROUPS = _groups(row_category)        # B: 10 categories
COLUMN_GROUPS = _groups(column_category)  # C / D: 9 categories
assert COLUMN_GROUPS == {k: sorted(v) for k, v in COLUMN_CATEGORIES.items()}

# Each experiment: the models it trains. A model = (name, category_of or None
# for identity labels, keep = set of classes it sees or None for all).
EXPERIMENTS = {
    'A': {'dir': 'A_flat_34', 'title': 'A: single 34-class model',
          'models': [('flat_34', None, None)]},
    'B': {'dir': 'B_rows_10', 'title': 'B: hierarchy by HamNoSys row (10 categories)',
          'models': [('broad', row_category, None)]
          + [('sub_' + g, None, set(cls)) for g, cls in sorted(ROW_GROUPS.items()) if len(cls) > 1]},
    'C': {'dir': 'C_columns_9', 'title': 'C: broad classifier by HamNoSys column (9 categories)',
          'models': [('broad', column_category, None)]},
    'D': {'dir': 'D_column_subcategories', 'title': 'D: sub-models within the column categories',
          'models': [('sub_' + g, None, set(cls)) for g, cls in sorted(COLUMN_GROUPS.items()) if len(cls) > 1]},
}


def class_mapping_rows():
    rows = []
    for c in FINE_CLASSES:
        p = c.split('_')
        if p[0] != '7':
            block, hrow, hcol = 'Handshape', TOP_ROWS[p[0]], TOP_COLUMNS[p[1]]
        else:
            block, hrow, hcol = 'Thumb opposition', THUMB_ROWS[p[1]], THUMB_COLUMNS[p[2]]
        rows.append({'class': c, 'chart_block': block, 'hamnosys_row': hrow, 'hamnosys_column': hcol,
                     'B_row_category': row_category(c), 'C_column_category': column_category(c)})
    return rows


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits', 'class_mapping.csv')
    rows = class_mapping_rows()
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', path)
    for name, groups in [('B rows', ROW_GROUPS), ('C columns', COLUMN_GROUPS)]:
        print(name + ':')
        for g, cls in sorted(groups.items(), key=lambda kv: int(kv[0])):
            print('  {:>2}: {}'.format(g, cls))
