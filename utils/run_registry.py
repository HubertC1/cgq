"""Flat CSV ledger of every run: one row per (run_group, exp_name), columns are
flattened config fields plus result columns. Not a replacement for wandb (curves
live there) -- this is the fast "which run_id matches this config" index.
"""
import csv
import os


def flatten(d, prefix=''):
    """Flattens a nested dict/ConfigDict into {'a.b.c': value} pairs."""
    out = {}
    items = d.items() if hasattr(d, 'items') else vars(d).items()
    for k, v in items:
        key = f'{prefix}{k}'
        if hasattr(v, 'items') or hasattr(v, 'to_dict'):
            out.update(flatten(v, prefix=f'{key}.'))
        else:
            out[key] = v
    return out


def upsert_run(csv_path, key_cols, row):
    """Insert or update the row matching `key_cols` in `row`. Adds new columns as needed."""
    row = {k: ('' if v is None else v) for k, v in row.items()}

    rows = []
    fieldnames = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    for k in row:
        if k not in fieldnames:
            fieldnames.append(k)

    def matches(r):
        return all(str(r.get(k, '')) == str(row.get(k, '')) for k in key_cols)

    updated = False
    for r in rows:
        if matches(r):
            r.update(row)
            updated = True
            break
    if not updated:
        rows.append(row)

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
