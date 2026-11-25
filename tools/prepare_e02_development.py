"""Build the private E02 development manifest from the public Mendeley V1 archive.

Only train/validation participants are admitted. The ``Test`` folders are
skipped **before any file is opened or hashed** — the historical holdout stays
closed. The known folder/name conflict (``Complete/Train/04_02_00_*``) goes to
quarantine, never fixed by guessing.

Writes the manifest outside the checkout (it carries per-clip hashes, which the
repository does not store).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

TRAIN = {'01', '02', '03', '04'}
VALIDATION = {'05'}
EXPECTED = {('train', 0): 47, ('train', 1): 30, ('validation', 0): 10, ('validation', 1): 10,
            ('quarantine', None): 10}
NAME = re.compile(r'(\d{2})_02_(00|01)_(\d{2})')


def sha256(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def collect(root):
    root = Path(root)
    rows, seen = [], set()
    for directory in ('Complete', 'Incomplete'):
        for folder in ('Train', 'Test'):
            if folder == 'Test':
                continue  # closed holdout: never opened
            for path in sorted((root / directory / folder).glob('*.mp4')):
                match = NAME.fullmatch(path.stem)
                if not match:
                    raise ValueError(f'unexpected E02 name: {path.name}')
                participant, code, _index = match.groups()
                if participant not in TRAIN | VALIDATION:
                    raise ValueError(f'participant outside the development split: {path.name}')
                label = int(code)
                split = 'train' if participant in TRAIN else 'validation'
                relative = path.relative_to(root.parents[1]).as_posix()
                if (directory, folder, participant, code) == ('Complete', 'Train', '04', '00'):
                    split, label = 'quarantine', None
                elif directory != ('Complete' if label else 'Incomplete'):
                    raise ValueError(f'label conflict: {relative}')
                digest = sha256(path)
                if digest in seen:
                    raise ValueError(f'duplicate bytes: {relative}')
                seen.add(digest)
                rows.append(dict(path=relative, participant=participant, label=label,
                                 split=split, sha256=digest, bytes=path.stat().st_size,
                                 reason='folder_filename_conflict' if label is None else None))
    return rows


def verify(rows):
    counts = Counter((r['split'], r['label']) for r in rows)
    if counts != EXPECTED:
        raise ValueError(f'unexpected counts: {dict(counts)}')
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('duplicate digest')
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path,
                    help='.../Exercise/2_Extending the Elbow of the extracted archive')
    ap.add_argument('--out', required=True, type=Path, help='manifest.json outside the checkout')
    args = ap.parse_args(argv)
    rows = collect(args.source)
    counts = verify(rows)
    manifest = dict(schema='seriq-e02-development-manifest-v1', exercise='E02',
                    source='Mendeley V1, CC BY 4.0, DOI 10.17632/49h9dcwx5v.1',
                    split_rule='participants 01-04 train, 05 validation; Test closed before opening',
                    counts={f'{k[0]}:{k[1]}': v for k, v in sorted(counts.items(), key=str)},
                    rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + '.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + '\n',
                   encoding='utf8')
    tmp.replace(args.out)
    print(json.dumps(dict(rows=len(rows), counts=manifest['counts'],
                          manifest_sha256=hashlib.file_digest(args.out.open('rb'),
                                                              'sha256').hexdigest()),
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
