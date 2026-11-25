"""Inventory the Qolda B/C intake without opening any corpus or manifest."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import stat
import subprocess

from .model_intake import ROOT, audit, protected


REPORT = ROOT / 'reports/ml/model-intake/audit.json'
KNOWN_PRIVATE_ROOTS = {'gpu_workspace': 'qolda-gpu', 'historical_workspace': 'seriq-local'}
CHECKOUT_DIRS = ('.', 'ml', 'artifacts/seriq', 'reports/ml', 'docs', 'tools',
                 'data', 'datasets', 'raw-data', 'local-data', '.seriq-local')
META_MARKERS = ('manifest', 'provenance', 'split', 'corpus', 'intake', 'e02')
META_SUFFIXES = {'.json', '.jsonl', '.csv', '.tsv', '.yaml', '.yml', '.toml'}


def linked(st):
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def directory_state(path):
    # Check ancestry without dereferencing linked directories.
    for part in [*reversed(path.parents), path]:
        try:
            st = part.lstat()
        except FileNotFoundError:
            return 'absent'
        except OSError:
            return 'unreadable'
        if linked(st):
            return 'link_not_followed'
        if not stat.S_ISDIR(st.st_mode):
            return 'not_directory'
    return 'present'


def inventory_dir(path, *, manifest_dir=False):
    state = directory_state(path)
    info: dict[str, str | int | bool] = {'state': state}
    if state != 'present':
        return info
    count, children = 0, 0
    known_public = False
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                name = entry.name.casefold()
                if protected(name):
                    continue
                st = entry.stat(follow_symlinks=False)
                if linked(st):
                    continue
                if stat.S_ISDIR(st.st_mode):
                    children += 1
                elif stat.S_ISREG(st.st_mode) and Path(name).suffix in META_SUFFIXES:
                    if manifest_dir and name == 'e02-dev-v1.json':
                        known_public = True
                    elif manifest_dir or any(marker in name for marker in META_MARKERS):
                        count += 1
    except OSError:
        return {'state': 'unreadable'}
    info.update(immediate_directory_count=children, other_metadata_candidate_count=count)
    if manifest_dir:
        info['documented_public_a_manifest_present'] = known_public
    return info


def build_report():
    source = Path.home() / 'source'
    gpu = source / KNOWN_PRIVATE_ROOTS['gpu_workspace']
    result, _ = audit(argparse.Namespace(metadata_dir=None, owned_name=None, owned_sha256=None,
                                       expanded_name=None, expanded_sha256=None, variant='both'))
    result.update(
        checked_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        inspection=dict(
            checkout_directory_metadata={name: inventory_dir(ROOT / name) for name in CHECKOUT_DIRS},
            documented_private_locations={name: inventory_dir(source / folder)
                                          for name, folder in KNOWN_PRIVATE_ROOTS.items()},
            gpu_workspace_metadata_dir=inventory_dir(gpu / 'manifests', manifest_dir=True),
            gpu_workspace_dataset_top_only=inventory_dir(gpu / 'dataset'),
            candidate_manifest_contents_opened=0,
            media_files_opened_or_hashed=0,
            old_holdout_and_quarantine_opened_or_hashed=False,
            notes=[
                'Directory-entry metadata only; this is not a search outside the project-owned locations.',
                'The documented public A manifest is excluded by identity, not opened or hashed.',
                'No owner-pinned handoff for either cohort was supplied to this run.',
            ],
            project_evidence=['execplan/qolda-gpu-handoff.md:101-117',
                              'reports/ml/gpu-handoff/M0-M1-report.md:24-34,107-112'],
        ),
        training=dict(trainer_invocations=0, checkpoints_created=0, optimization_steps=0),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-audit', action='store_true', help='write a new sanitized audit at the fixed report path')
    args = parser.parse_args()
    payload = json.dumps(build_report(), indent=2, sort_keys=True) + '\n'
    if args.write_audit:
        if directory_state(REPORT.parent.parent) != 'present':
            parser.error('unsafe report parent')
        REPORT.parent.mkdir(exist_ok=True)
        if directory_state(REPORT.parent) != 'present':
            parser.error('unsafe report directory')
        with REPORT.open('x', encoding='utf-8', newline='\n') as file:
            file.write(payload)
    print(payload, end='')
    return 2  # This inventory has no owner-approved handoff and cannot admit training.


if __name__ == '__main__':
    raise SystemExit(main())
