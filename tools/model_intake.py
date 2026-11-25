"""Check owner-pinned E02 B/C metadata; never open clips or launch training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qolda-e02-model-intake-v1'
AUDIT_SCHEMA = 'qolda-e02-model-intake-audit-v1'
SCOPES = {'B': 'qolda-owned-e02', 'C': 'qolda-owned-e02+approved-extension'}
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SHA = re.compile(r'[a-f0-9]{64}')
TOKEN = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,95}')
PATH_PART = re.compile(r'[a-z0-9][a-z0-9_-]{0,95}')
LEGACY_NAME = re.compile(r'\d{2}_02_(?:00|01)_\d+')
PROTECTED = re.compile(r'(?:^|[._ -])(?:08|(?:tests?|holdouts?|quarantined?)\d*)(?:$|[._ -])')
PUBLIC = ('mendeley', '49h9dcwx5v', 'extending the elbow', 'extending_the_elbow')
DEVICE = re.compile(r'(?:con|prn|aux|nul|com[1-9]|lpt[1-9])')
ATTESTATIONS = (
    'local_development_approved', 'canonical_groups_across_sources',
    'near_duplicates_reviewed', 'one_labeled_e02_cycle_per_clip',
    'public_original_excluded', 'sealed_holdout_and_quarantine_excluded',
    'groups_disjoint_from_sealed_reserves',
)


class Rejected(ValueError):
    """Only fixed reason codes, safe to put in the audit."""


def require(condition, code):
    if not condition:
        raise Rejected(code)


def protected(value):
    value = value.casefold()
    return bool(PROTECTED.search(value) or any(marker in value for marker in PUBLIC))


def matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def token(value):
    return matches(TOKEN, value) and not protected(value)


def keys(value, expected, code):
    require(isinstance(value, dict) and value.keys() == set(expected.split()), code)


def reject_reparse(st):
    require(not stat.S_ISLNK(st.st_mode)
            and not getattr(st, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
            'linked_location')


def private_dir(path):
    if path is None or not path.is_absolute():
        raise Rejected('private_directory_must_be_absolute')
    require(not any(part == '..' or protected(part) for part in path.parts[1:]), 'protected_location')
    require(not path.is_relative_to(ROOT), 'directory_inside_git_checkout')
    # Check parents first; do not follow symlinks or Windows junctions.
    for folder in [*reversed(path.parents), path]:
        st = folder.lstat()
        reject_reparse(st)
        require(stat.S_ISDIR(st.st_mode), 'not_a_directory')
        try:
            (folder / '.git').lstat()
        except FileNotFoundError:
            continue
        raise Rejected('directory_inside_git_checkout')
    return path


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'duplicate_json_key')
        obj[key] = value
    return obj


def load_manifest(directory, name, digest, cohort):
    # Names distinguish this handoff from historical manifests before file I/O.
    pattern = re.compile(r'qolda-e02-' + cohort + r'-[a-z0-9][a-z0-9_-]{0,63}\.json')
    require(matches(pattern, name) and not protected(name), 'unsafe_manifest_name')
    require(matches(SHA, digest) and digest != '0' * 64, 'invalid_owner_digest')
    path = private_dir(directory) / name
    st = path.lstat()
    reject_reparse(st)
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'manifest_not_regular_single_link')
    require(0 < st.st_size <= MAX_MANIFEST_BYTES, 'manifest_size_out_of_bounds')
    with path.open('rb') as file:
        raw = file.read(MAX_MANIFEST_BYTES + 1)
    require(len(raw) == st.st_size and hashlib.sha256(raw).hexdigest() == digest, 'owner_digest_mismatch')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object)


def valid_clip_path(value):
    if not isinstance(value, str) or len(value) > 240 or not value.endswith('.mp4'):
        return False
    parts = value[:-4].split('/')
    return not LEGACY_NAME.fullmatch(parts[-1]) and all(
        matches(PATH_PART, p) and not protected(p) and not DEVICE.fullmatch(p) for p in parts)


def group_id(value, prefix):
    return isinstance(value, str) and re.fullmatch(prefix + r'-[a-f0-9]{16,64}', value) is not None


def parse_manifest(doc, cohort):
    keys(doc, 'schema cohort exercise group_namespace attestation sources rows', 'manifest_shape')
    require(doc['schema'] == SCHEMA and doc['cohort'] == cohort and doc['exercise'] == 'E02',
            'wrong_schema_cohort_or_exercise')
    require(token(doc['group_namespace']), 'invalid_group_namespace')
    att = doc['attestation']
    keys(att, 'owner_id approval_ref ' + ' '.join(ATTESTATIONS), 'attestation_shape')
    require(att['owner_id'] == 'data-owner' and token(att['approval_ref'])
            and all(att[k] is True for k in ATTESTATIONS), 'missing_owner_attestation')

    require(isinstance(doc['sources'], list) and doc['sources'], 'no_provenance_sources')
    sources = {}
    expected_splits = {'train', 'validation'} if cohort == 'owned' else {'train'}
    for source in doc['sources']:
        keys(source, 'id kind revision provenance_ref rights_ref consent_ref annotation_ref local_training_authorized '
             'contains_patient_data allowed_splits', 'provenance_shape')
        require(all(token(source[k]) for k in ('id', 'revision', 'provenance_ref', 'rights_ref', 'consent_ref',
                                              'annotation_ref')), 'missing_source_proof_reference')
        require(source['kind'] == ('project_recorded' if cohort == 'owned' else 'approved_extension'),
                'wrong_source_kind')
        require(source['local_training_authorized'] is True and source['contains_patient_data'] is False,
                'source_rights_not_admitted')
        splits = source['allowed_splits']
        require(isinstance(splits, list) and splits and all(isinstance(s, str) for s in splits)
                and len(set(splits)) == len(splits) and set(splits) <= expected_splits, 'source_split_not_admitted')
        require(source['id'] not in sources, 'duplicate_source_id')
        sources[source['id']] = set(splits)

    require(isinstance(doc['rows'], list) and doc['rows'], 'empty_corpus')
    seen_sources = set()
    for row in doc['rows']:
        keys(row, 'path split label sha256 source_id subject recording family protocol duration_ms', 'row_shape')
        require(isinstance(row['split'], str) and row['split'] in expected_splits, 'protected_or_unexpected_split')
        require(token(row['source_id']) and row['split'] in sources.get(row['source_id'], ()), 'unapproved_row_source')
        require(valid_clip_path(row['path']), 'unsafe_or_legacy_clip_path')
        require(matches(SHA, row['sha256']) and row['sha256'] != '0' * 64, 'invalid_clip_digest')
        require(type(row['label']) is int and row['label'] in (0, 1), 'invalid_e02_label')
        require(all(group_id(row[name], prefix) for name, prefix in (('subject', 's'), ('recording', 'r'),
                                                                     ('family', 'f'))), 'invalid_group_id')
        require(row['protocol'] == 'e02-single-cycle-v1' and type(row['duration_ms']) is int
                and 0 < row['duration_ms'] <= 30_000, 'unapproved_clip_protocol_or_duration')
        seen_sources.add(row['source_id'])
    require(seen_sources == sources.keys(), 'unused_source_approval')
    return doc


def check_groups(rows):
    paths, digests = set(), set()
    subjects, recordings, families = {}, {}, {}
    for row in rows:
        require(row['path'] not in paths, 'duplicate_clip_path')
        require(row['sha256'] not in digests, 'duplicate_clip_digest')
        paths.add(row['path'])
        digests.add(row['sha256'])
        split, subject, recording, family = (row[k] for k in ('split', 'subject', 'recording', 'family'))
        require(subjects.setdefault(subject, split) == split, 'subject_split_overlap')
        require(recordings.setdefault(recording, (split, subject)) == (split, subject),
                'recording_split_or_subject_overlap')
        family_binding = (split, subject, recording, row['label'], row['source_id'])
        require(families.setdefault(family, family_binding) == family_binding, 'inconsistent_or_overlapping_family')


def summary(rows):
    labels = Counter(row['label'] for row in rows)
    return dict(clips=len(rows), incomplete=labels[0], complete=labels[1],
                **{f'{k}_groups': len({row[k] for row in rows}) for k in ('subject', 'recording', 'family')})


def intake(directory, name, digest, cohort):
    if name is None and digest is None:
        return None, 'missing_owner_pinned_' + cohort + '_manifest'
    if name is None or digest is None:
        return None, 'manifest_name_and_owner_digest_required'
    try:
        doc = parse_manifest(load_manifest(directory, name, digest, cohort), cohort)
        check_groups(doc['rows'])
        if cohort == 'owned':
            for split in ('train', 'validation'):
                require({r['label'] for r in doc['rows'] if r['split'] == split} == {0, 1},
                        'owned_' + split + '_requires_both_classes')
        return doc, None
    except Rejected as error:
        return None, str(error)
    except FileNotFoundError:
        return None, 'approved_manifest_or_directory_absent'
    except OSError:
        return None, 'metadata_io_error'
    except (UnicodeError, ValueError, RecursionError):
        return None, 'invalid_metadata_json'


def variant_status(variant, blockers, train=None, validation=None, extended_train=None):
    entry: dict[str, Any] = dict(status='blocked_inputs' if blockers else 'metadata_ready', blockers=blockers,
                                 corpus_scope=SCOPES[variant], training='not_run')
    if not blockers:
        entry.update(train=summary(train), validation=summary(validation))
        if extended_train:
            entry['extension_train'] = summary(extended_train)
    return entry


def audit(args) -> tuple[dict[str, Any], dict[str, list]]:
    own, own_error = intake(args.metadata_dir, args.owned_name, args.owned_sha256, 'owned')
    more, more_error = (intake(args.metadata_dir, args.expanded_name, args.expanded_sha256, 'expanded')
                        if args.variant != 'B' else (None, None))
    train = [dict(r, origin='owned') for r in own['rows'] if r['split'] == 'train'] if own else []
    val = [dict(r, origin='owned') for r in own['rows'] if r['split'] == 'validation'] if own else []
    extension = [dict(r, origin='expanded') for r in more['rows']] if more else []
    variants, rows = {}, {}
    if args.variant != 'C':
        variants['B'] = variant_status('B', [own_error] if own_error else [], train, val)
        rows['B'] = train + val
    if args.variant != 'B':
        blockers = [error for error in (own_error, more_error) if error]
        if own is not None and more is not None:
            try:
                require(own['group_namespace'] == more['group_namespace'], 'group_namespace_mismatch')
                require({s['id'] for s in own['sources']}.isdisjoint(s['id'] for s in more['sources']),
                        'source_id_reused_between_corpora')
                check_groups(train + extension + val)
            except Rejected as error:
                blockers.append(str(error))
        variants['C'] = variant_status('C', blockers, train + extension, val, extension)
        rows['C'] = train + extension + val
    return dict(schema=AUDIT_SCHEMA, metadata_only=True, variants=variants), rows


def emit(args, rows):
    directory = private_dir(args.emit_dir)
    variant = args.variant
    # Train-gpu manifest v2 carries indirect provenance needed to audit real exposure:
    # source_id and family (hashed group ids), plus which corpus the row came from.

    manifest = dict(schema='qolda-e02-train-gpu-manifest-v2', variant=variant, corpus_scope=SCOPES[variant],
                    owned_metadata_sha256=args.owned_sha256, rows=[
                        {key: row[key] for key in ('path', 'split', 'label', 'sha256', 'source_id', 'family')}
                        | {'origin': row['origin']} for row in rows])
    if variant == 'C':
        manifest['expanded_metadata_sha256'] = args.expanded_sha256
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with (directory / f'{variant}.train-gpu.json').open('xb') as file:
        file.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('B', 'C', 'both'), default='both')
    parser.add_argument('--metadata-dir', type=Path, help='explicitly approved absolute project-private directory')
    parser.add_argument('--owned-name', help='exact approved metadata JSON basename')
    parser.add_argument('--owned-sha256', help='digest confirmed through a trusted owner channel')
    parser.add_argument('--expanded-name', help='exact approved extension metadata JSON basename')
    parser.add_argument('--expanded-sha256', help='extension digest confirmed by the owner')
    parser.add_argument('--emit-dir', type=Path, help='optional existing private directory for one trainer manifest')
    args = parser.parse_args(argv)
    if args.emit_dir and args.variant == 'both':
        parser.error('--emit-dir requires --variant B or C')
    report, rows = audit(args)
    ready = all(v['status'] == 'metadata_ready' for v in report['variants'].values())
    rc = 0 if ready else 2
    if args.emit_dir and ready:
        try:
            report['trainer_manifest_sha256'] = emit(args, rows[args.variant])
            report['trainer_manifest_emitted'] = True
        except (Rejected, OSError):
            report['trainer_manifest_emitted'] = False
            report['emit_error'] = 'unsafe_unwritable_or_existing_output'
            rc = 3
    print(json.dumps(report, sort_keys=True, indent=2))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
