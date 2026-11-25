"""Build/apply a data-only, base-pinned Windows frontend update; never alter inputs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qolda-windows-frontend-update-v1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
WEB = re.compile(r'frontend/(?:index\.html|brand\.html|lab\.html|src/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.js|styles/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.css)\Z')
# Match the accepted engine's explicit public asset routes. Licences accompany them.
ASSETS = frozenset('assets/' + x for x in (
    'brand/qolda-mark.svg', 'brand/qolda-mark-dark.svg',
    'fonts/IBMPlexSans.ttf', 'fonts/SourceSerif4.ttf',
    'fonts/OFL-IBMPlexSans.txt', 'fonts/OFL-SourceSerif4.txt',
    'models/mannequin.glb', 'models/CREDITS.md',
    'vendor/three/BufferGeometryUtils.js', 'vendor/three/GLTFLoader.js',
    'vendor/three/RoomEnvironment.js', 'vendor/three/three.module.min.js',
    'vendor/three/LICENSE', 'instruction/instruction.ru.vtt',
    'instruction/instruction.kk.vtt'))
DOCS = {'frontend/README.md', 'frontend/DESIGN.md'}
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024


def fail(message):
    raise ValueError(message)


def sha_file(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def record(data):
    return [len(data), hashlib.sha256(data).hexdigest()]


def valid_sha(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        fail('expected lowercase SHA-256')
    return value


def safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        fail('invalid path')
    parts = name.split('/')
    if any(not p or p in ('.', '..') or ':' in p or p[-1] in ' .' or
           re.search(r'[<>"|?*\x00-\x1f]', p) or
           re.fullmatch(r'(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', p)
           for p in parts):
        fail('unsafe path: ' + name)
    return name


def read_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('ascii')


def inventory(z):
    result, folded = {}, set()
    for info in z.infolist():
        name = safe_name(info.filename)
        if info.is_dir() or name.casefold() in folded or (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
            fail('directory, link, or duplicate ZIP member: ' + name)
        if info.flag_bits & 1:
            fail('encrypted ZIP member: ' + name)
        folded.add(name.casefold())
        result[name] = info
    return result


def valid_record(value):
    if (not isinstance(value, list) or len(value) != 2 or type(value[0]) is not int or
            value[0] < 0):
        fail('invalid file record')
    valid_sha(value[1])
    return value


def allowed(name):
    return bool(WEB.fullmatch(name)) or name in ASSETS


def baseline(zip_path, expected):
    """Authenticate the ZIP, bind the engine manifest and all other base members."""
    valid_sha(expected)
    if sha_file(zip_path) != expected:
        fail('base ZIP SHA-256 mismatch')
    with ZipFile(zip_path) as z:
        infos = inventory(z)
        name = 'engine-win64/manifest.json'
        if name not in infos or infos[name].file_size > 32 * 1024 * 1024:
            fail('missing or oversized base engine manifest')
        raw = z.read(name)
        manifest = read_json(raw)
        if manifest.get('schema') != 'qolda-private-windows-engine-v1' or not isinstance(manifest.get('files'), dict):
            fail('wrong base engine manifest')
        expected_files = {name: record(raw)}
        for rel, value in manifest['files'].items():
            safe_name(rel)
            full = 'engine-win64/' + rel
            if full == name:
                fail('recursive manifest entry')
            expected_files[full] = valid_record(value)
        # Small non-engine members are authenticated directly by the trusted ZIP.
        for member, info in infos.items():
            if member.startswith('windows/') or member == 'START-HERE.txt':
                h = hashlib.sha256()
                with z.open(info) as src:
                    for chunk in iter(lambda: src.read(1024 * 1024), b''):
                        h.update(chunk)
                expected_files[member] = [info.file_size, h.hexdigest()]
        if set(infos) != set(expected_files) or any(infos[k].file_size != v[0] for k, v in expected_files.items()):
            fail('base ZIP and engine manifest inventory/size mismatch')
        if 'START-HERE.txt' not in infos or 'windows/SHA256SUMS.txt' not in infos:
            fail('base Windows metadata missing')
        lines = z.read('windows/SHA256SUMS.txt').decode('ascii').splitlines()
        sums = {}
        for line in lines:
            sha, rel = line.split('  ', 1)
            valid_sha(sha)
            full = 'windows/' + safe_name(rel)
            if full in sums:
                fail('duplicate Windows checksum')
            sums[full] = sha
        windows = {x for x in expected_files if x.startswith('windows/') and x != 'windows/SHA256SUMS.txt'}
        if set(sums) != windows or any(expected_files[k][1] != v for k, v in sums.items()):
            fail('base Windows checksums disagree')
    return manifest, raw, expected_files


def get_git_files(ref):
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', ref):
        fail('ref must be the full, explicitly approved lowercase commit ID')
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    git_env['GIT_NO_REPLACE_OBJECTS'] = '1'
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}'], cwd=ROOT, env=git_env, text=True).strip()
    if commit != ref:
        fail('ref is not the exact commit ID')
    listing = subprocess.check_output(['git', 'ls-tree', '-r', '-z', '-l', '--full-tree', commit, '--', 'frontend', 'assets'], cwd=ROOT, env=git_env)
    selected, total = {}, 0
    for item in listing.split(b'\0'):
        if not item:
            continue
        meta, raw_name = item.split(b'\t', 1)
        mode, kind, obj, size = meta.decode('ascii').split()
        name = safe_name(raw_name.decode('utf-8'))
        if name in DOCS:
            continue
        if not allowed(name):
            fail('Git path outside explicit supported allowlist: ' + name)
        if mode not in ('100644', '100755') or kind != 'blob':
            fail('Git link/submodule is not a frontend file: ' + name)
        size = int(size)
        if size > MAX_FILE:
            fail('frontend file too large: ' + name)
        total += size
        selected[name] = (obj, size)
    if total > MAX_TOTAL or 'frontend/index.html' not in selected:
        fail('invalid or oversized Git snapshot')
    if len({x.casefold() for x in selected}) != len(selected):
        fail('case-colliding Git paths')
    # Read exact immutable blobs, not worktree files or archive attribute filters.
    request = ''.join(obj + '\n' for obj, _ in selected.values()).encode('ascii')
    response = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT, env=git_env, input=request,
                              stdout=subprocess.PIPE, check=True).stdout
    stream, payload = io.BytesIO(response), {}
    for name, (obj, size) in selected.items():
        if stream.readline() != f'{obj} blob {size}\n'.encode('ascii'):
            fail('unexpected Git object header: ' + name)
        data = stream.read(size)
        if len(data) != size or stream.read(1) != b'\n':
            fail('truncated Git object: ' + name)
        payload[name] = data
    if stream.read(1):
        fail('unexpected extra Git object data')
    return commit, payload


def old_routes(manifest):
    # Non-public docs/licences in the old package remain untouched; every served
    # frontend path and every explicitly supported old asset must survive.
    result = set()
    for name in manifest['files']:
        if name.startswith('app/frontend/'):
            rel = name[4:]
            if rel in DOCS:
                continue
            if not WEB.fullmatch(rel):
                fail('unknown old frontend route: ' + rel)
            result.add(rel)
        elif name.startswith('app/assets/'):
            rel = name[4:]
            if rel.endswith(('.js', '.svg', '.ttf', '.glb', '.vtt', '.mp4')):
                if rel not in ASSETS:
                    fail('old public asset cannot be represented safely: ' + rel)
                result.add(rel)
    return result


def check_out(out, *inputs):
    out = out.absolute()
    parent = out.parent.resolve(strict=True)
    dest = parent / out.name
    if not out.name or dest.exists() or dest.is_relative_to(ROOT.resolve()):
        fail('output must be new, private, and outside the repository')
    for item in inputs:
        p = item.resolve(strict=True)
        if dest == p or dest.is_relative_to(p) or p.is_relative_to(dest):
            fail('output overlaps an input')
    return dest


def build(args):
    base_sha = valid_sha(args.base_sha256)
    dest = check_out(args.out, args.base_zip)
    base, raw, _ = baseline(args.base_zip, base_sha)
    commit, payload = get_git_files(args.ref)
    missing = old_routes(base) - payload.keys()
    if missing:
        fail('old routes missing from Git snapshot: ' + ', '.join(sorted(missing)))
    update = {'schema': SCHEMA, 'base_zip_sha256': base_sha,
              'base_engine_manifest_sha256': record(raw)[1],
              'base_code_commit': base.get('code_commit'), 'frontend_commit': commit,
              'files': {name: record(data) for name, data in sorted(payload.items())}}
    with tempfile.TemporaryDirectory(prefix='.qolda-update-', dir=dest.parent) as tmp:
        candidate = Path(tmp) / 'update.zip'
        with ZipFile(candidate, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr('update.json', json_bytes(update))
            for name, data in sorted(payload.items()):
                z.writestr('files/' + name, data)
        verify_update(candidate, sha_file(candidate), base, raw, base_sha)
        sha = sha_file(candidate)
        candidate.rename(dest)
    print(json.dumps({'update': str(dest), 'sha256': sha, 'bytes': dest.stat().st_size,
                      'frontend_commit': commit, 'files': len(payload), 'base_zip_sha256': base_sha}, indent=2))


def verify_update(path, expected_sha, base, base_raw, base_sha):
    if sha_file(path) != valid_sha(expected_sha):
        fail('update ZIP SHA-256 mismatch')
    with ZipFile(path) as z:
        infos = inventory(z)
        if 'update.json' not in infos or infos['update.json'].file_size > 4 * 1024 * 1024:
            fail('update metadata missing or too large')
        meta = read_json(z.read('update.json'))
        if (meta.get('schema') != SCHEMA or meta.get('base_zip_sha256') != base_sha or
                meta.get('base_engine_manifest_sha256') != record(base_raw)[1] or
                meta.get('base_code_commit') != base.get('code_commit') or
                not re.fullmatch(r'[0-9a-f]{40,64}', str(meta.get('frontend_commit', ''))) or
                not isinstance(meta.get('files'), dict)):
            fail('update metadata/base mismatch')
        files = meta['files']
        total = 0
        for name, value in files.items():
            safe_name(name)
            if not allowed(name):
                fail('update path not allowed: ' + name)
            size, _ = valid_record(value)
            total += size
            if size > MAX_FILE:
                fail('oversized update member')
        if total > MAX_TOTAL or 'frontend/index.html' not in files or old_routes(base) - files.keys():
            fail('update is oversized or drops old routes')
        if set(infos) != {'update.json'} | {'files/' + x for x in files}:
            fail('unexpected update ZIP members')
        payload = {}
        for name, value in files.items():
            info = infos['files/' + name]
            if info.file_size != value[0]:
                fail('update member size mismatch: ' + name)
            data = z.read(info)
            if record(data) != value:
                fail('update member SHA-256 mismatch: ' + name)
            payload[name] = data
    return meta, payload


def is_link(entry):
    st = entry.stat(follow_symlinks=False)
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, 'st_file_attributes', 0) & 0x400)


def source_inventory(root):
    result = {}
    def walk(directory, prefix=''):
        with os.scandir(directory) as entries:
            for entry in entries:
                rel = safe_name(prefix + entry.name)
                if is_link(entry):
                    fail('links/reparse points forbidden in base copy: ' + rel)
                if entry.is_dir(follow_symlinks=False):
                    walk(entry.path, rel + '/')
                elif entry.is_file(follow_symlinks=False):
                    result[rel] = Path(entry.path)
                else:
                    fail('non-regular base file: ' + rel)
    walk(root)
    if len({x.casefold() for x in result}) != len(result):
        fail('base copy has case-colliding paths')
    return result


def copy_checked(src, dst, expected):
    h = hashlib.sha256()
    size = 0
    if dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('rb') as inp:
        out = dst.open('xb') if dst is not None else None
        try:
            for chunk in iter(lambda: inp.read(1024 * 1024), b''):
                size += len(chunk)
                h.update(chunk)
                if out is not None:
                    out.write(chunk)
        finally:
            if out is not None:
                out.close()
    if [size, h.hexdigest()] != expected:
        fail('base copy differs from trusted ZIP: ' + str(src))


def apply(args):
    if os.name != 'nt':
        fail('atomic no-overwrite publication is supported only on Windows')
    if not args.confirm_stopped:
        fail('operator must confirm the base copy is not running')
    base_dir = args.base_dir.resolve(strict=True)
    if not base_dir.is_dir() or is_link(args.base_dir):
        fail('base copy must be a real directory')
    dest = check_out(args.out, args.base_zip, args.update, base_dir)
    base_sha = valid_sha(args.base_sha256)
    base, raw, expected = baseline(args.base_zip, base_sha)
    meta, payload = verify_update(args.update, args.update_sha256, base, raw, base_sha)
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', args.frontend_commit) or meta['frontend_commit'] != args.frontend_commit:
        fail('update frontend commit does not match the independently approved commit')
    sources = source_inventory(base_dir)
    if set(sources) != set(expected):
        fail('base copy inventory differs from trusted ZIP')
    new_manifest = dict(base)
    new_manifest['files'] = dict(base['files'])
    new_manifest['frontend_commit'] = meta['frontend_commit']
    new_manifest['frontend_update_sha256'] = args.update_sha256
    for name, data in payload.items():
        new_manifest['files']['app/' + name] = record(data)
    manifest_raw = json_bytes(new_manifest)
    final = dict(expected)
    final['engine-win64/manifest.json'] = record(manifest_raw)
    for name, data in payload.items():
        final['engine-win64/app/' + name] = record(data)
    with tempfile.TemporaryDirectory(prefix='.qolda-apply-', dir=dest.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        for name, src in sources.items():
            replace = name == 'engine-win64/manifest.json' or name.startswith('engine-win64/app/') and name[len('engine-win64/app/'):] in payload
            copy_checked(src, None if replace else stage / name, expected[name])
        for name, data in payload.items():
            target = stage / 'engine-win64/app' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (stage / 'engine-win64/manifest.json').write_bytes(manifest_raw)
        actual = source_inventory(stage)
        if set(actual) != set(final):
            fail('staged inventory mismatch')
        for name, path in actual.items():
            copy_checked(path, None, final[name])
        stage.rename(dest)  # Windows rename fails rather than replacing an existing path.
    print(json.dumps({'output': str(dest), 'frontend_commit': meta['frontend_commit'],
                      'base_zip_sha256': base_sha, 'update_sha256': args.update_sha256,
                      'engine_manifest_sha256': record(manifest_raw)[1],
                      'verified_files': len(final), 'base_code_commit': base.get('code_commit')}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build', help='create a data-only update from an explicit Git ref')
    b.add_argument('--ref', required=True)
    b.add_argument('--base-zip', required=True, type=Path)
    b.add_argument('--base-sha256', required=True)
    b.add_argument('--out', required=True, type=Path)
    a = sub.add_parser('apply', help='publish a verified NEW offline copy; never edit the base')
    a.add_argument('--base-zip', required=True, type=Path)
    a.add_argument('--base-sha256', required=True)
    a.add_argument('--base-dir', required=True, type=Path)
    a.add_argument('--update', required=True, type=Path)
    a.add_argument('--update-sha256', required=True)
    a.add_argument('--frontend-commit', required=True, help='independently approved full commit ID')
    a.add_argument('--out', required=True, type=Path)
    a.add_argument('--confirm-stopped', action='store_true')
    args = p.parse_args()
    try:
        (build if args.action == 'build' else apply)(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
