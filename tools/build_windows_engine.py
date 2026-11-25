"""Build Qolda's private relocatable Windows/CUDA engine from approved local inputs.

Run with the pinned Python 3.11 GPU venv. No downloads, training data, user-site
packages, registry edits or services. This is a private distribution, not a
public submission archive. The NVIDIA display driver stays a host prerequisite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WHEELS = ('torch', 'onnxruntime-gpu', 'onnx', 'numpy', 'opencv-contrib-python', 'safetensors',
          'protobuf', 'ml_dtypes', 'typing_extensions', 'packaging', 'flatbuffers', 'filelock',
          'fsspec', 'networkx', 'sympy', 'mpmath', 'Jinja2', 'MarkupSafe', 'setuptools')
MODULES = {'backend': ('api', 'runtime', 'contract', 'live'), 'ml': ('api', 'data', 'extract', 'gpu', 'models', 'scoring')}


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def wheel_files(site, dest):
    versions = {}
    for name in WHEELS:
        dist = metadata.distribution(name)
        versions[name] = dist.version
        if not dist.files or Path(str(dist.locate_file(''))).resolve() != site:
            raise ValueError('package outside the selected venv or missing wheel manifest: ' + name)
        for entry in dist.files:
            p = Path(entry)
            # Wheel script launchers reference the builder's Python. We only need imports.
            if (p.is_absolute() or '..' in p.parts or '__pycache__' in p.parts or
                    p.suffix.lower() in {'.pyc', '.pyo', '.pth'} or p.name == 'direct_url.json'):
                continue
            src = (site / p).resolve()
            if not src.is_relative_to(site):
                raise ValueError('package file outside the venv: ' + name)
            if not src.is_file():
                raise FileNotFoundError(p)
            copy(src, dest / p)
    return versions


def application(commit, app):
    paths = ['frontend', 'assets', 'artifacts/seriq/ATTRIBUTION.md', 'artifacts/seriq/LICENSE-SLiM.txt']
    paths += [f'{pkg}/{name}.py' for pkg, names in MODULES.items() for name in names]
    # Gate optional modules on the pinned tree, not on the moving worktree. This
    # keeps historical E02 builds reproducible and never picks up pending heads.
    optional = ('backend/extra_exercises.py', 'ml/extra_e01.py', 'ml/extra_e03.py', 'ml/extra_e03_service.py')
    present = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', *optional], cwd=ROOT, text=True)
    paths.extend(name for name in present.splitlines() if name in optional)
    archive = subprocess.check_output(['git', 'archive', '--format=zip', commit, '--', *paths], cwd=ROOT)
    with ZipFile(io.BytesIO(archive)) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            f = app / entry.filename
            if not f.resolve().is_relative_to(app.resolve()):
                raise ValueError('invalid Git archive path')
            f.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as inp, f.open('xb') as out:
                shutil.copyfileobj(inp, out)


def build(args):
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 11):
        raise ValueError('use the project Windows Python 3.11 CUDA venv')
    dest = args.out.resolve()
    if dest.is_relative_to(ROOT) or dest.exists():
        raise ValueError('select a new private output directory outside Git')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', args.ref + '^{commit}'], cwd=ROOT, text=True).strip()
    site = Path(sysconfig.get_paths()['purelib']).resolve()
    base = Path(sys.base_prefix)
    descriptor = json.loads(args.bundle.read_text(encoding='utf-8'))
    m, p = descriptor['model'], descriptor['pose']
    if (descriptor['schema'], m['id'], m['variant'], m.get('slot'), descriptor['runtime']['device_class']) != (
            'seriq-bundle-v1', 'A', 'A', 'scratch', 'nvidia-cuda') or m.get('base_sha256'):
        raise ValueError('expected a standalone A/scratch CUDA delivery')
    if Path(m['file']).name != m['file'] or m['file'] != 'A-tcn.safetensors':
        raise ValueError('unexpected A filename')
    models = [(args.pose, 'end2end.onnx', p['original_sha256']),
              (args.bundle.parent / 'end2end.b1.onnx', 'end2end.b1.onnx', p['export_sha256']),
              (args.weights, m['file'], m['sha256'])]
    for src, name, checksum in models:
        if digest(src) != checksum:
            raise ValueError('model input digest mismatch: ' + name)
    ffprobe = args.ffprobe.resolve(strict=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completely copied and hashed tree. No symlinks to the builder.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.qolda-stage-') as tmp:
        out = Path(tmp) / 'engine'
        py = out / 'python'
        for name in ('python.exe', 'python311.dll', 'python3.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'LICENSE.txt'):
            copy(base / name, py / name)
        ignored = shutil.ignore_patterns('site-packages', '__pycache__', '*.pyc', 'test', 'idlelib', 'ensurepip', 'tkinter', 'turtledemo')
        shutil.copytree(base / 'Lib', py / 'Lib', ignore=ignored)
        shutil.copytree(base / 'DLLs', py / 'DLLs', ignore=shutil.ignore_patterns('*_test*', '*tk*', '*tcl*'))
        versions = wheel_files(site, py / 'Lib/site-packages')
        (py / 'python311._pth').write_text('.\nLib\nDLLs\nLib/site-packages\n../app\n', encoding='ascii')
        application(commit, out / 'app')
        for src, name, checksum in models:
            target = out / 'runtime' / name
            copy(src, target)
            if digest(target) != checksum:
                raise OSError('bad model readback: ' + name)
        copy(args.bundle, out / 'runtime/bundle.json')
        copy(ffprobe, out / 'runtime/ffprobe.exe')
        copy(args.ffmpeg_license, out / 'licenses/FFmpeg-LICENSE.txt')
        cmd = ('@echo off\nsetlocal\nset "HOMEKIT=%~dp0"\n'
               '"%HOMEKIT%python\\python.exe" -B -m backend.api --port 8765 '
               '--bundle "%HOMEKIT%runtime\\bundle.json" --artifacts "%HOMEKIT%runtime" '
               '--pose-model "%HOMEKIT%runtime\\end2end.onnx" --ffprobe "%HOMEKIT%runtime\\ffprobe.exe" '
               '--private-scratch "%LOCALAPPDATA%\\Qolda\\scratch"\nexit /b %ERRORLEVEL%\n')
        (out / 'start-engine.cmd').write_text(cmd, encoding='ascii', newline='\r\n')
        manifest = {'schema': 'qolda-private-windows-engine-v1', 'code_commit': commit,
                    'model_source_commit': descriptor['source_commit'], 'platform': 'win-amd64',
                    'python': sys.version.split()[0], 'wheel_versions': versions, 'files': {}}
        for file in sorted(out.rglob('*')):
            if file.is_file():
                manifest['files'][file.relative_to(out).as_posix()] = [file.stat().st_size, digest(file)]
        manifest_name = out / 'manifest.json'
        manifest_name.write_text(json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
        checksum = digest(manifest_name)
        files = manifest['files']
        out.rename(dest)
    print(json.dumps({'files_excluding_manifest': len(files), 'bytes_excluding_manifest': sum(v[0] for v in files.values()),
                      'manifest_sha256': checksum, 'code_commit': commit}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('bundle', 'pose', 'weights', 'ffprobe', 'ffmpeg-license', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--ref', default='HEAD')
    build(parser.parse_args())
