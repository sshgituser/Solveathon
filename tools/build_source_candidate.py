"""Make a local, history-free jury source snapshot from an explicit Git commit.

Never creates a repository or publishes anything. Copies only an allowlisted
product tree, and records every payload hash. Human rights/access review is a
separate gate; heuristic secret checks are not a full security audit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {'assets', 'frontend', 'desktop', 'android', 'backend', 'ml', 'tools', 'fixtures', 'artifacts'}
EXACT = {'docs/DATA-CARD.md', 'docs/MODEL-EVALUATION.md', 'docs/JUDGING.md',
         'docs/contracts/api.md', 'docs/contracts/ui.md', 'docs/contracts/data.md', 'docs/contracts/live-camera.md',
         'reports/data-audit.json', 'reports/ml/gpu-handoff/M0-M1-report.md',
         'reports/ml/gpu-handoff/M3-web.md', 'reports/ml/gpu-handoff/M4.1-android-device.md',
         'reports/ml/gpu-handoff/T-017-android-gpu.md', 'reports/ml/gpu-handoff/T-018-library.md',
         'reports/security/prototype-audit.md',
         'docs/model-intake-bc.md', 'reports/ml/model-intake/corpus-c-hand-sources.md',
         'reports/ml/model-intake/audit.json', 'reports/platforms/web-atlas-loopback.md',
         'reports/platforms/windows-candidate.md', 'reports/platforms/windows-relocated-engine.md',
         'reports/platforms/android-candidate.md'}
BINARY = {'assets/fonts/IBMPlexSans.ttf', 'assets/fonts/SourceSerif4.ttf', 'assets/models/mannequin.glb'}
TEXT = {'.js','.jsx','.mjs','.cjs','.kts','.gradle','.kt','.java','.c','.cpp','.cc','.h','.hpp','.xml','.bat','.cmd','.ps1','.css','.html','.md','.txt','.json','.toml','.py','.sh','.svg','.properties','.vtt','.yaml','.yml','.pro','.cmake'}
DENY_DIRS = {'.git','.gradle','.cxx','build','dist','node_modules','__pycache__','.venv','.idea'}
SENSITIVE = [re.compile(x) for x in (rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
             rb'\bgh[pousr]_[A-Za-z0-9]{32,}',rb'\bAKIA[0-9A-Z]{16}\b',rb'\bsk-(?:proj-)?[A-Za-z0-9_\-]{36,}',
             rb'(?i)https?://[^/\s:@]{1,100}:[^/\s@]{4,100}@')]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def selected(path):
    p = PurePosixPath(path)
    return p.parts[0] in ROOTS or path in EXACT


def validate(path, data):
    p = PurePosixPath(path)
    if p.is_absolute() or (set(p.parts) & DENY_DIRS or any(x.startswith('.') for x in p.parts[:-1])) or '..' in p.parts or p.name in {'local.properties','.env','key.properties'}:
        raise ValueError('forbidden candidate path: ' + path)
    if path in BINARY:
        expected = b'glTF' if p.suffix == '.glb' else b'\x00\x01\x00\x00'
        if not data.startswith(expected):
            raise ValueError('asset signature mismatch: ' + path)
        return
    if path in {'tools/android-engine/gradle/wrapper/gradle-wrapper.jar', 'android/gradle/wrapper/gradle-wrapper.jar'}:
        if hashlib.sha256(data).hexdigest() != 'cb0da6751c2b753a16ac168bb354870ebb1e162e9083f116729cec9c781156b8':
            raise ValueError('not the approved official Gradle 8.7 wrapper: ' + path)
        return
    if p.suffix.lower() not in TEXT | {'.cs','.csproj'} and p.name not in {'gradlew','LICENSE','NOTICE','.gitignore','.gitattributes','CMakeLists.txt'}:
        raise ValueError('unreviewed type; review explicitly before inclusion: ' + path)
    if b'\x00' in data or any(expr.search(data) for expr in SENSITIVE):
        raise ValueError('binary or credential-shaped bytes (value withheld): ' + path)


def make(ref, target):
    if target.exists() or target.resolve().is_relative_to(ROOT):
        raise ValueError('select a new private output ZIP outside the worktree')
    commit = git('rev-parse','--verify',ref+'^{commit}').decode().strip()
    names = [x.decode('utf-8') for x in git('ls-tree','-r','-z','--name-only',commit).split(b'\0') if x]
    chosen = sorted(n for n in names if selected(n))
    if len(chosen) != len({p.casefold() for p in chosen}):
        raise ValueError('Windows path collision')
    archive = git('archive','--format=zip',commit,'--',*sorted(ROOTS.intersection(p.split('/')[0] for p in chosen)),
                  *sorted(p for p in chosen if p.split('/')[0] not in ROOTS))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        file = Path(tmp) / 'candidate.zip'
        hashes = {}
        with ZipFile(io.BytesIO(archive)) as src, ZipFile(file, 'x') as dest:
            def put(path, data):
                entry = ZipInfo(path, (2026,1,1,0,0,0)); entry.compress_type = ZIP_DEFLATED
                dest.writestr(entry, data)
                hashes[path] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            for path in chosen:
                e = src.getinfo(path)
                if (e.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('symlink not allowed: ' + path)
                data = src.read(e); validate(path, data); put(path, data)
            intro = f'''# Qolda / SerIQ E02 — source candidate\n\nSource revision `{commit}`. This snapshot contains only committed sources at that revision. Work-in-progress UI, uncommitted platform sources and private model packages are not captured. It is not proof of a final frontend, an APK build or full product acceptance. No Git history, private artifacts or external publication is included.\n\nFrom the extracted directory, `python -m backend.api --port 8765` serves the Web UI on http://127.0.0.1:8765/. Without the private model, analysis correctly reports model_unavailable. A real PC run requires the separate private Windows CUDA distribution or a local RTMPose export + trained A + strict runtime bundle; see `ml/requirements-gpu.txt`. No model weights, videos or individual participant records are distributed here.\n\n`docs/contracts/api.md` documents the API; `docs/DATA-CARD.md` describes the data and its limitations. `reports/ml/gpu-handoff/M3-web.md` and `reports/ml/gpu-handoff/M4.1-android-device.md` are measurements of specific builds, not blanket platform acceptance. The application describes observed motion, not pain, diagnosis or treatment.\n\nThird-party notices: `artifacts/seriq/ATTRIBUTION.md`, `assets/fonts/`, `assets/models/CREDITS.md` and `assets/vendor/three/LICENSE`. Research validation of one participant is not clinical validation.\n'''
            put('README.md', intro.encode('utf-8'))
            receipt = json.dumps({'schema':'qolda-source-candidate-v1','source_commit':commit,'files':hashes},
                                 ensure_ascii=False,sort_keys=True,indent=2).encode('utf-8')
            put('SOURCE-MANIFEST.json', receipt)
        with ZipFile(file) as z:
            if len(z.infolist()) != len(hashes) or set(z.namelist()) != set(hashes):
                raise ValueError('incorrect ZIP member set')
            for path, meta in hashes.items():
                with z.open(path) as f:
                    if z.getinfo(path).file_size != meta['bytes']:
                        raise ValueError('ZIP size mismatch: '+path)
                    checksum = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1 << 20), b''):
                        checksum.update(chunk)
                    if checksum.hexdigest() != meta['sha256']: raise ValueError('ZIP digest mismatch: '+path)
        file.rename(target)
    print(json.dumps({'commit':commit,'zip':target.name,'verified_members':len(hashes),'bytes':target.stat().st_size,
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'published':False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',required=True); parser.add_argument('--out',type=Path,required=True)
    opts = parser.parse_args(); make(opts.ref,opts.out)
