"""Verify the signed product APK's actual inputs, ABI and merged permissions."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile


ENGINE_JSON = 'assets/qolda/engine.json'
ENGINE_SHA = '754191c53b30b7880b2897b786a328643ceabeec38950e256e9225341a88bfa7'
APPROVED_AAR = {'bytes': 4224419, 'sha256': '4e82863a8ea5e7c1ecedbdbce604e3e8f91b748113a33948459caebc462e4022'}
ABI = 'arm64-v8a'
LIBS = ('libqolda_engine.so', 'libLiteRt.so', 'libLiteRtClGlAccelerator.so')
WEB = 'assets/web/'


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def record(stream, size):
    return {'bytes': size, 'sha256': file_digest(stream, 'sha256').hexdigest()}


def member(archive, name):
    with archive.open(name) as stream:
        return record(stream, archive.getinfo(name).file_size)


def disk(path):
    with path.open('rb') as stream:
        return record(stream, path.stat().st_size)


def cli(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace').stdout


def verify(apk_path, aar_path, aapt, apksigner, require_frontend=False):
    aar_record = disk(aar_path)
    require(aar_record == APPROVED_AAR, 'Unapproved T-018 AAR')
    permissions = cli(aapt, 'dump', 'permissions', str(apk_path)).splitlines()
    require(permissions == ['package: net.qolda.offline', "uses-permission: name='android.permission.CAMERA'"],
            f'Unexpected merged permissions: {permissions}')
    badging = cli(aapt, 'dump', 'badging', str(apk_path))
    require("name='net.qolda.offline' versionCode='1' versionName='0.1.0-t010'" in badging, 'Not the T-010 product package')
    require("native-code: 'arm64-v8a'" in badging, 'Not an ARM64-only product')
    signature = cli(apksigner, 'verify', '--verbose', '--print-certs', str(apk_path))
    cert = re.search(r'certificate SHA-256 digest: ([a-fA-F0-9]{64})', signature)
    if cert is None:
        raise SystemExit('Signer certificate digest not found')
    with ZipFile(apk_path) as apk, ZipFile(aar_path) as aar:
        names = Counter(apk.namelist())
        require(all(count == 1 for count in names.values()), 'Duplicate resource path in APK')
        engine = member(apk, ENGINE_JSON)
        require(engine['sha256'] == ENGINE_SHA, 'Unrecognized engine manifest')
        manifest = json.loads(apk.read(ENGINE_JSON))
        expected_models = {graph['path'] for graph in manifest['graphs'].values()}
        require({name for name in names if name.endswith('.tflite')} == expected_models, 'Missing or extra graph')
        require(not any(name.endswith(('.f32', '.mp4', '.webm', '.npz')) for name in names), 'Unexpected private sample/diagnostic material')
        entry = apk.read(WEB + 'frontend/index.html').decode('utf-8')
        require(all(x in entry for x in ('content="android"', 'content="ondevice"', '../android/bridge.js')),
                'Android entry point is missing profile or bridge injection')
        require('http://127.0.0.1:' not in entry, 'Entry policy still permits loopback')
        bridge = member(apk, WEB + 'android/bridge.js')
        for path in ('welcome.js', 'home.js', 'check.js', 'progress.js', 'guide.js', 'cabinet.js'):
            require(WEB + 'frontend/src/views/' + path in names, f'Common view missing: {path}')
        for path in ('assets/models/mannequin.glb', 'assets/fonts/IBMPlexSans.ttf', 'assets/vendor/three/three.module.min.js'):
            require(WEB + path in names, f'Common asset missing: {path}')
        js = [apk.read(name).decode('utf-8') for name in names if name.startswith(WEB + 'frontend/src/') and name.endswith('.js')]
        # Diagnostic, not a runtime/UI proof; the typed bridge is separate from bundled common source.
        references = [src for src in js if 'QoldaAndroid' in src]
        front = {'entry': member(apk, WEB + 'frontend/index.html'), 'bridge': bridge,
                 'common_javascript_files': len(js), 'common_references_native_bridge': bool(references)}
        if require_frontend:
            require(front['common_references_native_bridge'], 'Common frontend has not wired QoldaAndroid; only a technical build')
        native = {f'lib/{ABI}/{lib}' for lib in LIBS}
        require({name for name in names if name.startswith('lib/') and name.endswith('.so')} == native, 'Unexpected/missing APK native libraries')
        licenses = sorted(name for name in names if name.startswith('assets/qolda/licenses/') and not name.endswith('/'))
        require(licenses, 'Attribution/license assets missing')
        models, libs = {}, {}
        for name, graph in manifest['graphs'].items():
            item = member(apk, graph['path'])
            require(item == {k: graph[k] for k in ('bytes', 'sha256')}, f'{name} does not match engine manifest')
            models[name] = item
        for name in LIBS:
            path = f'lib/{ABI}/{name}'
            with apk.open(path) as stream:
                header = stream.read(20)
            require(header[:6] == b'\x7fELF\x02\x01' and int.from_bytes(header[18:20], 'little') == 183, f'{name} is not ELF64 AArch64')
            item = member(apk, path)
            require(item == member(aar, f'jni/{ABI}/{name}'), f'{name} differs from AAR')
            if name in manifest['runtime']['libraries']:
                expected = manifest['runtime']['libraries'][name]
                require(item == {k: expected[k] for k in ('bytes', 'sha256')}, f'{name} differs from pinned runtime')
            libs[name] = item
    return {'schema': 'qolda-android-apk-v1', 'package': 'net.qolda.offline', 'version_code': 1,
            'version_name': '0.1.0-t010', 'model_version': manifest['classifier']['model_version'],
            'abi': ABI, 'permissions': ['android.permission.CAMERA'], 'apk': disk(apk_path), 'aar': aar_record,
            'signer_sha256': cert[1].lower(), 'engine_manifest': engine, 'models': models, 'frontend': front,
            'libraries': libs, 'licenses': licenses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apk', type=Path, required=True)
    parser.add_argument('--aar', type=Path, required=True)
    parser.add_argument('--aapt', required=True)
    parser.add_argument('--apksigner', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--require-wired-frontend', action='store_true')
    args = parser.parse_args()
    evidence = verify(args.apk, args.aar, args.aapt, args.apksigner, args.require_wired_frontend)
    text = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
