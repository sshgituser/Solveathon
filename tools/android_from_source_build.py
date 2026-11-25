"""Build and verify a locally pinned Qolda Android debug APK without altering the checkout."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
from platform import machine
import re
import runpy
import subprocess
from zipfile import ZipFile

from tools.android_from_source import (
    ABI, CHANNELS, CONVERTER, GRAPHS, LITERT_VERSION, POSE_POLICY, ROOT, UPSTREAM_LITERT,
    disk, inspect_fp32, private_dir, read_bundle, read_json, require, write_bytes, write_json, zip_member,
)

ENGINE = 'tools/android-engine'
APP_GRADLE = 'android/app/build.gradle.kts'
ENGINE_GRADLE = f'{ENGINE}/engine/build.gradle.kts'
RESULT = 'android/app/src/main/java/net/qolda/offline/Result.java'
VERIFIER = 'android/verify_apk.py'
SDK_NDK = '26.1.10909125'
SDK_BUILD_TOOLS = '34.0.0'
RUNTIMES = {'litert': 'libLiteRt.so', 'litert-gpu': 'libLiteRtClGlAccelerator.so'}
JNI = 'libqolda_engine.so'
PLATFORM_LIBS = set(RUNTIMES.values()) | {JNI}
LICENSING = ('LICENSE', 'THIRD_PARTY_NOTICE.txt')


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def command_text(*args, **kwargs):
    return run(*args, capture_output=True, text=True, encoding='utf-8', errors='replace', **kwargs).stdout


def prepare_runtime(packages, sdk, kit):
    for package, (_, expected) in UPSTREAM_LITERT.items():
        require(disk(packages / package)['sha256'] == expected, f'wrong upstream package: {package}')
    with ZipFile(packages / 'litert_cc_sdk.zip') as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            rel = PurePosixPath(entry.filename)
            require(rel.parts and rel.parts[0] == 'litert_cc_sdk' and '..' not in rel.parts and
                    ':' not in entry.filename and '\\' not in entry.filename, 'unexpected C++ SDK path')
            write_bytes(sdk / 'sdk' / Path(*rel.parts), archive.read(entry))
    # Release SDK omits the generated selector. Use upstream's GPU configuration from the same tag.
    write_bytes(sdk / 'sdk/litert_cc_sdk/litert/build_common/build_config.h', (packages / 'build_config_gpu.h').read_bytes())
    for header in ('litert_common.h', 'litert_compiled_model.h'):
        require((sdk / 'sdk/litert_cc_sdk/litert/c' / header).is_file(), f'missing upstream header {header}')
    libraries = {}
    for component, name in RUNTIMES.items():
        with ZipFile(packages / f'{component}-{LITERT_VERSION}.aar') as archive:
            data = zip_member(archive, f'jni/{ABI}/{name}')
            require_elf(data, name)
            sdk_pin = write_bytes(sdk / 'jni' / ABI / name, data)
            rel = f'jniLibs/{ABI}/{name}'
            require(write_bytes(kit / rel, data) == sdk_pin, f'{name}: SDK and staged runtime differ')
            libraries[name] = {'path': rel, **sdk_pin}
            for entry in LICENSING:
                filename = entry if entry.endswith('.txt') else entry + '.txt'
                write_bytes(kit / 'assets/qolda/licenses' / f'{component}-{filename}', zip_member(archive, entry))
    write_bytes(kit / 'assets/qolda/licenses/ATTRIBUTION.md', (ROOT / 'artifacts/seriq/ATTRIBUTION.md').read_bytes())
    return libraries


def require_elf(data, name):
    require(data[:6] == b'\x7fELF\x02\x01' and int.from_bytes(data[18:20], 'little') == 183,
            f'{name}: not ELF64 AArch64')


def check_symbol(readelf, library, symbol):
    text = command_text(readelf, '--dyn-syms', '--wide', library)
    require(any(re.search(r'\b(?:GLOBAL|WEAK)\s+(?:DEFAULT|PROTECTED)\s+(?!UND\b)\S+\s+'
                          + re.escape(symbol) + r'(?:@\S+)?(?:\s|$)', line) for line in text.splitlines()),
            f'{library.name}: missing defined dynamic export {symbol}')


def stage(bundle_path, conversion, packages, sdk, kit, readelf):
    bundle_pin = disk(bundle_path)
    bundle = read_bundle(bundle_path)
    receipt = read_json(conversion / 'conversion.json')
    require(receipt['schema'] == 'qolda-android-local-conversion-v1' and receipt['converter'] == CONVERTER and
            receipt['bundle'] == bundle_pin and set(receipt['graphs']) == set(GRAPHS),
            'conversion does not match its local source bundle')
    libraries = prepare_runtime(packages, sdk, kit)
    check_symbol(readelf, sdk / 'jni' / ABI / 'libLiteRt.so', 'LiteRtCompiledModelIsNonCpuFullyAccelerated')
    for name in RUNTIMES.values():
        require('[libc++_shared.so]' not in command_text(readelf, '-d', sdk / 'jni' / ABI / name),
                f'{name}: unexpectedly needs libc++_shared.so')
    graphs = {}
    for key, (stem, _, _, _) in GRAPHS.items():
        name = f'{stem}-fp32.tflite'
        record = receipt['graphs'][key]
        source = conversion / name
        expected = {k: record[k] for k in ('bytes', 'sha256')}
        require(record['file'] == name and disk(source) == expected, f'{key}: converted graph pin mismatch')
        data = source.read_bytes()
        spec = inspect_fp32(data, key)
        require(all(spec[k] == record[k] for k in spec), f'{key}: converted graph contract differs')
        rel = f'assets/qolda/{name}'
        require(write_bytes(kit / rel, data) == expected, f'{key}: staged graph pin mismatch')
        graphs[key] = {'path': rel, **expected, **spec}
    model = bundle['model']
    manifest = {
        'schema': 'qolda-offline-gpu-assets-v1', 'status': 'local_source_build_device_and_parity_unverified',
        'provenance': {'classifier_training_source_commit': bundle['source_commit'],
                       'local_bundle_sha256': bundle_pin['sha256'], 'classifier_weights_sha256': model['sha256'],
                       'training_manifest_sha256': bundle['training_manifest_sha256'],
                       'pose_upstream_sha256': bundle['pose']['original_sha256'],
                       'pose_b1_sha256': bundle['pose']['export_sha256'], 'converter': CONVERTER},
        'runtime': {'version': LITERT_VERSION, 'abi': ABI, 'requested_accelerator': 'GPU',
                    'require_full_delegation_before_run': True, 'gpu_options_toml': 'precision = 2\n',
                    'libraries': libraries},
        'graphs': graphs,
        'classifier': {'slot': 'scratch', 'id': 'A', 'model_version': f'{model["version"]}+{model["sha256"][:12]}',
                       'threshold': model['threshold'], 'uncertain_margin': 0.10,
                       'pose_policy': POSE_POLICY, 'channels': CHANNELS},
    }
    require(disk(bundle_path) == bundle_pin, 'bundle changed during staging')
    manifest_pin = write_json(kit / 'assets/qolda/engine.json', manifest)
    assets = {'qolda/engine.json': manifest_pin, **{
        graph['path'].removeprefix('assets/'): {k: graph[k] for k in ('bytes', 'sha256')}
        for graph in graphs.values()
    }}
    return manifest, assets


def source_allowed(rel):
    if rel in {APP_GRADLE, ENGINE_GRADLE, VERIFIER, f'{ENGINE}/engine/consumer-rules.pro',
               f'{ENGINE}/build.gradle.kts', 'frontend/index.html', 'assets/models/mannequin.glb',
               'assets/models/CREDITS.md'}:
        return True
    if any(rel == f'{project}/{name}' for project in ('android', ENGINE)
           for name in ('settings.gradle.kts', 'gradle.properties', 'gradle/libs.versions.toml')):
        return True
    if rel.startswith(('android/app/src/', f'{ENGINE}/engine/src/')):
        return Path(rel).suffix in {'.java', '.xml', '.js', '.h', '.cc', '.cpp'} or rel == (
            f'{ENGINE}/engine/src/main/cpp/CMakeLists.txt')
    if rel.startswith(('frontend/src/', 'frontend/styles/')):
        return Path(rel).suffix in {'.js', '.css'}
    if rel.startswith(('assets/brand/', 'assets/fonts/', 'assets/vendor/three/')):
        return Path(rel).suffix in {'.svg', '.ttf', '.txt', '.js'} or rel == 'assets/vendor/three/LICENSE'
    return False


def snapshot(destination):
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', 'android', ENGINE, 'frontend', 'assets'], cwd=ROOT)
    paths = [rel for rel in tracked.decode('utf-8').split('\0') if rel and source_allowed(rel)]
    require(len(paths) == len(set(paths)), 'unmerged build sources in Git index')
    records = {}
    for rel in paths:
        path = ROOT / rel
        require(path.resolve() == path and path.is_file(), f'missing or redirected source: {rel}')
        records[rel] = write_bytes(destination / rel, path.read_bytes())
    require({APP_GRADLE, ENGINE_GRADLE, RESULT, VERIFIER, 'frontend/index.html'} <= records.keys(),
            'missing tracked source project')
    return {'checkout_head': command_text('git', 'rev-parse', 'HEAD', cwd=ROOT).strip(),
            'files': records, 'tools': {path.name: disk(path) for path in sorted((ROOT / 'tools').glob('android_from_source*'))
                                       if path.is_file()}}


def replace_unique(text, regex, replacement):
    text, count = re.subn(regex, lambda _: replacement, text, flags=re.MULTILINE)
    require(count == 1, f'expected one source binding for {regex}; stop and review source revision')
    return text


def overlay_engine(workspace):
    path = workspace / ENGINE_GRADLE
    anchor = '            excludes += "**/libc++_shared.so"'
    update = anchor + '\n            keepDebugSymbols += listOf("**/libLiteRt.so", "**/libLiteRtClGlAccelerator.so")'
    # Preserve the authentic Maven libraries byte-for-byte in the AAR (and later APK).
    path.write_text(replace_unique(path.read_text('utf-8'), '^' + re.escape(anchor) + '$', update),
                    encoding='utf-8', newline='\n')


def bind_app(workspace, manifest, pins):
    path = workspace / APP_GRADLE
    text = path.read_text('utf-8')
    text = replace_unique(text, r'^val approvedAarSha = "[0-9a-f]{64}"$',
                          f'val approvedAarSha = "{pins["aar"]["sha256"]}"')
    text = replace_unique(text, r'(?<=engineAar\.length\(\) == )[0-9]+L', f'{pins["aar"]["bytes"]}L')
    text = replace_unique(text, re.escape('T-018 AAR does not match the approved release.'),
                          'AAR does not match the verified pins of this local build.')
    for rel, expected in pins['assets'].items():
        text = replace_unique(text, rf'^    "{re.escape(rel)}" to "[0-9a-f]{{64}}",$',
                              f'    "{rel}" to "{expected["sha256"]}",')
    path.write_text(text, encoding='utf-8', newline='\n')
    path = workspace / RESULT
    text = path.read_text('utf-8')
    model = manifest['classifier']
    # Keep the threshold/margin semantics of ml/api.py, without keeping another run's model identity.
    text = replace_unique(text, r'^    static final String MODEL_VERSION = "[A-Za-z0-9._+-]+";$',
                          f'    static final String MODEL_VERSION = "{model["model_version"]}";\n'
                          f'    private static final double THRESHOLD = {model["threshold"]};')
    text = replace_unique(text, r'^        double confidence = Math\.abs\(p - \.5\) \* 2;$',
                          '        double confidence = p >= THRESHOLD ? (p - THRESHOLD) / (1 - THRESHOLD)\n'
                          '                : (THRESHOLD - p) / THRESHOLD;')
    text = replace_unique(text, r'(?<=return new Result\(p >= )\.5(?= \? "complete")', 'THRESHOLD')
    path.write_text(text, encoding='utf-8', newline='\n')


def verify_aar(aar, manifest, scratch, readelf):
    with ZipFile(aar) as archive:
        require(all(n == 1 for n in Counter(archive.namelist()).values()), 'duplicate AAR member')
        require({name for name in archive.namelist() if name.startswith('jni/') and name.endswith('.so')} ==
                {f'jni/{ABI}/{name}' for name in PLATFORM_LIBS}, 'unexpected AAR native libraries')
        with ZipFile(BytesIO(zip_member(archive, 'classes.jar'))) as jar:
            for name in ('GpuEngine', 'EngineException'):
                zip_member(jar, f'net/qolda/engine/{name}.class')
        for name in sorted(PLATFORM_LIBS):
            data = zip_member(archive, f'jni/{ABI}/{name}')
            require_elf(data, name)
            pin = write_bytes(scratch / name, data)
            if name in manifest['runtime']['libraries']:
                record = manifest['runtime']['libraries'][name]
                require(pin == {k: record[k] for k in ('sha256', 'bytes')}, f'{name}: AAR modified the pinned runtime')
        for method in ('nativeOpen', 'nativeInferPose', 'nativeInferLogit', 'nativeClose'):
            check_symbol(readelf, scratch / JNI, 'Java_net_qolda_engine_GpuEngine_' + method)
        require('[libc++_shared.so]' not in command_text(readelf, '-d', scratch / JNI), 'JNI must use static libc++')


def verify_package(workspace, apk, aar, pins, java, aapt, apksigner):
    verify = runpy.run_path(str(workspace / VERIFIER))['verify']

    def cli(tool, *args):
        if tool == apksigner:
            return command_text(java, '-jar', apksigner, *args)
        return command_text(tool, *args)

    # Rebind only evidence pins and CLI process launching; preserve all verification rules.
    verify.__globals__.update(APPROVED_AAR=pins['aar'], ENGINE_SHA=pins['assets']['qolda/engine.json']['sha256'], cli=cli)
    return verify(apk, aar, aapt, apksigner, require_frontend=True)


def build(bundle, conversion, litert, gradle, destination):
    require(os.name == 'nt' and machine().lower() in {'amd64', 'x86_64'}, 'the recipe requires 64-bit Windows')
    java_home = Path(os.environ['JAVA_HOME']).resolve()
    android_home = Path(os.environ['ANDROID_HOME']).resolve()
    gradle = gradle.resolve()
    java, keytool = (java_home / 'bin' / (name + '.exe') for name in ('java', 'keytool'))
    build_tools = android_home / 'build-tools' / SDK_BUILD_TOOLS
    aapt, apksigner = build_tools / 'aapt.exe', build_tools / 'lib/apksigner.jar'
    readelf = android_home / 'ndk' / SDK_NDK / 'toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-readelf.exe'
    launcher = gradle / 'lib/gradle-launcher-8.7.jar'
    for path in (java, keytool, aapt, apksigner, readelf, launcher, android_home / 'platforms/android-34/android.jar',
                 android_home / 'cmake/3.22.1/bin/cmake.exe'):
        require(path.is_file(), f'missing pinned toolchain file: {path}')
    jdk = run(java, '-version', capture_output=True, text=True).stderr
    require(re.search(r' version "17\.', jdk), 'JAVA_HOME must point to JDK 17')
    out = private_dir(destination)
    workspace, sdk, kit = out / 'src', out / 'litert-2.4.0', out / 'inputs'
    manifest, assets = stage(bundle, conversion, litert, sdk, kit, readelf)
    qolda = kit / 'assets/qolda'
    licenses = {f'assets/qolda/licenses/{path.name}': disk(path) for path in sorted((qolda / 'licenses').iterdir())}
    originals = snapshot(workspace)
    overlay_engine(workspace)
    android_user = out / 'android-user'
    android_user.mkdir()
    env = {**os.environ, 'JAVA_HOME': str(java_home), 'ANDROID_HOME': str(android_home),
           'ANDROID_SDK_ROOT': str(android_home), 'ANDROID_USER_HOME': str(android_user),
           'GRADLE_USER_HOME': str(out / 'gradle-user')}
    env.pop('ANDROID_SDK_HOME', None)  # Never pull in a previous station's user profile or key.
    gradlew = [java, '-Dorg.gradle.appname=gradle', '-classpath', launcher, 'org.gradle.launcher.GradleMain',
               '--no-daemon', '--console=plain', '--no-build-cache', '--no-watch-fs']
    run(*gradlew, ':engine:assembleRelease', f'-PqoldaEngineKit={kit}', f'-PqoldaLiteRtSdk={sdk}',
        env=env, cwd=workspace / ENGINE)
    aar = workspace / ENGINE / 'engine/build/outputs/aar/engine-release.aar'
    aar_pin = disk(aar)
    verify_aar(aar, manifest, out / 'aar-elf', readelf)

    pins = {'schema': 'qolda-android-local-pins-v1', 'aar': aar_pin, 'assets': assets, 'licenses': licenses,
            'runtime': manifest['runtime']['libraries']}
    require(disk(aar) == aar_pin and all(disk(kit / 'assets' / rel) == pin for rel, pin in assets.items()),
            'AAR or staged assets changed before pinning')
    write_json(out / 'pins.json', pins)
    bind_app(workspace, manifest, pins)
    originals['local_overlays'] = {rel: disk(workspace / rel) for rel in (ENGINE_GRADLE, APP_GRADLE, RESULT)}
    write_json(out / 'source-pins.json', originals)
    run(*gradlew, ':app:assembleDebug', f'-PqoldaEngineAar={aar}', f'-PqoldaEngineKit={kit}',
        env=env, cwd=workspace / 'android')
    apk = workspace / 'android/app/build/outputs/apk/debug/app-debug.apk'
    receipt = verify_package(workspace, apk, aar, pins, java, aapt, apksigner)
    with ZipFile(apk) as archive:
        require(set(receipt['licenses']) == licenses.keys(), 'unexpected/missing APK licenses')
        for name, expected in licenses.items():
            # zip_member also rejects duplicates, independently of the base verifier.
            data = zip_member(archive, name)
            require({'bytes': len(data), 'sha256': sha256(data).hexdigest()} == expected, f'{name}: changed license')
    debug_key = android_user / 'debug.keystore'
    require(debug_key.is_file(), 'APK must use a fresh debug key under the job directory')
    cert_log = command_text(keytool, '-J-Duser.language=en', '-J-Duser.country=US', '-list', '-v',
                            '-keystore', debug_key, '-storepass', 'android', '-alias', 'androiddebugkey')
    cert = re.search(r'SHA256:\s+([A-Fa-f0-9:]+)', cert_log)
    require(cert is not None and cert[1].replace(':', '').lower() == receipt['signer_sha256'],
            'APK signer differs from the job-local debug certificate')
    require(disk(apk) == receipt['apk'] and disk(aar) == aar_pin, 'APK or AAR changed after verification')
    receipt.update(schema='qolda-android-local-apk-v1', model_threshold=manifest['classifier']['threshold'],
                   local_pins=disk(out / 'pins.json'), local_sources=disk(out / 'source-pins.json'),
                   installed=False, gpu_verified=False, parity_verified=False, e2e_verified=False)
    write_json(out / 'apk-receipt.json', receipt)
    for path in (aar, apk, out / 'pins.json', out / 'source-pins.json', out / 'apk-receipt.json'):
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('bundle', 'conversion', 'litert', 'gradle', 'out'):
        parser.add_argument('--' + arg, required=True, type=Path)
    args = parser.parse_args()
    build(args.bundle, args.conversion, args.litert, args.gradle, args.out)


if __name__ == '__main__':
    main()
