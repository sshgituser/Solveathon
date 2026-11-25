"""Read back only the installed Qolda package and its allowlisted receipt lines."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


PACKAGE = 'net.qolda.offline'


def collect(adb, manifest, serial=None):
    base = [adb] + (['-s', serial] if serial else [])

    def cmd(*args, check=True):
        result = subprocess.run(base + list(args), capture_output=True, encoding='utf-8', errors='replace', check=check)
        return result.stdout.strip()

    pkg = cmd('shell', 'dumpsys', 'package', PACKAGE)
    version = re.search(r'\bversionCode=(\d+)', pkg)
    name = re.search(r'\bversionName=([^\s]+)', pkg)
    package_paths = [line.removeprefix('package:') for line in cmd('shell', 'pm', 'path', PACKAGE).splitlines()
                     if line.startswith('package:') and line.endswith('/base.apk')]
    if len(package_paths) != 1 or not version or not name:
        raise SystemExit('Qolda product package is not installed or has an unexpected shape')
    actual = cmd('shell', 'sha256sum ' + shlex.quote(package_paths[0])).split()[0].lower()
    if (actual != manifest['apk']['sha256'] or int(version[1]) != manifest['version_code']
            or name[1] != manifest['version_name']):
        raise SystemExit('Installed APK is not the verified file')
    facts = {'schema': 'qolda-android-device-v1', 'package': PACKAGE,
             'version_code': int(version[1]), 'version_name': name[1], 'installed_apk_sha256': actual,
             'device': {'model': cmd('shell', 'getprop', 'ro.product.model'),
                        'sdk': int(cmd('shell', 'getprop', 'ro.build.version.sdk')),
                        'abi': cmd('shell', 'getprop', 'ro.product.cpu.abi')},
             'offline_without_cable_verified_by_this_receipt': False,
             'runs': [], 'delegation': {}, 'web_document_ready': False, 'main_frame_bridge_ready': False,
             'both_graphs_full_gpu_observed': False}
    for key in ('firstInstallTime', 'lastUpdateTime'):
        item = re.search(r'\b' + key + r'=([\d\-: ]+)', pkg)
        if item:
            facts[key] = item[1].strip()
    pid = cmd('shell', 'pidof', '-s', PACKAGE, check=False)
    if not re.fullmatch(r'\d+', pid):
        return facts
    # Keep the raw read process-scoped and emit only the allowlisted evidence below.
    log = cmd('logcat', '-d', '--pid=' + pid, '-v', 'raw')
    for line in log.splitlines():
        if line.startswith('web_document_ready version='):
            facts['web_document_ready'] = True
            elapsed = re.search(r'elapsed_since_process_ms=(\d+)', line)
            if elapsed and 'first_web_document_ms' not in facts:
                facts['first_web_document_ms'] = int(elapsed[1])
        elif line == 'main_frame_bridge_ready':
            facts['main_frame_bridge_ready'] = True
        elif re.fullmatch(r'gpu_ready_ms=\d+ apk_version=[\w.-]+', line):
            facts['gpu_open_ms'] = int(line.split()[0].split('=')[1])
        elif (nodes := re.fullmatch(r'(pose|classifier): (\d+)/(\d+) nodes on LiteRT GPU', line)):
            facts['delegation'][nodes[1]] = {'delegated': int(nodes[2]), 'total': int(nodes[3]), 'accelerator': 'LiteRT GPU'}
        elif 'OpenCL not supported on this platform. Using OpenGL instead.' in line:
            facts['runtime_backend_observed'] = 'OpenGL'
        elif line.startswith('{'):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get('schema') == 'qolda-android-run-v1' and value.get('package') == PACKAGE:
                facts['runs'].append(value)
    facts['both_graphs_full_gpu_observed'] = all(
        facts['delegation'].get(graph, {}).get('total') == required
        and facts['delegation'].get(graph, {}).get('delegated') == required
        for graph, required in (('pose', 278), ('classifier', 60))) and 'gpu_open_ms' in facts
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', default='adb')
    parser.add_argument('--serial', help='Only where more than one device is connected; never recorded')
    parser.add_argument('--apk-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    expected = json.loads(args.apk_receipt.read_text(encoding='utf-8'))
    text = json.dumps(collect(args.adb, expected, args.serial), indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
