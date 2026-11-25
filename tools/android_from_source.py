"""Acquire official Android inputs and convert Qolda's two locally owned graphs to FP32."""
from __future__ import annotations

import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LITERT_VERSION = '2.4.0'
ABI = 'arm64-v8a'
POSE_SHA = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
POSE_POLICY = 'seriq-e02-forearm-2d-v1'
CHANNELS = ['right_elbow_x', 'right_elbow_y', 'right_wrist_x', 'right_wrist_y', 'pair_mask_0_or_1']
POSE_ARCHIVE = 'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'
POSE_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/' + POSE_ARCHIVE
POSE_ZIP_SHA = 'f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741'
POSE_MEMBER = f'20230831/rtmpose_onnx/{POSE_ARCHIVE[:-4]}/end2end.onnx'
MAVEN = 'https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert'
UPSTREAM_LITERT = {
    'build_config_gpu.h': (
        'https://raw.githubusercontent.com/google-ai-edge/LiteRT/14b37ee5489894161548390f448c4a68c94f4ce5/'
        'litert/build_common/config/build_config_gpu.h',
        '43957016e5f78881be3a53f57b3e2678278bd6d77d00314553bafe204d022b62',
    ),
    'litert_cc_sdk.zip': (
        'https://github.com/google-ai-edge/LiteRT/releases/download/v2.4.0/litert_cc_sdk.zip',
        '2d46770d2d8244e21f3f897b6af07d78513d00e0d6c68a332897681b4252c052',
    ),
    'litert-2.4.0.aar': (
        f'{MAVEN}/litert/2.4.0/litert-2.4.0.aar',
        '13926c98f35417b6fe4f104f780ab869747b4a69f08a91ccd224013c6bf71615',
    ),
    'litert-gpu-2.4.0.aar': (
        f'{MAVEN}/litert-gpu/2.4.0/litert-gpu-2.4.0.aar',
        '9bf935660d3ba33c45e43345fe77b7f4cc0d8f63ec8e007e45994863254385b2',
    ),
}
# Shapes and counts are the native engine contract, not pins for historical output bytes.
GRAPHS = {
    'pose': ('rtmpose', {'input': [1, 256, 192, 3]}, {'simcc_x': [1, 17, 384], 'simcc_y': [1, 17, 512]}, 278),
    'classifier': ('a-tcn', {'features': [1, 64, 5]}, {'logit': [1, 1]}, 60),
}
CONVERTER = 'onnx2tf==2.6.9/flatbuffer_direct'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def disk(path):
    with path.open('rb') as source:
        return {'bytes': os.fstat(source.fileno()).st_size, 'sha256': file_digest(source, 'sha256').hexdigest()}


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        out.write(data)
    record = {'bytes': len(data), 'sha256': sha256(data).hexdigest()}
    require(disk(path) == record, f'failed readback: {path}')
    return record


def write_json(path, payload):
    return write_bytes(path, (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def read_json(path):
    return json.loads(path.read_bytes())


def private_dir(path):
    path = path.resolve()
    require(not path.is_relative_to(ROOT) and not any((p / '.git').exists() for p in (path, *path.parents)),
            'output must be outside any Git checkout')
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_bundle(path):
    bundle = read_json(path)
    model, pose = bundle['model'], bundle['pose']
    require(bundle['schema'] == 'seriq-bundle-v1' and
            (model['id'], model['variant'], model['slot'], model['pose_policy'], model['device']) ==
            ('A', 'A', 'scratch', POSE_POLICY, 'cuda:0') and not model.get('base_sha256') and
            (pose['engine'], pose['policy'], pose['original_sha256']) == ('rtmpose', POSE_POLICY, POSE_SHA),
            'expected a locally trained standalone CUDA E02/A bundle with the pinned RTMPose')
    require(re.fullmatch(r'[0-9a-f]{40}', bundle['source_commit']) and all(
        re.fullmatch(r'[0-9a-f]{64}', v) for v in (
            pose['export_sha256'], model['sha256'], bundle['training_manifest_sha256'],
        )), 'invalid bundle provenance pin')
    require(re.fullmatch(r'[A-Za-z0-9._-]{1,80}', model['version']) and
            type(model['threshold']) in (int, float) and 0 < model['threshold'] < 1 and
            type(bundle['artifact_bytes']) is int and 0 < bundle['artifact_bytes'] <= 90_000_000,
            'invalid classifier version, threshold or artifact size')
    return bundle


def zip_member(archive, name):
    members = [entry for entry in archive.infolist() if entry.filename == name]
    require(len(members) == 1 and not members[0].is_dir(), f'expected one ZIP file member: {name}')
    return archive.read(members[0])


def download(url, expected_sha, dest):
    with urlopen(url, timeout=90) as response, dest.open('xb') as out:
        shutil.copyfileobj(response, out)
    pin = disk(dest)
    require(pin['sha256'] == expected_sha, f'upstream SHA-256 mismatch: {url}')
    print(f'{dest.name} SHA-256 {pin["sha256"]}', flush=True)
    return pin


def fetch(kind, dest):
    out = private_dir(dest)
    if kind == 'pose':
        source = out / POSE_ARCHIVE
        provenance = download(POSE_URL, POSE_ZIP_SHA, source)
        with ZipFile(source) as archive:
            graph = zip_member(archive, POSE_MEMBER)
        require(sha256(graph).hexdigest() == POSE_SHA, 'wrong upstream RTMPose ONNX')
        pin = write_bytes(out / 'end2end.onnx', graph)
        write_json(out / 'upstream-pose.json', {'url': POSE_URL, 'archive': provenance, 'onnx': pin})
    else:
        provenance = {name: {'url': url, **download(url, digest, out / name)}
                      for name, (url, digest) in UPSTREAM_LITERT.items()}
        write_json(out / 'upstream-litert.json', provenance)


def inspect_fp32(data, name):
    from ai_edge_litert import schema_py_generated as fb

    from tools.stage_android_engine import inspect_model

    _, inputs, outputs, count = GRAPHS[name]
    require(0 < len(data) <= 64 << 20, f'{name}: model exceeds engine asset limit')
    # Reuse the existing signature/name/shape/type/op-count gate without its APK extraction path.
    spec = inspect_model(data, inputs, outputs, count)
    model = fb.Model.GetRootAsModel(data, 0)
    graph = model.Subgraphs(0)
    unsupported = {fb.BuiltinOperator.CUSTOM, fb.BuiltinOperator.QUANTIZE, fb.BuiltinOperator.DEQUANTIZE}
    for i in range(graph.OperatorsLength()):
        code = model.OperatorCodes(graph.Operators(i).OpcodeIndex())
        require(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()) not in unsupported,
                f'{name}: custom/quantized operator is not an FP32 build input')
    allowed_types = {fb.TensorType.FLOAT32, fb.TensorType.BOOL, fb.TensorType.INT32, fb.TensorType.INT64}
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        require(tensor.Type() in allowed_types, f'{name}: unexpected non-FP32 tensor type')
    # Some converter-inferred ShapeSignature hints contain -1 even for the proven
    # batch-1 pose. The concrete Shape and serving_default mapping are gated above;
    # the unchanged native engine also verifies the actual tensor contract at open.
    return spec


def convert(bundle_path, pose, classifier, dest):
    import onnx

    from tools.export_android_onnx import rewrite_classifier, rewrite_pose

    require(os.name == 'nt' and sys.version_info[:3] == (3, 12, 13) and
            version('onnx2tf') == '2.6.9' and version('onnx') == '1.20.1' and
            version('ai-edge-litert') == '2.1.2', 'use the pinned Windows converter venv')
    exe = Path(sys.executable).with_name('onnx2tf.exe')
    require(exe.is_file(), 'onnx2tf.exe not found in the converter venv')
    bundle_pin = disk(bundle_path)
    bundle = read_bundle(bundle_path)
    sources = {'pose': pose.resolve(), 'classifier': classifier.resolve()}
    source_pins = {name: disk(path) for name, path in sources.items()}
    require(source_pins['pose']['sha256'] == bundle['pose']['export_sha256'], 'batch-1 RTMPose differs from bundle')
    a = read_json(classifier.with_suffix(classifier.suffix + '.json'))
    require(a['schema'] == 'qolda-android-local-a-onnx-v1' and a['bundle'] == bundle_pin and
            a['artifact'] == {'bytes': bundle['artifact_bytes'], 'sha256': bundle['model']['sha256']} and
            a['graph'] == source_pins['classifier'] and a['device'] == 'cuda:0' and a['channels'] == CHANNELS,
            'classifier ONNX does not match its CUDA export receipt and bundle')
    out = private_dir(dest)
    env = {**os.environ, 'PATH': str(exe.parent) + os.pathsep + os.environ.get('PATH', '')}
    result = {}
    for name, source in sources.items():
        # Do not pass a training or bundle path to the converter, even with -nuo.
        stem = GRAPHS[name][0]
        rewritten = out / f'{stem}-android.onnx'
        working = out / 'work' / rewritten.name
        working.parent.mkdir(exist_ok=True)
        if name == 'pose':
            rewrite_pose(source, rewritten)
        else:
            rewrite_classifier(source, rewritten)
        onnx.checker.check_model(str(rewritten))
        rewrite_pin = disk(rewritten)
        shutil.copyfile(rewritten, working)
        require(disk(working) == rewrite_pin, 'converter working copy differs from the rewrite')
        converted = out / 'work' / name
        flags = ['-tb', 'flatbuffer_direct', '-nuo', '-dsm', '-osd', '-coion', '-v', 'warn']
        if name == 'classifier':
            flags += ['-kat', 'features']
        subprocess.run([str(exe), '-i', str(working), '-o', str(converted), *flags],
                       cwd=working.parent, env=env, check=True)
        require(disk(source) == source_pins[name] and disk(rewritten) == rewrite_pin,
                f'{name}: protected ONNX changed during conversion')
        data = (converted / f'{working.stem}_float32.tflite').read_bytes()
        specimen = inspect_fp32(data, name)
        output = out / f'{stem}-fp32.tflite'
        result[name] = {'file': output.name, **write_bytes(output, data), **specimen,
                        'input_onnx': source_pins[name], 'rewritten_onnx': rewrite_pin}
    require(disk(bundle_path) == bundle_pin, 'bundle changed during conversion')
    receipt = out / 'conversion.json'
    write_json(receipt, {'schema': 'qolda-android-local-conversion-v1', 'converter': CONVERTER,
                         'bundle': bundle_pin, 'graphs': result, 'parity_verified': False,
                         'phone_gpu_verified': False})
    print(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    acquire = commands.add_parser('fetch', help='download and verify upstream packages only')
    acquire.add_argument('--kind', required=True, choices=('pose', 'litert'))
    acquire.add_argument('--out', required=True, type=Path)
    run = commands.add_parser('convert', help='convert the locally bound CUDA ONNX exports')
    for arg in ('bundle', 'pose', 'classifier', 'out'):
        run.add_argument('--' + arg, required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'fetch':
        fetch(args.kind, args.out)
    else:
        convert(args.bundle, args.pose, args.classifier, args.out)


if __name__ == '__main__':
    main()
