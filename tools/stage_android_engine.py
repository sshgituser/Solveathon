"""Stage only the runtime assets whose exact bytes passed the Qolda T-017 probe.

This is a build input for T-010, not an APK or product E2E result. Read verified
model/library bytes from an APK with the same payload, even if its outer hash or
app id differs. Do not extract the diagnostic .so, reference tensors, or its
stale bundle.json. Run with the pinned Android export environment (schema only).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from ai_edge_litert import schema_py_generated as fb  # pyright: ignore[reportMissingImports] — pinned export-env only.

ROOT = Path(__file__).resolve().parents[1]
PROVEN_APK = '2a42b2cfc9fe048ad25f3f838049f33918339f8adf4ddbcfc0e7e93ae3d68afa'
PROOF_COMMIT = '6335f04df613b88ed8da1535e23077262b4c0bb4'
LITERT = '2.4.0'
ABI = 'arm64-v8a'
MODELS = {
    'pose': ('rtmpose-fp32.tflite', 54358484, '2dd322a0c414e77fa591a82d0b670ec36c3e7547a57f42b121b9870b6ce5b8cd',
             {'input': [1, 256, 192, 3]}, {'simcc_x': [1, 17, 384], 'simcc_y': [1, 17, 512]}, 278),
    'classifier': ('a-tcn-fp32.tflite', 22164, '99cb4409cad108ece6baddb3b1f90ad7c8bfc7e8906390dacd86fd29b6d7ec34',
                   {'features': [1, 64, 5]}, {'logit': [1, 1]}, 60),
}
LIBRARIES = {
    'libLiteRt.so': (5935696, 'df7fcd5a55a84c5850c256a7f1bcb3c3cc8ee4de4c40426a4fee0d0ef7101cba'),
    'libLiteRtClGlAccelerator.so': (3415408, '23c7f0113bcd99677105aaa2b8f0986baf07db9d4ebcfc1d893e7dcf21dbe465'),
    'libc++_shared.so': (1330832, '69e517e645a6d38f7536d64108d245f5a8208a4a5cd43375b4906f60a60a86c4'),
}


def digest(data):
    return sha256(data).hexdigest()


def pinned_member(archive, member, size, checksum):
    entries = [info for info in archive.infolist() if info.filename == member]
    if len(entries) != 1 or entries[0].file_size != size:
        raise ValueError(f'{member}: not the unique T-017 member at the recorded size')
    data = archive.read(entries[0])
    if digest(data) != checksum:
        raise ValueError(f'{member}: not the recorded T-017 bytes')
    return data


def inspect_model(data, inputs, outputs, count):
    if not fb.Model.ModelBufferHasIdentifier(data, 0):
        raise ValueError('expected TF Lite FlatBuffer')
    model = fb.Model.GetRootAsModel(data, 0)
    if model.SubgraphsLength() != 1 or model.SignatureDefsLength() != 1:
        raise ValueError('expected one static graph and signature')
    graph, signature = model.Subgraphs(0), model.SignatureDefs(0)
    if (signature.SignatureKey() != b'serving_default' or signature.SubgraphIndex() != 0
        or graph.OperatorsLength() != count):
        raise ValueError('unexpected signature or op count')

    def check_tensors(indices, expected, maps, length):
        actual = {}
        for index in indices:
            tensor = graph.Tensors(int(index))
            if tensor.Type() != fb.TensorType.FLOAT32:
                raise ValueError('expected FP32 public tensors')
            actual[tensor.Name().decode()] = [int(n) for n in tensor.ShapeAsNumpy()]
        mapped = {maps(i).Name().decode(): int(maps(i).TensorIndex()) for i in range(length)}
        if actual != expected or mapped != {graph.Tensors(int(i)).Name().decode(): int(i) for i in indices}:
            raise ValueError('shape/name or signature mapping differs from T-017')

    check_tensors(graph.InputsAsNumpy(), inputs, signature.Inputs, signature.InputsLength())
    check_tensors(graph.OutputsAsNumpy(), outputs, signature.Outputs, signature.OutputsLength())
    return {'signature': 'serving_default', 'dtype': 'float32', 'inputs': inputs, 'outputs': outputs, 'ops': count}


def stage(apk, litert_dir, out):
    if out.resolve().is_relative_to(ROOT) or out.exists():
        raise ValueError('output must be a new private directory outside the repository')
    payload = {}
    graphs = {}
    runtime = {}
    # Capture one file image; a collaborator can replace app-debug.apk during this step.
    raw = apk.read_bytes()
    with ZipFile(BytesIO(raw)) as archive:
        for key, (name, size, checksum, inputs, outputs, count) in MODELS.items():
            data = pinned_member(archive, 'assets/' + name, size, checksum)
            rel = 'assets/qolda/' + name
            payload[rel] = data
            graphs[key] = {'path': rel, 'bytes': size, 'sha256': checksum,
                           **inspect_model(data, inputs, outputs, count)}
        for name, (size, checksum) in LIBRARIES.items():
            rel = f'jniLibs/{ABI}/{name}'
            payload[rel] = pinned_member(archive, f'lib/{ABI}/{name}', size, checksum)
            runtime[name] = {'path': rel, 'sha256': checksum, 'bytes': size}
    for component in ('litert', 'litert-gpu'):
        with ZipFile(litert_dir / f'{component}-{LITERT}.aar') as archive:
            for entry in ('LICENSE', 'THIRD_PARTY_NOTICE.txt'):
                name = entry if entry.endswith('.txt') else entry + '.txt'
                payload[f'assets/qolda/licenses/{component}-{name}'] = archive.read(entry)
    payload['assets/qolda/licenses/ATTRIBUTION.md'] = (ROOT / 'artifacts/seriq/ATTRIBUTION.md').read_bytes()
    manifest = {
        'schema': 'qolda-offline-gpu-assets-v1',
        'status': 't017_payload_verified_t010_e2e_pending',
        'evidence': {'commit': PROOF_COMMIT, 'apk_sha256': PROVEN_APK,
                     'apk_id': 'net.qolda.diag', 'apk_version': 't017-1',
                     'extracted_from_apk_sha256': digest(raw), 'device': 'OnePlus CPH2399 / API 34',
                     'scope': 'T-017 diagnostic compiled/run on one development input per graph; not full-clip/offline app'},
        'runtime': {'version': LITERT, 'abi': ABI, 'requested_accelerator': 'GPU', 'observed_backend': 'OpenGL',
                    'require_full_delegation_before_run': True, 'gpu_options_toml': 'precision = 2\n',
                    'libraries': runtime},
        'graphs': graphs,
        'classifier': {'slot': 'scratch', 'id': 'A', 'model_version': 'seriq-e02-forearm-1.0.0+8dd234d4a3e1',
                       'threshold': 0.5, 'uncertain_margin': 0.10, 'pose_policy': 'seriq-e02-forearm-2d-v1',
                       'channels': ['right_elbow_x', 'right_elbow_y', 'right_wrist_x', 'right_wrist_y', 'pair_mask_0_or_1']},
    }
    manifest_name = 'assets/qolda/engine.json'
    payload[manifest_name] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    out.mkdir(parents=True, exist_ok=False)
    for rel, data in payload.items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(data)
        if digest(dest.read_bytes()) != digest(data):
            raise OSError(f'{rel}: failed readback')
    print(json.dumps({'manifest': manifest_name, 'manifest_sha256': digest(payload[manifest_name]),
                      'files': len(payload), 'proof_apk_sha256': PROVEN_APK,
                      'extraction_apk_sha256': digest(raw), 'e2e_verified': False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apk', required=True, type=Path)
    parser.add_argument('--litert-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    stage(args.apk, args.litert_dir, args.out)


if __name__ == '__main__':
    main()
