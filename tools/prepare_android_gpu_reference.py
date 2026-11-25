"""Produce private, generated-tensor CUDA references for the Android A probe.

This is a reference-data build, not a comparison with Android. No clips,
participant data, or holdout inputs are opened. See team/tasks/T-017-android-gpu.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch  # pyright: ignore[reportMissingImports] — use the pinned CUDA venv, not the editor Python.

from ml.api import load_artifact
from ml.data import FOREARM_NTU, FOREARM_POLICY, FRAMES, sha256
from ml.gpu import assert_cuda_only, assert_model_on_cuda, open_session


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require_digest(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'{path.name}: SHA-256 mismatch')
    return digest


def load_inputs(args):
    desktop = load_json(args.desktop_bundle)
    if desktop.get('schema') != 'seriq-bundle-v1':
        raise ValueError('expected the desktop A bundle')
    model, pose = desktop['model'], desktop['pose']
    if (model.get('id'), model.get('variant'), model.get('slot'), model.get('pose_policy'),
        pose.get('policy'), desktop['runtime'].get('device_class')) != (
            'A', 'A', 'scratch', FOREARM_POLICY, FOREARM_POLICY, 'nvidia-cuda') or model.get('base_sha256'):
        raise ValueError('reference only supports standalone A / scratch / forearm / CUDA')
    candidate = load_json(args.android_bundle)
    if (candidate.get('schema') != 'qolda-android-export-candidate-v1'
            or candidate.get('pose_policy') != FOREARM_POLICY):
        raise ValueError('unexpected Android export schema or policy')
    mobile_pose, mobile_a = (candidate['graphs'][k] for k in ('pose', 'classifier'))
    if (mobile_pose['source_onnx_sha256'] != pose['export_sha256']
            or mobile_a['source_safetensors_sha256'] != model['sha256']
            or mobile_a['threshold'] != model['threshold']):
        raise ValueError('Android and desktop bundles do not pin the same source models')
    for spec in (mobile_pose, mobile_a):
        name = spec['file']
        if not name.endswith('.tflite') or Path(name).name != name:
            raise ValueError('expected a bundle-local TFLite basename')
        require_digest(args.android_bundle.parent / name, spec['sha256'])
    graph = args.desktop_bundle.parent / 'end2end.b1.onnx'
    require_digest(graph, pose['export_sha256'])
    require_digest(args.weights, model['sha256'])
    return graph, model, pose, mobile_pose, mobile_a


def pose_inputs():
    # Inputs already in normalized N,H,W,C layout. They are not decoded videos.
    yy, xx = np.mgrid[0:256, 0:192]
    ramp = np.stack((2.6 * xx / 191 - 1.3, 2.2 * yy / 255 - 1.1,
                     ((xx // 13 + yy // 17) % 2) * 1.5 - .7), axis=-1)
    return {'neutral': np.zeros((1, 256, 192, 3), dtype=np.float32),
            'rgb_pattern': ramp[None, ...].astype(np.float32)}


def classifier_inputs():
    t = np.linspace(0, 1, FRAMES, dtype=np.float32)
    phase = np.pi * (1 - t)
    elbow = np.stack((.06 * t, .045 * t * (1 - t)), axis=-1)
    wrist = elbow + .30 * np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    values = np.concatenate((elbow, wrist, np.ones((FRAMES, 1), np.float32)), axis=1)[None, ...]
    short_motion = values.copy()
    short_motion[0, :, 2:4] = elbow + .30 * np.stack((np.cos(phase * .35), np.sin(phase * .35)), axis=-1)
    sparse = values.copy()
    sparse[0, 2::4, 4] = 0
    none = values.copy()
    none[0, :, 4] = 0
    return {'full_motion': values, 'short_motion': short_motion, 'sparse_motion': sparse, 'all_missing': none}


def tensor_file(out, name, value):
    fp32 = np.ascontiguousarray(value, dtype='<f4')
    if not np.isfinite(fp32).all():
        raise ValueError(f'{name}: non-finite tensor')
    path = out / (name + '.f32')
    with path.open('xb') as f:
        f.write(fp32.tobytes())
    return {'file': path.name, 'shape': list(fp32.shape), 'dtype': 'float32-le',
            'bytes': path.stat().st_size, 'sha256': sha256(path)}


def pose_reference(graph, out, cases):
    session = open_session(graph)
    name, = (meta.name for meta in session.get_inputs())
    cuda_feeds = {k: {name: np.ascontiguousarray(v.transpose(0, 3, 1, 2))} for k, v in cases.items()}
    placement = assert_cuda_only(session, next(iter(cuda_feeds.values())), label='Android pose reference')
    results = {}
    for key, frame in cases.items():
        sx, sy = (np.asarray(output) for output in session.run(['simcc_x', 'simcc_y'], cuda_feeds[key]))
        results[key] = {'input': tensor_file(out, f'pose-{key}-input', frame),
                        'simcc_x': tensor_file(out, f'pose-{key}-simcc_x', sx),
                        'simcc_y': tensor_file(out, f'pose-{key}-simcc_y', sy),
                        'axis_peak_mask_at_0_5': (np.minimum(sx.max(axis=-1), sy.max(axis=-1))[0] >= .5).tolist()}
    return placement, results


def classifier_reference(predictor, out, cases):
    assert_model_on_cuda(predictor.model, 'cuda:0')
    elbow, wrist = FOREARM_NTU
    results = {}
    for key, features in cases.items():
        coordinates = np.zeros((1, FRAMES, 25, 3), np.float32)
        observed = np.zeros((1, FRAMES, 25), bool)
        coordinates[:, :, elbow, :2] = features[:, :, :2]
        coordinates[:, :, wrist, :2] = features[:, :, 2:4]
        observed[:, :, elbow] = observed[:, :, wrist] = features[:, :, 4] == 1
        inputs = tuple(torch.as_tensor(v, device='cuda:0') for v in (
            coordinates, observed, np.zeros((1, FRAMES, 2), np.float32), np.zeros((1, FRAMES, 2), bool)))
        with torch.inference_mode():
            logit = predictor.model(*inputs)
            if logit.device.type != 'cuda' or logit.shape != (1,):
                raise ValueError('classifier returned an invalid or non-CUDA logit')
            probability = logit.float().sigmoid().item()
        results[key] = {'input': tensor_file(out, f'a-{key}-features', features),
                        'logit': tensor_file(out, f'a-{key}-logit', logit.float().cpu().numpy()),
                        'p_complete': probability, 'threshold_side': probability >= predictor.metadata['threshold']}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--desktop-bundle', required=True, type=Path)
    parser.add_argument('--android-bundle', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(ROOT):
        raise ValueError('reference tensors must stay outside Git')
    if args.out.exists():
        raise FileExistsError(args.out.name)
    graph, model, pose, mobile_pose, mobile_a = load_inputs(args)
    predictor = load_artifact(args.weights, device='cuda:0', expected_sha256=model['sha256'])
    if (predictor.metadata.get('variant'), predictor.metadata['threshold'], predictor.metadata['version']) != (
            'A', model['threshold'], model['version']):
        raise ValueError('loaded model differs from the pinned A bundle')
    args.out.mkdir(parents=True, exist_ok=False)
    pose_placement, poses = pose_reference(graph, args.out, pose_inputs())
    classifications = classifier_reference(predictor, args.out, classifier_inputs())
    result = {'schema': 'qolda-android-cuda-reference-v1',
              'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'scope': 'graph-level generated tensors only; not real clips or final app acceptance',
              'synthetic': True, 'participant_data': False, 'device': torch.cuda.get_device_name(0),
              'runtime': {'ort': ort.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda},
              'android_bundle_sha256': sha256(args.android_bundle), 'pose_policy': FOREARM_POLICY,
              'pose': {'layout': 'NHWC RGB float32 (already normalized)', 'ort_export_sha256': pose['export_sha256'],
                       'tflite_sha256': mobile_pose['sha256'], 'cuda_profile': pose_placement, 'cases': poses},
              'classifier': {'input_channels': mobile_a['channel_order'], 'threshold': model['threshold'],
                             'source_sha256': model['sha256'], 'tflite_sha256': mobile_a['sha256'],
                             'device': 'cuda:0', 'cases': classifications},
              'comparison': {'max_probability_delta': .001, 'threshold_side_must_match': True,
                             'max_bin_delta_for_reference_visible_axes': 1, 'axis_peak_mask_must_match': True,
                             'note': 'All-missing is a diagnostic tensor, not a legitimate classified attempt. '
                                     'Report real-video visibility/label parity separately.'}}
    receipt = args.out / 'reference.json'
    with receipt.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'receipt': receipt.name, 'sha256': sha256(receipt), 'device': result['device'],
                      'pose_cases': len(poses), 'classifier_cases': len(classifications),
                      'pose_cuda_nodes': pose_placement['nodes'], 'pose_non_cuda': pose_placement['non_cuda']}, indent=2))


if __name__ == '__main__':
    main()
