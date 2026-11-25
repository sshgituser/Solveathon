"""Local video adapter for the frozen SerIQ E02 models. No remote inference or raw-video logs."""
from __future__ import annotations

import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from backend.contract import ClipRejected, VERSION, summarize
from backend.live import CameraAttempt

ROOT = Path(__file__).resolve().parents[1]
MAX_DURATION = 30.0
MAX_PIXELS = 4096 * 2160
MAX_PACKETS = 9000
MAX_FRAMES = 3600
MODEL_FILES = {'scratch': ('A', 'A-tcn.safetensors'), 'transfer': ('B', 'B-slim-last-block.safetensors')}
MISSING = {'no_visible_forearm_pair', 'no_visible_arm_triplet'}
VISIBILITY = {'insufficient_visible_forearm_pair', 'incomplete_forearm_coverage', 'ambiguous_forearm_tracks'}
DEVICE_MARKERS = ('cuda', 'cudnn', 'cublas', 'device-side', 'out of memory', 'no kernel image')


def device_failure(exc):
    """A broken GPU must retire the engine, not be reported as a bad clip."""
    text = str(exc).lower()
    return any(marker in text for marker in DEVICE_MARKERS)


class Engines(dict):
    """Engine registry that withdraws readiness after a device failure."""

    def __init__(self):
        super().__init__()
        self.dead = {}
        self.live: CameraAttempt | None = None  # only the strict CUDA bundle installs a camera adapter

    def mark_dead(self, name, reason):
        self.dead[name] = str(reason)[:240]

    def get(self, name, default=None):
        if name in self.dead:
            return None
        return super().get(name, default)

    def readiness(self):
        return {name: callable(self.get(name)) for name in MODEL_FILES}


def guard(registry, name, analyze):
    def guarded(body, mime):
        try:
            return analyze(body, mime)
        except ClipRejected:
            raise
        except Exception as exc:
            if device_failure(exc):
                registry.mark_dead(name, exc)
            raise
    return guarded


def wire_result(model, version, prediction, margin):
    label, reason = prediction.get('label'), prediction.get('reason')
    warnings, attempts = [], []
    if reason and label is None:
        if reason in MISSING:
            warnings = ['no_pose']
        else:
            warnings = ['low_visibility'] if reason in VISIBILITY else []
            attempts = [{'index': 1, 'label': 'uncertain', 'confidence': None,
                         'reason': 'low_visibility' if warnings else 'unscorable'}]
    else:
        confidence = prediction.get('confidence')
        if (reason or label not in {'complete', 'incomplete'} or type(confidence) not in (int, float)
                or not isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError('invalid predictor result')
        low = confidence < margin
        attempts = [{'index': 1, 'label': 'uncertain' if low else label, 'confidence': confidence,
                     'reason': 'low_confidence' if low else None}]
    return {'version': VERSION, 'model': model, 'model_version': version,
            'attempts': attempts, 'counts': summarize(attempts), 'warnings': warnings}


def number(value):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if isfinite(n) else None


def probe_frames(path, mime, ffprobe):
    with path.open('rb') as stream:
        signature = stream.read(16)
    if mime == 'video/mp4' and signature[4:8] == b'ftyp':
        demuxer, codecs = 'mov', {'h264', 'hevc', 'mpeg4', 'av1'}
        extra = ['-enable_drefs', '0', '-use_absolute_path', '0']
    elif mime == 'video/webm' and signature[:4] == b'\x1a\x45\xdf\xa3':
        demuxer, codecs, extra = 'matroska', {'vp8', 'vp9', 'av1'}, []
    else:
        raise ClipRejected()
    cmd = [ffprobe, '-v', 'error', '-max_alloc', '134217728', '-protocol_whitelist', 'file',
           '-f', demuxer, *extra, '-read_intervals', f'%+#{MAX_PACKETS+1}', '-show_packets',
           '-show_entries', 'stream=index,codec_type,codec_name,width,height,duration:format=duration:packet=stream_index,pts_time,duration_time',
           '-of', 'json', '-i', str(path)]
    try:
        ran = subprocess.run(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, timeout=8, check=True)
        meta = json.loads(ran.stdout)
        video = next(stream for stream in meta['streams'] if stream['codec_type'] == 'video')
        w, h = int(video['width']), int(video['height'])
        packets = meta['packets']
        timing = [(number(p.get('pts_time')), number(p.get('duration_time')))
                  for p in packets if p.get('stream_index') == video['index']]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ClipRejected() from exc
    if video.get('codec_name') not in codecs or not (0 < w <= 4096 and 0 < h <= 4096 and w*h <= MAX_PIXELS):
        raise ClipRejected()
    if len(packets) > MAX_PACKETS or not 0 < len(timing) <= MAX_FRAMES:
        raise ClipRejected()
    times = [(pts, max(0.0, length or 0.0)) for pts, length in timing if pts is not None]
    durations = [number(video.get('duration')), number(meta.get('format', {}).get('duration'))]
    if times:
        durations.append(max(pts+d for pts,d in times) - min(pts for pts,_ in times))
    duration = max((n for n in durations if n is not None), default=0)
    if duration > MAX_DURATION:
        raise ClipRejected('duration_limit')
    if duration <= 0:
        raise ClipRejected()
    return len(timing)


def extract_video(path, pose, packet_count):
    import cv2
    import numpy as np
    from ml.extract import phase_indices

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise ClipRejected()
        native = number(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Same MP4 sampling as training. Unfinalized WebM can lack a reliable reported frame count.
        count = int(native) if path.suffix == '.mp4' and native and 1 <= native <= MAX_FRAMES else packet_count
        xyz, masks = [], []
        cursor = -1
        for i in phase_indices(count, repeat_short=False):
            while cursor + 1 < i:
                if not cap.grab():
                    raise ClipRejected()
                cursor += 1
            ok, frame = cap.read()
            cursor += 1
            if not ok or frame.size // 3 > MAX_PIXELS:
                raise ClipRejected()
            coords, observed = pose(frame)
            xyz.append(coords)
            masks.append(observed)
        return np.stack(xyz), np.stack(masks)
    finally:
        cap.release()


def make_clip_adapter(model, version, predict, pose, scratch, ffprobe):
    margin = predict.metadata['low_confidence_below']

    def analyze(body, mime):
        with TemporaryDirectory(prefix='seriq-upload-', dir=scratch) as temp:
            clip = Path(temp) / ('clip.mp4' if mime == 'video/mp4' else 'clip.webm')
            clip.write_bytes(body)
            packet_count = probe_frames(clip, mime, ffprobe)
            coords, observed = extract_video(clip, pose, packet_count)
            return wire_result(model, version, predict.predict_pose(coords, observed), margin)

    return analyze


def load_bundle_engines(bundle_path, artifacts, pose_model, scratch, ffprobe, slim_base=None):
    """Load a versioned private bundle (``seriq-bundle-v1``) on strict CUDA.

    Unlike the historical freeze, the bundle is not tied to the previous machine's
    source hashes: it names the pose graph, the classifier and the training
    manifest by digest, and every digest is verified before the engine registers.
    """
    scratch = scratch.resolve()
    if scratch.is_relative_to(ROOT):
        raise ValueError('video scratch must be outside the checkout')
    scratch.mkdir(parents=True, exist_ok=True)
    ffprobe = shutil.which(ffprobe)
    if not ffprobe:
        raise ValueError('ffprobe executable is required for real video analysis')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file|enable_drefs;0|use_absolute_path;0'
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    import cv2
    import numpy as np
    import torch
    from ml.api import load_artifact
    from ml.data import FOREARM_POLICY, pose_sequence, sha256, verify_training_inputs
    from ml.gpu import CudaUnavailable, assert_model_on_cuda, open_cuda_pose, prepare_static_graph

    bundle_path = Path(bundle_path)
    bundle = json.loads(bundle_path.read_text('utf-8'))
    if bundle.get('schema') != 'seriq-bundle-v1':
        raise ValueError('unsupported runtime bundle schema')
    model_spec, pose_spec = bundle['model'], bundle['pose']
    commit = bundle.get('source_commit')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('bundle must be tied to a source commit')
    if bundle.get('runtime', {}).get('device_class') != 'nvidia-cuda':
        raise ValueError('bundle is not a CUDA runtime delivery')
    if pose_spec.get('policy') != FOREARM_POLICY or model_spec.get('pose_policy') != FOREARM_POLICY:
        raise ValueError('bundle is not a forearm-policy engine')
    # Only the agreed A/scratch delivery is accepted here. A bundle carrying another
    # identifier, variant or SLiM base is refused instead of being served as "scratch".
    slot = model_spec.get('slot') or ('scratch' if (
        model_spec.get('id') == 'A' and model_spec.get('variant') == 'A'
        and not model_spec.get('base_sha256')) else None)
    if slot != 'scratch' or model_spec.get('id') != 'A' or model_spec.get('variant') != 'A':
        raise ValueError('bundle is not the agreed A/scratch delivery')
    if model_spec.get('base_sha256'):
        raise ValueError('standalone A must not declare a SLiM base')
    if sha256(pose_model) != pose_spec['original_sha256']:
        raise ValueError('pose model digest does not match the bundle')
    graph = bundle_path.parent / 'end2end.b1.onnx'
    if not graph.exists():
        prepare_static_graph(pose_model, graph)
    if sha256(graph) != pose_spec['export_sha256']:
        raise ValueError('pose graph digest does not match the bundle')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if model_spec.get('file') != 'A-tcn.safetensors':
        raise ValueError('scratch bundle requires its named standalone A file')
    artifact = Path(artifacts) / model_spec['file']
    predictor = load_artifact(artifact, slim_base=slim_base, device='cuda:0',
                              expected_sha256=model_spec['sha256'])
    assert_model_on_cuda(predictor.model, 'cuda:0')
    m = predictor.metadata
    if (m['kind'], m['pose_policy'], m['threshold'], m['version']) != (
            model_spec['id'], FOREARM_POLICY, model_spec['threshold'], model_spec['version']):
        raise ValueError('artifact metadata does not match the bundle')
    verify_training_inputs(m, manifest=bundle.get('training_manifest_sha256'),
                           original=pose_spec['original_sha256'], graph=pose_spec['export_sha256'])
    try:
        extractor = open_cuda_pose(graph)
    except CudaUnavailable as exc:
        raise ValueError(f'strict CUDA pose engine is unavailable: {exc}') from exc
    xyz = np.zeros((64, 33, 3), dtype=np.float32)
    observed = np.zeros((64, 33), dtype=bool)
    xyz[:, 14, 0] = np.linspace(.10, .25, 64)
    xyz[:, 16, 0] = xyz[:, 14, 0] + .30
    observed[:, (14, 16)] = True
    warm_pose = pose_sequence(xyz, observed, FOREARM_POLICY)
    if not isfinite(predictor.predict_many([warm_pose])[0]):
        raise ValueError('artifact did not pass a local forward warm-up')
    registry = Engines()
    registry.live = CameraAttempt(registry, extractor, predictor, m['version'] + '+' + model_spec['sha256'][:12])
    registry['scratch'] = guard(registry, 'scratch', make_clip_adapter(
        'scratch', m['version'] + '+' + model_spec['sha256'][:12], predictor, extractor, scratch, ffprobe))
    return registry


def load_frozen_engines(artifacts, freeze_path, pose_model, scratch, ffprobe, slim_base=None):
    scratch = scratch.resolve()
    if scratch.is_relative_to(ROOT):
        raise ValueError('video scratch must be outside the checkout')
    scratch.mkdir(parents=True, exist_ok=True)
    ffprobe = shutil.which(ffprobe)
    if not ffprobe:
        raise ValueError('ffprobe executable is required for real video analysis')
    # Configure this process before importing the decoder. Video is never opened as a URL.
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file|enable_drefs;0|use_absolute_path;0'
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    import cv2
    import numpy as np
    import torch
    from ml.api import load_artifact
    from ml.data import FOREARM_POLICY, pose_sequence, sha256
    from ml.gpu import CudaUnavailable, assert_model_on_cuda, open_cuda_pose

    config = json.loads(freeze_path.read_text('utf-8'))
    spec = config['pose']
    if (spec['engine'], spec['policy'], spec['sha256']) != ('rtmpose', FOREARM_POLICY, sha256(pose_model)):
        raise ValueError('pose engine, policy or hash does not match the freeze')
    if config['sealed_admission']['passed'] is not True:
        raise ValueError('expected an admitted forearm freeze')
    for name in ('api', 'data', 'extract', 'models', 'scoring', 'train'):
        file = f'ml/{name}.py'
        if sha256(ROOT / file) != config['sources'][file]:
            raise ValueError('ML source digest does not match the freeze')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    try:
        # Strict CUDA: the session refuses CPU fallback and proves full delegation.
        extractor = open_cuda_pose(pose_model)
    except CudaUnavailable as exc:
        raise ValueError(f'strict CUDA pose engine is unavailable: {exc}') from exc
    xyz = np.zeros((64, 33, 3), dtype=np.float32)
    observed = np.zeros((64, 33), dtype=bool)
    xyz[:, 14, 0] = np.linspace(.10, .25, 64)
    xyz[:, 16, 0] = xyz[:, 14, 0] + .30
    observed[:, (14, 16)] = True
    warm_pose = pose_sequence(xyz, observed, FOREARM_POLICY)
    registry = Engines()
    for name, (kind, filename) in MODEL_FILES.items():
        if name == 'transfer' and slim_base is None:
            continue
        spec = config['artifacts'][kind]
        expected = spec['sha256']
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('invalid artifact digest in the freeze')
        predictor = load_artifact(artifacts / filename, slim_base=slim_base, device='cuda:0',
                                  expected_sha256=expected)
        assert_model_on_cuda(predictor.model, 'cuda:0')
        m = predictor.metadata
        if (m['kind'], m['pose_policy'], m['threshold'], m['version']) != (kind, FOREARM_POLICY, spec['threshold'], config['version']):
            raise ValueError('artifact metadata does not match the freeze')
        if not isfinite(predictor.predict_many([warm_pose])[0]):
            raise ValueError('artifact did not pass a local forward warm-up')
        registry[name] = guard(registry, name, make_clip_adapter(
            name, m['version'] + '+' + expected[:12], predictor, extractor, scratch, ffprobe))
    return registry
