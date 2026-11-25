"""E01 lift-and-return pose proxy. No object detector or object tracking.

Train only from the owner's E01 manifest: python -m ml.extra_e01 --help.
Runtime entry point: open_engine(model_dir), returning a callable for video Paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
import time
from threading import Lock
from typing import Literal, TypedDict

import cv2
import numpy as np
import torch
from torch import nn

from ml.gpu import CUDA, CudaPose, CudaUnavailable, assert_model_on_cuda, cuda_stack

__all__ = ['open_engine']

VERSION = 'seriq-e01-pose-proxy-1.1.0'
SCHEMA = 'qolda-e01-cuda-bundle-v1'
FEATURES = 'phase48-arms-motion-wrist-gated-v2'
DATASET = '10.17632/49h9dcwx5v.1'
POSE = 'end2end.b1.onnx'
POSE_SHA = '2e0a9c0c5287bfa8897440975c32e98e26be90e88ad9ddd9e72fba8644b68b78'
WEIGHTS = 'checkpoint.safetensors'
CONFIG = 'engine.json'
DEVICE = torch.device('cuda:0')
SHA = re.compile(r'[a-f0-9]{64}')
DENIED = re.compile(r'(^|[\W_])(?:tests?\d*|holdouts?\d*|quarantin\w*|карантин\w*)(?=$|[\W_])', re.I)
ROW_KEYS = {'path', 'split', 'participant', 'label', 'sha256'}
SPLITS = ('train', 'validation')
JOINTS = (11, 13, 15, 12, 14, 16)  # left, then right: shoulder, elbow, wrist
JOINT_NAMES = ('left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist')
PHASES, VALUES = 48, 18
EDGE = slice(0, 10), slice(-10, None)
MIDDLE = slice(10, -10)
MIN_CONFIDENCE = .60  # Specified before reading validation; not calibrated.
SEED, EPOCHS, BATCH, LR = 601, 36, 16, .002


class Prediction(TypedDict):
    label: Literal['complete', 'incomplete', 'uncertain'] | None
    confidence: float | None
    reason: Literal['no_pose', 'low_visibility', 'partial_coverage', 'low_confidence', 'unscorable'] | None
    version: str


@dataclass(frozen=True)
class Row:
    path: str
    split: str
    participant: str
    label: int
    sha256: str


def _sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _json(raw: bytes):
    def unique(pairs):
        out = dict(pairs)
        if len(out) != len(pairs):
            raise ValueError('duplicate JSON key')
        return out

    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique)


def _dump(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _manifest_relpath(text: str) -> PurePosixPath:
    if not isinstance(text, str) or any(c in text for c in '\\:<>"|?*') or any(ord(c) < 32 for c in text):
        raise ValueError('invalid E01 media path')
    rel = PurePosixPath(text)
    if (rel.as_posix() != text or len(rel.parts) < 2 or rel.parts[0] != 'media'
            or rel.suffix.lower() != '.mp4'
            or any(p in ('.', '..') or p.rstrip(' .') != p or PureWindowsPath(p).is_reserved()
                   for p in rel.parts)
            or any(DENIED.search(p) for p in rel.parts)
            or any(re.search(r'(^|[\W_])08($|[\W_])', p) for p in rel.parts[:-1])):
        raise ValueError('invalid or forbidden E01 media path')
    return rel


def _load_manifest(path: Path, expected_sha256: str) -> tuple[list[Row], str]:
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError('E01 manifest bytes do not match the owner-pinned SHA-256')
    doc = _json(raw)
    if not isinstance(doc, dict) or doc.get('exercise') != 'E01' or not isinstance(doc.get('rows'), list):
        raise ValueError('expected an E01 manifest with rows')
    rows, paths, digests = [], set(), set()
    for i, value in enumerate(doc['rows'], 1):
        if not isinstance(value, dict) or set(value) != ROW_KEYS:
            raise ValueError(f'E01 manifest row {i}: unexpected fields')
        row = Row(**value)
        if (row.split not in SPLITS or row.participant not in ('01', '02', '03', '04', '05')
                or type(row.label) is not int or row.label not in (0, 1)
                or not isinstance(row.sha256, str) or not SHA.fullmatch(row.sha256)
                or row.sha256 == '0' * 64):
            raise ValueError(f'E01 manifest row {i}: bad split, participant, label or digest')
        rel = _manifest_relpath(row.path)
        key = rel.as_posix().casefold()
        if key in paths or row.sha256 in digests:
            raise ValueError(f'E01 manifest row {i}: duplicate clip path or SHA-256')
        paths.add(key)
        digests.add(row.sha256)
        rows.append(row)
    for split in SPLITS:
        if {r.label for r in rows if r.split == split} != {0, 1}:
            raise ValueError(f'E01 manifest {split} must contain both classes')
    return rows, actual_sha256


def _verified_clip(root: Path, row: Row) -> Path:
    path = root
    for part in _manifest_relpath(row.path).parts:
        path /= part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('E01 media must not use symlinks or junctions')
    if not path.is_file() or _sha(path) != row.sha256:
        raise ValueError('E01 clip is not a regular file with its admitted SHA-256')
    return path


def _phase_frames(path: Path):
    """Read each selected phase at most once; no frames or video written to disk."""
    cap = cv2.VideoCapture(os.fspath(path), cv2.CAP_FFMPEG)
    try:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(count) or count < 12:
            return
        targets = np.rint(np.linspace(0, int(count) - 1, PHASES)).astype(int)
        cursor = -1
        for target in np.unique(targets):
            while cursor + 1 < target:
                if not cap.grab():
                    return
                cursor += 1
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                return
            cursor += 1
            yield np.flatnonzero(targets == target), frame
    finally:
        cap.release()


class _Pose:
    """Strict shared RTMPose decoder, profiled on the first real frame."""

    def __init__(self, graph: Path):
        import onnxruntime as ort
        from ml.gpu import enable_cuda_dlls

        if not torch.cuda.is_available() or CUDA not in ort.get_available_providers():
            raise CudaUnavailable('E01 requires CUDA for RTMPose and the head')
        enable_cuda_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        options.enable_profiling = True
        # Profiles include operator metadata only; keep them outside the bundle.
        self._profile = tempfile.TemporaryDirectory(prefix='qolda-e01-ort-')
        options.profile_file_prefix = os.fspath(Path(self._profile.name) / 'rtmpose')
        session = ort.InferenceSession(os.fspath(graph), options, providers=[(CUDA, {'device_id': 0})])
        session.disable_fallback()
        if session.get_providers()[0] != CUDA:
            raise CudaUnavailable('the E01 RTMPose CUDA provider failed to load')
        ins = [(o.name, o.shape, o.type) for o in session.get_inputs()]
        outs = session.get_outputs()
        expected_outs = (('simcc_x', 384), ('simcc_y', 512))
        if (len(ins) != 1 or ins[0][1:] != ([1, 3, 256, 192], 'tensor(float)')
                or len(outs) != len(expected_outs)
                or any(o.name != name or o.type != 'tensor(float)' or len(o.shape) != 3 or o.shape[-1] != bins
                       or any(isinstance(d, int) and d != expected for d, expected in zip(o.shape[:2], (1, 17)))
                       for o, (name, bins) in zip(outs, expected_outs))):
            raise ValueError('unexpected E01 RTMPose tensor signature')
        self.pose = CudaPose(session, graph_sha256=POSE_SHA)
        self.profile: dict | None = None

    def __call__(self, frame):
        coords, mask = self.pose(frame)
        if self.profile is None:
            filename = self.pose.session.end_profiling()
            try:
                events = _json(Path(filename).read_bytes())
                counts = Counter(e.get('args', {}).get('provider', 'unknown') for e in events if e.get('cat') == 'Node')
                if not counts or set(counts) != {CUDA}:
                    raise CudaUnavailable(f'E01 RTMPose non-CUDA node events: {dict(counts)}')
                self.profile = {'by_provider': dict(counts), 'cpu_node_events': 0, 'real_frame': True}
            finally:
                self._profile.cleanup()
        return coords, mask

    def close(self):
        if self.pose.session is not None:
            if self.profile is None:
                self.pose.session.end_profiling()
            self.pose.close()
        self._profile.cleanup()


@dataclass
class Motion:
    values: np.ndarray
    mask: np.ndarray
    observed: np.ndarray  # actual visibility of each wrist/elbow/shoulder at every sampled phase
    n_phases: int         # decoder coverage only; never enters the head or baseline
    reason: str | None
    lift_return: float | None
    scoreable_arms: int = 0
    has_arm_scale: bool = False


def _motion(path: Path, pose: _Pose) -> Motion:
    xy = np.zeros((PHASES, len(JOINTS), 2), np.float32)
    seen = np.zeros((PHASES, len(JOINTS)), bool)
    phases = 0
    for slots, frame in _phase_frames(path):
        xyz, mask = pose(frame)
        picked = xyz[list(JOINTS), :2]
        xy[slots] = picked
        seen[slots] = mask[list(JOINTS)] & np.isfinite(picked).all(axis=-1)
        phases += len(slots)
    values = np.zeros((PHASES, VALUES), np.float32)
    valid = np.zeros_like(values, dtype=bool)
    if phases != PHASES:
        return Motion(values, valid, seen, phases, 'partial_coverage' if phases else 'unscorable', None)
    if not seen.any():
        return Motion(values, valid, seen, phases, 'no_pose', None)

    # Arm geometry compensates for zoom when visible. Otherwise xy is already in
    # isotropic image units (fraction of the longer side); use a fixed prior scale.
    spans = []
    for a, b, ratio in ((0, 3, 1.), (0, 1, 1.4), (3, 4, 1.4), (1, 2, 1.4), (4, 5, 1.4)):
        ok = seen[:, a] & seen[:, b]
        length = np.linalg.norm(xy[ok, a] - xy[ok, b], axis=-1)
        good = length[(length >= .025) & (length <= .75)]
        if len(good) >= 6:
            spans.append(float(np.median(good)) * ratio)
    scale = float(np.median(spans)) if spans else .25

    arm_scores = []
    for arm, k in enumerate((0, 3)):
        j = 9 * arm
        for i in range(3):
            joint = k + i
            early = seen[EDGE[0], joint]
            if early.sum() >= 3:
                base = np.median(xy[EDGE[0], joint][early], axis=0)
                values[:, j + 2*i:j + 2*i + 2] = (xy[:, joint] - base) / scale
                valid[:, j + 2*i:j + 2*i + 2] = seen[:, joint, None]
        for i, (a, b) in enumerate(((2, 1), (2, 0), (1, 0))):
            values[:, j + 6 + i] = (xy[:, k+a, 1] - xy[:, k+b, 1]) / scale
            valid[:, j + 6 + i] = seen[:, k+a] & seen[:, k+b]

        # This dataset can frame the hands but crop the elbow/shoulder. Neither
        # missing proximal joint makes a well-observed wrist unusable; their
        # actual coordinates still contribute through masked channels above.
        arm_mask = seen[:, k+2]
        if (arm_mask.sum() >= 24 and all(arm_mask[s].sum() >= 4 for s in EDGE)
                and arm_mask[MIDDLE].sum() >= 8):
            # Same wrist must go up and return; relative to its shoulder when observable.
            relative = arm_mask & seen[:, k]
            if all(relative[s].sum() >= 4 for s in EDGE) and relative[MIDDLE].sum() >= 8:
                y, ok = values[:, j + 7], relative
            else:
                y, ok = values[:, j + 5], arm_mask
            start, finish = (np.median(y[s][ok[s]]) for s in EDGE)
            peak = np.quantile(y[MIDDLE][ok[MIDDLE]], .85)
            arm_scores.append(float(min(peak - start, peak - finish)))
    values = np.where(valid, np.clip(values, -6., 6.), 0.).astype(np.float32)
    return Motion(values, valid, seen, phases, None if arm_scores else 'low_visibility',
                  max(arm_scores) if arm_scores else None, len(arm_scores), bool(spans))


class _Head(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.temporal = nn.Sequential(
            nn.Conv1d(2 * VALUES + len(JOINTS), 24, 5, padding=2, device=DEVICE), nn.GELU(),
            nn.Conv1d(24, 24, 5, dilation=2, padding=4, device=DEVICE), nn.GELU(),
            nn.Conv1d(24, 24, 5, dilation=4, padding=8, device=DEVICE), nn.GELU(),
        )
        self.predict = nn.Sequential(nn.Linear(24 * 4, 24, device=DEVICE), nn.GELU(),
                                     nn.Dropout(.15), nn.Linear(24, 1, device=DEVICE))

    def forward(self, values, mask, observed):
        z = ((values - self.mean) / self.std).clamp(-8, 8).masked_fill(~mask, 0.)
        phase = self.temporal(torch.cat((z, mask.float(), observed.float()), dim=-1).transpose(1, 2))
        # Fixed phase bins preserve direction/order (lift versus lower) without duration.
        bins = phase.reshape(phase.shape[0], 24, 4, PHASES // 4).mean(dim=-1).flatten(1)
        return self.predict(bins).squeeze(1)


def _inputs(batch: list[Motion]):
    return tuple(torch.as_tensor(np.stack([getattr(m, key) for m in batch]), device=DEVICE)
                 for key in ('values', 'mask', 'observed'))


def _fit(train: list[tuple[Row, Motion]]):
    scored = [(row, m) for row, m in train if m.reason is None]
    classes = Counter(row.label for row, _ in scored)
    if set(classes) != {0, 1}:
        raise ValueError('scorable E01 train data must contain both classes')
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    xs, masks, obs = _inputs([m for _, m in scored])
    labels = torch.tensor([r.label for r, _ in scored], dtype=torch.float32, device=DEVICE)
    count = masks.sum(dim=(0, 1)).clamp_min(1)
    mean = xs.sum(dim=(0, 1)) / count
    var = ((xs - mean).square() * masks).sum(dim=(0, 1)) / count
    model = _Head(mean, var.sqrt().clamp_min(.05))
    assert_model_on_cuda(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01, fused=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(classes[0] / classes[1], device=DEVICE))
    model.train()
    losses = []
    steps = 0
    for _ in range(EPOCHS):
        total = torch.zeros((), device=DEVICE)
        order = torch.as_tensor(rng.permutation(len(scored)), device=DEVICE)
        for idx in order.split(BATCH):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xs[idx], masks[idx], obs[idx]), labels[idx])
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
            steps += 1
        losses.append(float((total / len(scored)).item()))
    if not all(map(math.isfinite, losses)):
        raise ValueError('E01 training diverged')
    assert_model_on_cuda(model)
    return model.eval(), {'epochs': EPOCHS, 'batch': BATCH, 'optimizer_steps': steps, 'lr': LR,
                          'positive_weight': classes[0] / classes[1], 'seed': SEED,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'epoch_loss': losses}


def _baseline(train: list[tuple[Row, Motion]]) -> float:
    truth = np.array([r.label for r, m in train if m.reason is None])
    scores = np.array([m.lift_return for _, m in train if m.reason is None])
    uniq = np.unique(scores)
    mids = (uniq[:-1] + uniq[1:]) / 2
    candidates = np.r_[np.nextafter(uniq[0], -np.inf), mids, np.nextafter(uniq[-1], np.inf)]
    def ranking(t: float) -> tuple[float, float]:
        ba = .5 * (np.mean(scores[truth == 1] >= t) + np.mean(scores[truth == 0] < t))
        return float(ba), -abs(t - .25)

    return float(max(map(float, candidates), key=ranking))


def _resource(root: Path, name: str) -> Path:
    path = root / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError(f'E01 {name}: expected an ordinary bundled file')
    return path


def _load_head(path: Path, version: str) -> _Head:
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework='pt', device=0) as f:
        if f.metadata() != {'exercise': 'E01', 'version': version, 'features': FEATURES}:
            raise ValueError('wrong E01 safetensors identity')
        state = {key: f.get_tensor(key) for key in f.keys()}
    if any(t.device != DEVICE or t.dtype != torch.float32 or not torch.isfinite(t).all().item()
           for t in state.values()):
        raise ValueError('E01 safetensors must contain only finite CUDA FP32 state')
    model = _Head(torch.zeros(VALUES, device=DEVICE), torch.ones(VALUES, device=DEVICE))
    model.load_state_dict(state, strict=True)
    if torch.any(model.std < .05).item():
        raise ValueError('invalid E01 normalization')
    assert_model_on_cuda(model)
    return model.eval()


def _decision(p: float | None, reason: str | None, version: str) -> Prediction:
    if p is None:
        if reason in ('no_pose', 'low_visibility', 'partial_coverage'):
            return {'label': 'uncertain', 'confidence': None, 'reason': reason, 'version': version}
        return {'label': None, 'confidence': None, 'reason': 'unscorable', 'version': version}
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise RuntimeError('non-finite E01 prediction')
    confidence = max(p, 1 - p)
    if confidence < MIN_CONFIDENCE:
        return {'label': 'uncertain', 'confidence': confidence, 'reason': 'low_confidence', 'version': version}
    return {'label': 'complete' if p >= .5 else 'incomplete', 'confidence': confidence,
            'reason': None, 'version': version}


class _Engine:
    def __init__(self, pose: _Pose, model: _Head, version: str):
        self._pose, self._head, self._version = pose, model, version
        self._lock = Lock()
        self._head_audit: dict | None = None
        self._closed = False

    @property
    def evidence(self) -> dict:
        return {'pose': self._pose.profile, 'head': self._head_audit}

    def _probability(self, m: Motion) -> float:
        tensors = _inputs([m])
        with torch.inference_mode():
            if self._head_audit is None:
                from torch.utils._python_dispatch import TorchDispatchMode
                from torch.utils._pytree import tree_leaves

                ops = Counter()

                class CudaAudit(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs or {}
                        result = func(*args, **kwargs)
                        devices = {t.device for t in tree_leaves((args, kwargs, result)) if isinstance(t, torch.Tensor)}
                        if devices != {DEVICE}:
                            raise RuntimeError(f'E01 head operator not CUDA-only: {func}')
                        ops[str(func)] += 1
                        return result

                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record(torch.cuda.current_stream(DEVICE))
                with CudaAudit():
                    p = self._head(*tensors).sigmoid()
                after.record(torch.cuda.current_stream(DEVICE))
                after.synchronize()
                if not ops:
                    raise RuntimeError('no CUDA head operations audited')
                self._head_audit = {'device': str(DEVICE), 'tensor_ops': sum(ops.values()),
                                    'by_op': dict(ops), 'non_cuda_tensor_ops': 0,
                                    'elapsed_cuda_ms': before.elapsed_time(after), 'real_video': True}
            else:
                p = self._head(*tensors).sigmoid()
            return float(p.item())  # Scalar formatting happens on host, not the neural forward.

    def _evaluate(self, path: Path) -> tuple[Prediction, Motion, float | None]:
        with self._lock:
            if self._closed:
                raise RuntimeError('E01 engine is closed')
            try:
                motion = _motion(path, self._pose)
                p = self._probability(motion) if motion.reason is None else None
                return _decision(p, motion.reason, self._version), motion, p
            except Exception:
                self._closed = True
                raise

    def __call__(self, path: Path) -> Prediction:
        return self._evaluate(Path(path))[0]

    def close(self):
        with self._lock:
            self._closed = True
            self._pose.close()


def _open_engine(model_dir: Path) -> _Engine:
    doc = _json(_resource(model_dir, CONFIG).read_bytes())
    if not isinstance(doc, dict) or any(doc.get(k) != v for k, v in {
        'schema': SCHEMA, 'exercise': 'E01', 'version': VERSION, 'features': FEATURES,
        'device': str(DEVICE), 'dataset_doi': DATASET, 'pose_policy': 'rtmpose-body17-full-frame-conf50',
        'min_confidence': MIN_CONFIDENCE, 'source_sha256': _sha(Path(__file__)),
    }.items()):
        raise ValueError('wrong or incompatible E01 bundle')
    resources = doc.get('resources')
    if (not isinstance(doc.get('manifest_sha256'), str) or not SHA.fullmatch(doc['manifest_sha256'])
            or not isinstance(resources, dict) or set(resources) != {POSE, WEIGHTS}
            or resources[POSE] != POSE_SHA
            or not isinstance(resources[WEIGHTS], str) or not SHA.fullmatch(resources[WEIGHTS])):
        raise ValueError('invalid E01 resource hashes')
    graph, weights = (_resource(model_dir, name) for name in (POSE, WEIGHTS))
    if _sha(graph) != resources[POSE] or _sha(weights) != resources[WEIGHTS]:
        raise ValueError('E01 bundled resource digest mismatch')
    if not torch.cuda.is_available():
        raise CudaUnavailable('E01 requires cuda:0; no CPU fallback')
    head = _load_head(weights, doc['version'])
    return _Engine(_Pose(graph), head, doc['version'])


def open_engine(model_dir: Path) -> _Engine:
    """Open a private E01-only directory or raise; no shared-model fallback."""
    return _open_engine(Path(model_dir))


def _coverage(items: list[tuple[Row, Motion]]) -> dict:
    seen = np.stack([m.observed for _, m in items])
    def both(indices):
        return int(seen[:, :, indices].all(axis=-1).sum())

    return {'manifest': len(items), 'manifest_classes': dict(Counter(r.label for r, _ in items)),
            'decoded_full': sum(m.n_phases == PHASES for _, m in items),
            'scoreable': sum(m.reason is None for _, m in items),
            'scoreable_classes': dict(Counter(r.label for r, m in items if m.reason is None)),
            'scoreable_arms_per_clip': dict(Counter(m.scoreable_arms for _, m in items)),
            'scoreable_with_arm_scale': sum(m.reason is None and m.has_arm_scale for _, m in items),
            'real_visibility_over_sampled_phases': {'expected_phases': len(items) * PHASES,
                'decoded_phases': sum(m.n_phases for _, m in items),
                **{name: int(seen[:, :, i].sum()) for i, name in enumerate(JOINT_NAMES)},
                'both_wrists': both([2, 5]),
                'left_wrist_and_elbow': both([1, 2]), 'right_wrist_and_elbow': both([4, 5]),
                'both_wrists_and_elbows': both([1, 2, 4, 5])},
            'unscorable': dict(Counter(m.reason for _, m in items if m.reason is not None))}


def _metrics(truth: list[int], pred: list[int | None]) -> dict:
    # Rows are actual 0,1. Columns are predicted 0,1,reject. Missing classes => undefined BA/F1.
    rows = [[sum(a == actual and p == wanted for a, p in zip(truth, pred, strict=True))
             for wanted in (0, 1, None)] for actual in (0, 1)]
    (tn, fp, r0), (fn, tp, r1) = rows
    n0, n1 = sum(rows[0]), sum(rows[1])
    s0, s1 = tn + fp, fn + tp
    def measure(neg, pos, missed_pos):
        return {'ba': (tn / neg + tp / pos) / 2 if neg and pos else None,
                'f1_complete': 2 * tp / (2 * tp + fp + missed_pos) if neg and pos else None}
    return {'confusion_0_1_reject': rows, 'rejected': r0 + r1, 'decided': s0 + s1,
            'coverage': (s0 + s1) / len(truth), 'decided_only': measure(s0, s1, fn),
            'all_with_abstention_penalty': measure(n0, n1, fn + r1)}


def _prefix_clip_error(root: Path, row: Row, index: int) -> Path:
    try:
        return _verified_clip(root, row)
    except (OSError, ValueError):
        raise ValueError(f'{row.split} clip {index}: media path or SHA-256 verification failed') from None


def _run(manifest: Path, manifest_hash: str, graph: Path, source_commit: str, driver: str):
    rows, manifest_sha = _load_manifest(manifest, manifest_hash)
    if _sha(graph) != POSE_SHA:
        raise ValueError('use the pinned full E01 RTMPose graph')
    stack = cuda_stack()
    stack.update(opencv=cv2.__version__, numpy=np.__version__, driver=driver)
    import safetensors
    stack['safetensors'] = safetensors.__version__
    root = manifest.parent.resolve(strict=True)
    run = root / 'run'
    run.mkdir(exist_ok=True)
    if run.is_symlink() or getattr(run.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('E01 run directory must not redirect outside the manifest directory')
    if (run / CONFIG).exists():
        raise FileExistsError('E01 run already has an engine bundle; refusing to overwrite')
    lock = run / '.running'
    with lock.open('x', encoding='utf-8'):
        pass
    started = time.monotonic()
    train_rows = [r for r in rows if r.split == 'train']
    val_rows = [r for r in rows if r.split == 'validation']
    participants = {split: {r.participant for r in rows if r.split == split} for split in SPLITS}
    source_sha = _sha(Path(__file__))
    receipt = {'exercise': 'E01', 'version': VERSION, 'start_utc': _utc(), 'stage': 'train_extraction',
               'dataset_doi': DATASET, 'manifest_sha256': manifest_sha, 'pose_sha256': POSE_SHA,
               'source_sha256': source_sha, 'source_base_commit': source_commit, 'stack': stack,
               'clip_path_and_sha_disjoint': True,
               'participant_overlap_count': len(participants['train'] & participants['validation']),
               'participants_per_split': {key: len(value) for key, value in participants.items()},
               'independent_validation': False, 'object_tracking': False, 'promotion_allowed': False,
               'promotion_basis': 'dev-only validation; independent evidence and owner approval not established',
               'planned_by_split': {split: dict(Counter(r.label for r in rows if r.split == split)) for split in SPLITS}}
    try:
        _dump(run / 'receipt.json', receipt)
        pose = _Pose(graph)
        train = []
        try:
            for i, row in enumerate(train_rows, 1):
                train.append((row, _motion(_prefix_clip_error(root, row, i), pose)))
                if i % 10 == 0 or i == len(train_rows):
                    print(f'E01 train pose {i}/{len(train_rows)} ({time.monotonic()-started:.1f}s)', flush=True)
            receipt['train_pose_cuda'] = pose.profile
        finally:
            pose.close()
        receipt['train_coverage'] = _coverage(train)
        receipt['stage'] = 'fitting'
        _dump(run / 'receipt.json', receipt)

        model, receipt['fit'] = _fit(train)
        threshold = _baseline(train)
        receipt['baseline'] = {'score': 'max_same_wrist_min(peak-start,peak-finish)', 'train_threshold': threshold}
        from safetensors.torch import save_file
        # Explicit staging: can keep an actual checkpoint without exposing an unverified bundle.
        candidate = run / 'candidate'
        candidate.mkdir(exist_ok=True)
        if candidate.is_symlink() or getattr(candidate.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('E01 candidate directory must not redirect')
        weights = candidate / WEIGHTS
        tmp_weights = candidate / (WEIGHTS + '.tmp')
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                  os.fspath(tmp_weights), metadata={'exercise': 'E01', 'version': VERSION, 'features': FEATURES})
        tmp_weights.replace(weights)
        del model
        graph_copy = candidate / POSE
        tmp_graph = candidate / (POSE + '.tmp')
        shutil.copyfile(graph, tmp_graph)
        tmp_graph.replace(graph_copy)
        metadata = {'schema': SCHEMA, 'exercise': 'E01', 'version': VERSION, 'features': FEATURES,
                    'device': str(DEVICE), 'dataset_doi': DATASET, 'min_confidence': MIN_CONFIDENCE,
                    'pose_policy': 'rtmpose-body17-full-frame-conf50', 'manifest_sha256': manifest_sha,
                    'source_sha256': source_sha, 'source_base_commit': source_commit,
                    'resources': {POSE: POSE_SHA, WEIGHTS: _sha(weights)}}
        _dump(candidate / CONFIG, metadata)
        receipt.update(stage='post_load_validation', resources=metadata['resources'])
        _dump(run / 'receipt.json', receipt)

        engine = open_engine(candidate)
        evaluated, decisions, forced, baseline, reasons = [], [], [], [], Counter()
        try:
            for i, row in enumerate(val_rows, 1):
                # The public callable and this metrics path share this exact uncached whole-video inference.
                result, motion, p = engine._evaluate(_prefix_clip_error(root, row, i))
                evaluated.append((row, motion))
                if result['label'] in ('complete', 'incomplete'):
                    decisions.append(int(result['label'] == 'complete'))
                else:
                    decisions.append(None)
                    reasons[result['reason']] += 1
                forced.append(int(p >= .5) if p is not None else None)
                baseline.append(int(motion.lift_return >= threshold) if motion.lift_return is not None else None)
                if p is not None and 'post_load_example' not in receipt:
                    receipt.update(post_load_example=result, post_load_cuda=engine.evidence)
                    _dump(run / 'receipt.json', receipt)
                if i % 10 == 0 or i == len(val_rows):
                    print(f'E01 post-load validation {i}/{len(val_rows)} ({time.monotonic()-started:.1f}s)', flush=True)
            receipt['post_load_cuda'] = engine.evidence
            # Exercise the public callable itself on an already admitted, scoreable validation clip.
            for i, (row, motion) in enumerate(evaluated, 1):
                if motion.reason is None:
                    replay = engine(_prefix_clip_error(root, row, i))
                    if replay != receipt['post_load_example']:
                        raise RuntimeError('E01 post-load callable disagrees with validation execution')
                    receipt['public_callable_replay'] = replay
                    break
        finally:
            engine.close()

        y = [r.label for r, _ in evaluated]
        receipt.update(validation_coverage=_coverage(evaluated), validation={
            'baseline': _metrics(y, baseline), 'tcn': _metrics(y, decisions),
            'tcn_forced_at_0_5': _metrics(y, forced), 'tcn_refusal_reasons': dict(reasons)},
            validation_elapsed_s=round(time.monotonic() - started, 2))
        if receipt['post_load_cuda']['head'] is None or len(receipt['validation_coverage']['scoreable_classes']) != 2:
            receipt['stage'] = 'partial_validation_coverage'
            return receipt

        # Move vetted bytes into the published directory; the config goes last.
        for name in (WEIGHTS, POSE, CONFIG):
            (candidate / name).replace(run / name)
        candidate.rmdir()
        receipt['stage'] = 'validated_dev'
        return receipt
    except Exception as exc:
        receipt['failed_stage'] = receipt['stage']
        receipt['stage'] = 'blocked'
        receipt['error_type'] = type(exc).__name__
        raise
    finally:
        receipt['finished_utc'] = _utc()
        receipt['wall_s'] = round(time.monotonic() - started, 3)
        try:
            _dump(run / 'receipt.json', receipt)
        finally:
            lock.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path, help='owner-provided E01 manifest.json')
    p.add_argument('--manifest-sha256', required=True, help='owner-pinned SHA-256 of the exact manifest bytes')
    p.add_argument('--pose', required=True, type=Path, help='pinned static RTMPose graph')
    p.add_argument('--source-commit', required=True, help='repository HEAD at training; source bytes additionally hashed')
    p.add_argument('--driver', default='unrecorded', help='NVIDIA driver version from nvidia-smi')
    args = p.parse_args()
    if not re.fullmatch(r'[a-f0-9]{40}', args.source_commit):
        p.error('--source-commit requires a 40-hex Git commit id')
    if not SHA.fullmatch(args.manifest_sha256):
        p.error('--manifest-sha256 requires a lowercase 64-hex digest')
    if args.manifest.name != 'manifest.json' or args.manifest.parent.name != 'E01':
        p.error('--manifest must name the owner-provided E01/manifest.json')
    result = _run(args.manifest, args.manifest_sha256, args.pose, args.source_commit, args.driver)
    print(f"E01: {result['stage']}; see private run/receipt.json", flush=True)
    if result['stage'] != 'validated_dev':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
