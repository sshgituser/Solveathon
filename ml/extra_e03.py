"""E03 RGB-motion research engine. Video decode is on CPU; the CNN is CUDA-only."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Callable, TypedDict


SLOT = 'E03'
MODEL = 'rgb-motion-cnn-v1'
SEED = 20261125
FRAMES, SIZE = 24, 96
BATCH, EPOCHS = 12, 32
MARGIN = 0.70
CLASSES = ('incomplete', 'complete')
TRAIN_IDS = frozenset(('02', '03', '04', '06'))
VAL_ID = '05'
SHA256 = re.compile(r'[0-9a-f]{64}')
WEIGHTS = 'checkpoint.safetensors'
METADATA = 'model.json'
MAX_FRAME_COUNT = 12000


class Result(TypedDict):
    label: str | None
    confidence: float | None
    reason: str | None
    version: str


def _result(version: str, label: str | None = None, confidence: float | None = None,
            reason: str | None = None) -> Result:
    return {'label': label, 'confidence': confidence, 'reason': reason, 'version': version}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


@dataclass(frozen=True)
class Row:
    path: Path
    split: str
    participant: str
    label: int
    sha256: str


def _load_manifest(manifest: Path, approved_sha: str) -> tuple[list[Row], str]:
    """All lexical/schema checks precede media filesystem access."""
    if manifest.name != 'manifest.json':
        raise ValueError('expected the owner-supplied E03 manifest.json')
    raw = manifest.read_bytes()
    if not SHA256.fullmatch(approved_sha) or _sha(raw) != approved_sha:
        raise ValueError('approved E03 manifest SHA-256 mismatch')
    obj = json.loads(raw)
    required = {'exercise', 'rows'}
    provenance = {'schema': 'qolda-additional-development-v1', 'source': '10.17632/49h9dcwx5v.1',
                  'rights': 'CC BY 4.0'}
    if not isinstance(obj, dict) or obj.get('exercise') != SLOT:
        raise ValueError('not the E03 manifest schema')
    if obj.keys() != required and (obj.keys() != required | provenance.keys() or
                                   any(obj.get(k) != v for k, v in provenance.items())):
        raise ValueError('unrecognized E03 manifest fields or provenance')
    if not isinstance(obj['rows'], list) or not obj['rows']:
        raise ValueError('E03 manifest has no rows')
    paths, digests, rows = set(), set(), []
    for i, item in enumerate(obj['rows']):
        def reject(reason):
            raise ValueError(f'manifest row {i + 1}: {reason}')

        if not isinstance(item, dict) or item.keys() != {'path', 'split', 'participant', 'label', 'sha256'}:
            reject('unexpected schema')
        p, split, subject, label, digest = (item[k] for k in ('path', 'split', 'participant', 'label', 'sha256'))
        if not (type(label) is int and label in (0, 1) and isinstance(digest, str) and SHA256.fullmatch(digest)):
            reject('invalid label or SHA-256')
        if not isinstance(subject, str) or not ((split == 'train' and subject in TRAIN_IDS) or
                                                 (split == 'validation' and subject == VAL_ID)):
            reject('participant/split not approved')
        if not isinstance(p, str) or not p.startswith('media/') or '\\' in p or ':' in p or '\x00' in p:
            reject('path must be relative to the manifest under media/')
        parts = p.split('/')
        if any(part in ('', '.', '..') or part.endswith((' ', '.')) for part in parts):
            reject('invalid path component')
        forbidden = r'(?i)(?:^|[^a-z0-9])(?:test|quarantine|карантин|id[_-]?0?8)(?:$|[^a-z0-9])'
        if (PurePosixPath(p).suffix.lower() != '.mp4' or any(re.search(forbidden, part) for part in parts)
                or any(part == '08' for part in parts[:-1])):
            reject('unapproved media path')
        if p.casefold() in paths or digest in digests:
            reject('duplicate path or content digest')
        paths.add(p.casefold())
        digests.add(digest)
        rows.append(Row(manifest.parent.joinpath(*parts), split, subject, label, digest))
    for split in ('train', 'validation'):
        if {r.label for r in rows if r.split == split} != {0, 1}:
            raise ValueError(f'{split} must contain both labels')
    return rows, approved_sha


def _check_local_files(root: Path, rows: list[Row]) -> None:
    """Do not follow symbolic links, Windows junctions or hard-linked media."""
    checked_dirs = set()
    for row in rows:
        parts = row.path.relative_to(root).parts
        p = root
        for i, part in enumerate(parts):
            p = p / part
            if p in checked_dirs:
                continue
            s = p.lstat()
            if stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 0x400:
                raise ValueError('media link or Windows reparse point not permitted')
            if i == len(parts) - 1:
                if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
                    raise ValueError('media must be an ordinary unlinked file')
            else:
                if not stat.S_ISDIR(s.st_mode):
                    raise ValueError('media parent is not a directory')
                checked_dirs.add(p)


def _decode(path: Path, cv, np):
    """Resize 24 equally spaced samples spanning the full clip; never use FPS as a feature."""
    cap = cv.VideoCapture(str(path), cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened():
            return None
        count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or not FRAMES <= count <= MAX_FRAME_COUNT:
            return None
        indices = iter(np.linspace(0, int(count) - 1, FRAMES).round().astype('int32'))
        next_index = next(indices)
        result = []
        for i in range(int(count)):
            if not cap.grab():
                return None
            if i == next_index:
                ok, frame = cap.retrieve()
                if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    return None
                small = cv.resize(frame, (SIZE, SIZE), interpolation=cv.INTER_AREA)
                result.append(cv.cvtColor(small, cv.COLOR_BGR2RGB))
                next_index = next(indices, None)
        if len(result) != FRAMES or cap.grab():
            return None
        return np.stack(result)
    finally:
        cap.release()


def _load_one(row: Row, cv, np):
    if _file_sha(row.path) != row.sha256:
        raise ValueError('approved media SHA-256 mismatch')
    try:
        return _decode(row.path, cv, np)
    except cv.error:
        return None


def _net(torch):
    from torch import nn

    class MotionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Sequential(
                nn.Conv2d(6, 16, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
                nn.AvgPool2d(4, 4),
            )
            self.temporal = nn.Sequential(
                nn.Conv1d(288, 48, 3, padding=1, bias=False), nn.BatchNorm1d(48), nn.SiLU(),
                nn.Conv1d(48, 48, 3, padding=2, dilation=2, bias=False), nn.BatchNorm1d(48), nn.SiLU(),
                nn.AvgPool1d(8, 8), nn.Flatten(), nn.Dropout(0.20), nn.Linear(144, 1),
            )

        def forward(self, x):
            n, t, c, h, w = x.shape
            x = self.spatial(x.reshape(n * t, c, h, w))
            return self.temporal(x.reshape(n, t, -1).transpose(1, 2)).squeeze(1)

    return MotionNet()


def _motion(rgb, torch):
    """Static RGB and whole-frame channel shifts never reach the learned model."""
    x = rgb.permute(0, 1, 4, 2, 3).float().div(255)
    # The indexed CUDA median is disallowed in deterministic mode. The lower
    # middle sorted value is the same statistic without that kernel.
    middle = x.shape[1] // 2
    static = x.sort(dim=1, stable=True).values[:, middle - 1:middle]
    residual = x - static
    residual = residual - residual.mean(dim=(-1, -2), keepdim=True)
    delta = residual[:, 1:] - residual[:, :-1]
    delta = torch.cat((torch.zeros_like(residual[:, :1]), delta), dim=1)
    return residual, delta, static[:, 0]


def _stack_motion(residual, delta, torch):
    return torch.cat((residual, delta), dim=2).mul(4).clamp(-2, 2)


def _motion_quality(residual):
    """Scene-level rejectors, not wrist visibility or angle estimates."""
    active = (residual.abs().mean(dim=2) >= 0.055).float().mean(dim=(-2, -1))
    enough = (active >= 0.002).sum(dim=1) >= 4
    global_motion = (active >= 0.45).sum(dim=1) >= 8
    return enough, global_motion


def _features(residual, delta, static, torch):
    from torch.nn import functional as F

    # Mirror-invariant motion statistics (level, range and timing over thirds).
    features = []
    for field in (residual, delta):
        magnitude = field.abs().mean(dim=2)
        a = F.avg_pool2d(magnitude, 32).flatten(2)
        a = a.reshape(a.shape[0], FRAMES, 3, 3)
        a = torch.stack(((a[..., 0] + a[..., 2]) / 2, a[..., 1]), dim=-1).flatten(2)
        occupancy = (magnitude > 0.055).float().mean(dim=(-1, -2)).unsqueeze(2)
        a = torch.cat((a, occupancy), dim=2)
        thirds = a.reshape(a.shape[0], 3, FRAMES // 3, -1)
        features.extend((thirds.mean(dim=2).flatten(1), a.amax(dim=1), a.std(dim=1, unbiased=False)))
    # Deliberately static negative control; may include a stationary person as well as background.
    return torch.cat(features, dim=1), F.adaptive_avg_pool2d(static, (6, 6)).flatten(1)


def _augment(x, rng, torch):
    # One transform per description of a temporal sequence, never a different flip per frame.
    if rng.random() < 0.5:
        x = x.flip(-1)
    dy, dx = rng.integers(-5, 6, size=2).tolist()
    if dx or dy:
        x = torch.roll(x, (dy, dx), dims=(-2, -1))
        if dx > 0:
            x[..., :dx] = 0
        elif dx < 0:
            x[..., dx:] = 0
        if dy > 0:
            x[..., :dy, :] = 0
        elif dy < 0:
            x[..., dy:, :] = 0
    gain = torch.as_tensor(rng.uniform(0.85, 1.15, size=(len(x), 1, 1, 1, 1)),
                           device=x.device, dtype=x.dtype)
    return x * gain


def _cuda_only(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('E03 needs CUDA; CPU fallback is disabled')
    return torch.device('cuda:0')


def _assert_cuda(net, device):
    if any(t.device != device for t in (*net.parameters(), *net.buffers())):
        raise RuntimeError('E03 CUDA device placement failed')


def _views(net, x, torch):
    if x.device.type != 'cuda':
        raise RuntimeError('CPU model input refused')
    return torch.stack((net(x).sigmoid(), net(x.flip(-1)).sigmoid()), dim=1)


def _decide(version, views, enough=True, global_motion=False) -> Result:
    if not enough or global_motion:
        return _result(version, 'uncertain', reason='partial_coverage' if global_motion else 'low_visibility')
    a, b = map(float, views)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (a, b)):
        return _result(version, reason='unscorable')
    if (a >= 0.5) != (b >= 0.5):
        return _result(version, 'uncertain', reason='low_confidence')
    score = min(max(a, 1 - a), max(b, 1 - b))
    if score < MARGIN:
        return _result(version, 'uncertain', reason='low_confidence')
    return _result(version, CLASSES[int(a >= 0.5)], score)


def _open(model_dir: Path):
    import torch
    from safetensors.torch import load

    device = _cuda_only(torch)
    meta = json.loads((model_dir / METADATA).read_bytes())
    payload = (model_dir / WEIGHTS).read_bytes()
    digest = _sha(payload)
    version = MODEL + '+' + digest[:12]
    if any(meta.get(k) != v for k, v in dict(exercise=SLOT, model=MODEL, weights_sha256=digest,
                                             source_sha256=_file_sha(Path(__file__)), version=version,
                                             device='cuda:0').items()):
        raise ValueError('wrong E03 bundle, source, or weights SHA-256')
    net = _net(torch).to(device)
    tensors = load(payload)
    expected = net.state_dict()
    if tensors.keys() != expected.keys() or any(t.dtype != expected[k].dtype or t.shape != expected[k].shape
                                                for k, t in tensors.items()):
        raise ValueError('wrong E03 checkpoint tensors')
    if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('nonfinite E03 weights')
    net.load_state_dict(tensors, strict=True)
    _assert_cuda(net, device)
    return net.eval(), version, device


def open_engine(model_dir: Path) -> Callable[[Path], Result]:
    """Open the local research engine. Unavailable engines return no classification.

    Read receipt.json separately: this factory also permits local replay of an
    experimental (non-promoted) checkpoint. `confidence` is not calibrated.
    """
    try:
        import cv2 as cv
        import numpy as np
        import torch
        from safetensors import SafetensorError
    except (ImportError, OSError):
        return lambda path: _result(MODEL, reason='unscorable')
    try:
        net, version, device = _open(Path(model_dir))
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError, SafetensorError):
        return lambda path: _result(MODEL, reason='unscorable')

    healthy = True

    def predict(path: Path) -> Result:
        nonlocal healthy
        if not healthy:
            return _result(version, reason='unscorable')
        try:
            rgb = _decode(Path(path), cv, np)
        except (OSError, TypeError, ValueError, cv.error):
            rgb = None
        if rgb is None:
            return _result(version, 'uncertain', reason='unscorable')
        try:
            with torch.inference_mode():
                _assert_cuda(net, device)
                residual, delta, _ = _motion(torch.as_tensor(rgb, device=device).unsqueeze(0), torch)
                enough, global_motion = (bool(x.item()) for x in _motion_quality(residual))
                if not enough or global_motion:
                    return _decide(version, (0, 0), enough, global_motion)
                views = _views(net, _stack_motion(residual, delta, torch), torch)[0].tolist()
                result = _decide(version, views)
                if result['label'] is None:
                    healthy = False
                return result
        except RuntimeError:
            healthy = False
            return _result(version, reason='unscorable')

    return predict


def _metrics(labels, choices):
    totals, cm, rejected = [0, 0], [[0, 0], [0, 0]], Counter()
    for y, choice in zip(labels, choices, strict=True):
        totals[y] += 1
        if type(choice) is int:
            cm[y][choice] += 1
        else:
            rejected[str(choice)] += 1
    scored = [sum(row) for row in cm]
    hits = [cm[i][i] for i in range(2)]
    f1 = [2 * hits[i] / d if (d := scored[i] + cm[0][i] + cm[1][i]) else 0.0 for i in range(2)]
    return dict(n=sum(totals), scored=sum(scored), class_total=totals, class_scored=scored,
                coverage=sum(scored) / sum(totals), class_coverage=[scored[i] / totals[i] if totals[i] else None for i in range(2)],
                balanced_accuracy=sum(hits[i] / scored[i] for i in range(2)) / 2 if all(scored) else None,
                unconditional_balanced_recall=sum(hits[i] / totals[i] for i in range(2)) / 2 if all(totals) else None,
                f1_macro=sum(f1) / 2 if sum(scored) else None, f1_complete=f1[1] if sum(scored) else None,
                confusion=cm, abstentions=dict(rejected))


def _linear(x, y, weights, torch):
    from torch.nn import functional as F
    from torch import nn

    mean, std = x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(0.0001)
    x = ((x - mean) / std).clamp(-8, 8)
    model = nn.Linear(x.shape[1], 1, device=x.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay=0.08)
    for _ in range(160):
        opt.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=weights)
        loss.backward()
        opt.step()

    @torch.inference_mode()
    def predict(features):
        return model(((features - mean) / std).clamp(-8, 8)).squeeze(1).sigmoid()

    return predict


def _profile_cuda(net, x, torch):
    from torch.utils._pytree import tree_leaves
    from torch.utils._python_dispatch import TorchDispatchMode

    ops = Counter()
    cpu_count = 0

    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal cpu_count
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            for t in tree_leaves((args, kwargs, out)):
                if isinstance(t, torch.Tensor) and t.device != x.device:
                    cpu_count += 1
            ops[str(func)] += 1
            return out

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode(), Audit():
        start.record()
        out = _views(net, x, torch)
        end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    if cpu_count or not ops or elapsed <= 0 or not torch.isfinite(out).all().item():
        raise RuntimeError('CUDA forward dispatch audit failed')
    return dict(device=str(x.device), graph_operator_calls=sum(ops.values()), graph_ops=dict(ops),
                non_cuda_tensor_operands_or_outputs=cpu_count, cuda_event_elapsed_ms=elapsed)


def _counts(rows, clips):
    return {s: dict(class_total=[sum(r.split == s and r.label == c for r in rows) for c in range(2)],
                    class_decoded=[sum(r.split == s and r.label == c and x is not None
                                       for r, x in zip(rows, clips, strict=True)) for c in range(2)],
                    participant_counts=dict(sorted(Counter(r.participant for r in rows if r.split == s).items())))
            for s in ('train', 'validation')}


def _environment(torch, cv, np):
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                         text=True, timeout=5).strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        driver = None
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                opencv=cv.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)), driver=driver,
                vram_free_total_bytes=list(torch.cuda.mem_get_info(0)))


def _fit_and_validate(rows, clips, run, receipt, torch, np, device):
    from safetensors.torch import save
    from torch.nn import functional as F

    decoded = [i for i, x in enumerate(clips) if x is not None]
    y = torch.tensor([rows[i].label for i in decoded], dtype=torch.float32, device=device)
    train = [j for j, i in enumerate(decoded) if rows[i].split == 'train']
    val = [j for j, i in enumerate(decoded) if rows[i].split == 'validation']
    y_train = y[train]
    if set(y_train.tolist()) != {0.0, 1.0} or set(y[val].tolist()) != {0.0, 1.0}:
        raise ValueError('both classes must be decodable in each split')
    count = torch.bincount(y_train.long(), minlength=2)
    class_weights = len(train) / (2 * count)
    weights = class_weights[y_train.long()]
    rng = np.random.default_rng(SEED)
    net = _net(torch).to(device)
    _assert_cuda(net, device)
    initial = net.spatial[0].weight.detach().clone()
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=0.00015)
    steps, losses = 0, []
    seen = set()
    dyn_train, bg_train, x_train, good_train = [], [], [], []
    # Validation features remain untouched until the final checkpoint is frozen.
    for start in range(0, len(train), BATCH):
        batch = train[start:start + BATCH]
        pixels = torch.as_tensor(np.stack([clips[decoded[j]] for j in batch]), device=device)
        residual, delta, static = _motion(pixels, torch)
        dyn, bg = _features(residual, delta, static, torch)
        dyn_train.append(dyn)
        bg_train.append(bg)
        enough, whole = _motion_quality(residual)
        good_train.extend((enough & ~whole).tolist())
        x_train.append(_stack_motion(residual, delta, torch))
    x_train = torch.cat(x_train)
    baseline = _linear(torch.cat(dyn_train), y_train, weights, torch)
    static_probe = _linear(torch.cat(bg_train), y_train, weights, torch)
    for epoch in range(EPOCHS):
        net.train()
        order = rng.permutation(len(train)).tolist()
        total_loss = torch.zeros((), device=device)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            opt.zero_grad(set_to_none=True)
            logits = net(_augment(x_train[batch], rng, torch))
            loss = F.binary_cross_entropy_with_logits(logits, y_train[batch], weight=weights[batch])
            loss.backward()
            opt.step()
            steps += 1
            seen.update(rows[decoded[train[j]]].participant for j in batch)
            total_loss += loss.detach() * len(batch)
        sched.step()
        losses.append(float(total_loss.item()) / len(train))
        if not math.isfinite(losses[-1]):
            raise RuntimeError('nonfinite training loss')
        if epoch == 0 or (epoch + 1) % 8 == 0:
            print(f'E03 epoch {epoch+1}/{EPOCHS}; train_loss={losses[-1]:.5f}', flush=True)
    updated = int((net.spatial[0].weight != initial).sum().item())
    if not updated or not steps:
        raise RuntimeError('no trained CNN weights')
    net.eval()
    weights_bytes = save({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()})
    digest = _sha(weights_bytes)
    version = MODEL + '+' + digest[:12]
    _write(run / WEIGHTS, weights_bytes)
    _write(run / METADATA, _json_bytes(dict(exercise=SLOT, model=MODEL, device=str(device), version=version,
                                             weights_sha256=digest, source_sha256=receipt['source_sha256'])))
    receipt['training'] = dict(device=str(device), seed=SEED, epochs=EPOCHS, steps=steps, batch_size=BATCH,
                               checkpoint_epoch=EPOCHS, parameters=sum(p.numel() for p in net.parameters()),
                               first_conv_weights_changed=updated, class_weights=class_weights.tolist(),
                               train_loss=losses, participants_exposed=sorted(seen), train_quality_pass=sum(good_train))
    receipt['weights_sha256'], receipt['version'] = digest, version
    _write(run / 'receipt.json', _json_bytes(receipt))

    labels = [r.label for r in rows if r.split == 'validation']
    candidates = {name: [] for name in ('cnn_raw', 'cnn_selective', 'dynamic_raw', 'dynamic_selective',
                                       'static_probe', 'cnn_shuffled_time')}
    expected = {}
    raw_probabilities = {}
    quality_pass = {}
    val_by_index = {decoded[j]: j for j in val}
    versioned_indices = [i for i, r in enumerate(rows) if r.split == 'validation']
    permutation = np.random.default_rng(SEED + 1).permutation(FRAMES).tolist()
    with torch.inference_mode():
        for idx in versioned_indices:
            if idx not in val_by_index:
                for choices in candidates.values():
                    choices.append('unscorable')
                expected[idx] = _result(version, 'uncertain', reason='unscorable')
                continue
            rgb = torch.as_tensor(clips[idx], device=device).unsqueeze(0)
            residual, delta, static = _motion(rgb, torch)
            enough, whole = (bool(x.item()) for x in _motion_quality(residual))
            quality_pass[idx] = enough and not whole
            inputs = _stack_motion(residual, delta, torch)
            scores = _views(net, inputs, torch)[0].tolist()
            result = _decide(version, scores, enough, whole)
            expected[idx] = result
            raw_probabilities[idx] = float(sum(scores) / 2)
            candidates['cnn_raw'].append(int(raw_probabilities[idx] >= 0.5))
            candidates['cnn_selective'].append(CLASSES.index(result['label']) if result['label'] in CLASSES else result['reason'])
            dynamics, background = _features(residual, delta, static, torch)
            dp, sp = float(baseline(dynamics).item()), float(static_probe(background).item())
            bd = _decide(version, (dp, dp), enough, whole)
            candidates['dynamic_raw'].append(int(dp >= 0.5))
            candidates['dynamic_selective'].append(CLASSES.index(bd['label']) if bd['label'] in CLASSES else bd['reason'])
            candidates['static_probe'].append(int(sp >= 0.5))
            a, b, _ = _motion(rgb[:, permutation], torch)
            shuffled_score = float(_views(net, _stack_motion(a, b, torch), torch).mean().item())
            candidates['cnn_shuffled_time'].append(int(shuffled_score >= 0.5))
    metrics = {name: _metrics(labels, c) for name, c in candidates.items()}

    reloaded, actual_version, actual_device = _open(run)
    predictor = open_engine(run)
    max_difference, mismatches = 0.0, 0
    # Re-decode each original approved complete video file (not saved frames or a feature cache).
    for idx in versioned_indices:
        if _file_sha(rows[idx].path) != rows[idx].sha256:
            raise ValueError('approved video changed before reload verification')
        actual, before = predictor(rows[idx].path), expected[idx]
        mismatches += any(actual[k] != before[k] for k in ('label', 'reason', 'version'))
        if before['confidence'] is not None and actual['confidence'] is not None:
            max_difference = max(max_difference, abs(before['confidence'] - actual['confidence']))
    probe_index = next((i for i in versioned_indices if quality_pass.get(i)), decoded[val[0]])
    with torch.inference_mode():
        rgb = torch.as_tensor(clips[probe_index], device=device).unsqueeze(0)
        a, b, _ = _motion(rgb, torch)
        probe = _stack_motion(a, b, torch)
        cuda_profile = _profile_cuda(reloaded, probe, torch)
        raw_reloaded = float(_views(reloaded, probe, torch).mean().item())
    raw_difference = abs(raw_probabilities[probe_index] - raw_reloaded)
    reload_ok = (mismatches == 0 and max_difference <= 1e-6 and raw_difference <= 1e-6 and
                 actual_version == version and actual_device == device)
    receipt['reload'] = dict(ok=reload_ok, n_full_validation_videos=len(versioned_indices),
                             decision_mismatches=mismatches, max_confidence_difference=max_difference,
                             probe_raw_probability_difference=raw_difference, cuda_audit=cuda_profile)
    receipt['validation'] = metrics
    raw, dyn = metrics['cnn_raw'], metrics['dynamic_raw']
    sel, dynsel, bg = metrics['cnn_selective'], metrics['dynamic_selective'], metrics['static_probe']
    gate = {
        'raw_beats_dynamic': raw['balanced_accuracy'] > dyn['balanced_accuracy'] + 1e-9 and raw['f1_macro'] >= dyn['f1_macro'],
        'coverage_adjusted_beats_dynamic': sel['unconditional_balanced_recall'] > dynsel['unconditional_balanced_recall'] + 1e-9,
        'coverage': sel['coverage'] >= 0.70 and min(sel['class_coverage']) >= 0.50,
        'above_chance': sel['unconditional_balanced_recall'] > 0.50,
        'beyond_static_control': raw['balanced_accuracy'] >= bg['balanced_accuracy'] + 0.05,
        'all_train_participants_exposed': set(seen) == TRAIN_IDS,
        'reload': reload_ok,
    }
    receipt.update(status='trained', promotion_checks=gate, promotion_allowed=all(gate.values()))
    _write(run / 'receipt.json', _json_bytes(receipt))
    print(json.dumps(dict(status=receipt['status'], version=version, promotion_allowed=receipt['promotion_allowed'],
                          validation=metrics, reload=receipt['reload'], promotion_checks=gate), sort_keys=True), flush=True)
    if not reload_ok:
        raise RuntimeError('E03 reload parity failed')


def train(manifest: Path, approved_sha: str):
    manifest = manifest.absolute()
    rows, manifest_sha = _load_manifest(manifest, approved_sha)
    _check_local_files(manifest.parent, rows)
    run = manifest.parent / 'run'
    if run.exists() and (run.is_symlink() or getattr(run.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('run must not be a link or Windows reparse point')
    if any((run / name).exists() for name in (WEIGHTS, METADATA, 'receipt.json', WEIGHTS + '.tmp', METADATA + '.tmp', 'receipt.json.tmp')):
        raise FileExistsError('E03 run contains artifacts; retain/move them before starting a new run')

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import cv2 as cv
    import numpy as np
    import torch

    device = _cuda_only(torch)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cv.setNumThreads(1)
    print(f'E03 approved manifest accepted: {len(rows)} rows', flush=True)
    clips, sample_hashes = [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, arr in enumerate(pool.map(lambda row: _load_one(row, cv, np), rows)):
            clips.append(arr)
            if arr is not None:
                sample = _sha(arr.tobytes())
                if sample_hashes.setdefault(sample, rows[i].split) != rows[i].split:
                    raise ValueError('exact decoded input duplicate across train and validation')
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                print(f'E03 decoded {i+1}/{len(rows)}', flush=True)
    run.mkdir(exist_ok=True)
    receipt = dict(status='training', exercise=SLOT, model=MODEL, device=str(device),
                   class_order=list(CLASSES), source_sha256=_file_sha(Path(__file__)),
                   manifest_sha256=manifest_sha, environment=_environment(torch, cv, np),
                   counts=_counts(rows, clips), promotion_allowed=False, confidence_floor=MARGIN,
                   frame_count=FRAMES, frame_size=SIZE, split_policy='participant-disjoint-validation-only')
    with (run / 'receipt.json').open('xb') as file:
        file.write(_json_bytes(receipt))
    try:
        _fit_and_validate(rows, clips, run, receipt, torch, np, device)
    except Exception:
        receipt.update(status='failed', promotion_allowed=False)
        _write(run / 'receipt.json', _json_bytes(receipt))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cmd = sub.add_parser('train', help='fit only the explicitly approved E03 manifest')
    cmd.add_argument('--manifest', required=True, type=Path)
    cmd.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    train(args.manifest, args.manifest_sha256)


if __name__ == '__main__':
    main()
