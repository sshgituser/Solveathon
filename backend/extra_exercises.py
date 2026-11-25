"""Independent, opt-in local file adapters for E01/E03; never the E02 registry."""
from __future__ import annotations

import hashlib
import importlib
import json
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from backend.contract import ClipRejected
from backend.runtime import probe_frames

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qolda.extra-exercise.v1'
EXERCISES = ('E01', 'E03')
REASONS = {'no_pose', 'low_visibility', 'partial_coverage', 'low_confidence', 'unscorable'}


def private_directory(path):
    """Require an existing owner-managed directory outside this and any enclosing Git checkout."""
    path = Path(path).resolve(strict=True)
    if not path.is_dir() or path.is_relative_to(ROOT):
        raise ValueError('private directory must be outside the checkout')
    if any((parent / '.git').exists() for parent in (path, *path.parents)):
        raise ValueError('private directory must not be inside a Git checkout')
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate receipt field')
        result[key] = value
    return result


def _receipt(run, exercise):
    receipt_path = run / 'promotion.json'
    checkpoint = run / 'checkpoint.safetensors'
    for path in (receipt_path, checkpoint):
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True).parent != run:
            raise ValueError('missing local promotion file')
    if receipt_path.stat().st_size > 16384:
        raise ValueError('receipt too large')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(receipt, dict) or set(receipt) != {'schema', 'exercise', 'owner_id', 'decision', 'checkpoint', 'sha256', 'low_confidence_below', 'file_platforms'}:
        raise ValueError('invalid receipt')
    if (receipt['schema'] != 'qolda-extra-promotion-v1' or receipt['exercise'] != exercise
            or receipt['owner_id'] != 'data-owner' or receipt['decision'] != 'approved'
            or receipt['checkpoint'] != 'checkpoint.safetensors'
            or not isinstance(receipt['sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256'])):
        raise ValueError('promotion not approved')
    margin = receipt['low_confidence_below']
    if type(margin) not in (int, float) or not isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError('invalid confidence policy')
    platforms = receipt['file_platforms']
    if not isinstance(platforms, dict) or set(platforms) != {'web', 'windows'} or any(type(v) is not bool for v in platforms.values()):
        raise ValueError('invalid platform approvals')
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != receipt['sha256']:
        raise ValueError('checkpoint digest mismatch')
    return margin, platforms


def wire_result(exercise, prediction, margin):
    if not isinstance(prediction, dict) or set(prediction) != {'label', 'confidence', 'reason', 'version'}:
        raise ValueError('invalid predictor envelope')
    label, confidence, reason, version = (prediction[k] for k in ('label', 'confidence', 'reason', 'version'))
    if label not in ('complete', 'incomplete', 'uncertain', None):
        raise ValueError('invalid label')
    if confidence is not None and (type(confidence) not in (int, float) or not isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError('invalid confidence')
    if reason is not None and reason not in REASONS:
        raise ValueError('invalid reason')
    if not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+-]{0,95}', version):
        raise ValueError('invalid model version')
    if reason == 'no_pose':
        result, confidence = 'no_pose', None
    elif reason in ('low_visibility', 'partial_coverage', 'low_confidence', 'unscorable'):
        result = 'uncertain'
    elif label in ('complete', 'incomplete') and confidence is not None:
        result = 'uncertain' if confidence < margin else label
        reason = 'low_confidence' if result == 'uncertain' else None
    elif label in ('uncertain', None):
        result, reason = 'uncertain', 'unscorable'
    else:
        raise ValueError('scored label needs confidence')
    return {'version': VERSION, 'exercise': exercise, 'model_version': version,
            'result': result, 'confidence': confidence, 'reason': reason}


class ExtraEngines:
    def __init__(self, scratch=None, ffprobe='ffprobe'):
        self.scratch = scratch
        self.ffprobe = ffprobe
        self.engines = {}
        self.margins = {}
        self.platforms = {}

    def ready(self, exercise):
        return callable(self.engines.get(exercise))

    def register(self, root):
        root = private_directory(root)
        self.scratch = private_directory(self.scratch)
        for exercise in EXERCISES:
            try:
                run = root / exercise / 'run'
                # Resolve and reject directory links/junction escapes and cross-exercise aliases.
                if (root / exercise).is_symlink() or run.is_symlink() or run.resolve(strict=True) != run:
                    raise ValueError('invalid run directory')
                if not run.is_dir() or (root / exercise / '.git').exists() or (run / '.git').exists():
                    raise ValueError('missing run directory')
                margin, platforms = _receipt(run, exercise)
                if not any(platforms.values()):
                    raise ValueError('no approved file platform')
                module = importlib.import_module({'E01': 'ml.extra_e01', 'E03': 'ml.extra_e03_service'}[exercise])
                engine = module.open_engine(run)
                if not callable(engine):
                    raise ValueError('engine not callable')
                self.engines[exercise] = engine
                self.margins[exercise] = margin
                self.platforms[exercise] = platforms
            except Exception:
                # One rejected/uninstalled adapter must not affect the other or E02.
                self.engines.pop(exercise, None)
                self.margins.pop(exercise, None)
                self.platforms.pop(exercise, None)

    def analyze(self, exercise, body, mime):
        engine = self.engines[exercise]
        with TemporaryDirectory(prefix='qolda-extra-', dir=self.scratch) as temp:
            clip = Path(temp) / ('clip.mp4' if mime == 'video/mp4' else 'clip.webm')
            clip.write_bytes(body)
            probe_frames(clip, mime, self.ffprobe)
            try:
                return wire_result(exercise, engine(clip), self.margins[exercise])
            except ClipRejected:
                raise
            except Exception:
                # Do not keep claiming readiness after adapter/device/contract failure.
                self.engines.pop(exercise, None)
                raise


def catalog(engines, extra):
    e02_file = any(callable(engines.get(name)) for name in ('scratch', 'transfer'))
    e02_live = getattr(engines, 'live', None) is not None and callable(engines.get('scratch'))
    def platforms(web_file, web_live, windows_file=False, windows_live=False):
        return {'web': {'file': web_file, 'live': web_live},
                'windows': {'file': windows_file, 'live': windows_live},
                'android': {'file': False, 'live': False}}
    def additional(exercise):
        approved = extra.platforms.get(exercise, {})
        ready = extra.ready(exercise)
        return platforms(ready and approved.get('web', False), False,
                         ready and approved.get('windows', False), False)
    return {'version': VERSION, 'exercises': [
        {'id': 'E01', 'role': 'additional', 'variants': ['single'], 'readiness': additional('E01')},
        {'id': 'E02', 'role': 'primary', 'variants': list(('scratch', 'transfer')), 'readiness': platforms(e02_file, e02_live)},
        {'id': 'E03', 'role': 'additional', 'variants': ['single'], 'readiness': additional('E03')},
    ]}
