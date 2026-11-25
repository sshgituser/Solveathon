"""Replay canonical E03 development validation in packaged Python; never register or promote a model."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

MANIFEST = '125abc289062c15a26a490bbaf7306d73767fcaddbea6c62fff81282a0d2b7ea'
CLASSES = ('incomplete', 'complete')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def private(path):
    p = path.absolute()
    require(p.resolve(strict=True) == p and not any((d / '.git').exists() for d in (p, *p.parents)),
            'expected a private path outside Git without redirects')
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, limit):
    path = private(path)
    require(path.is_file() and path.stat().st_nlink == 1 and 0 < path.stat().st_size <= limit,
            'unexpected input file')
    return path.read_bytes()


def receive(args):
    engine = Path(sys.executable).resolve().parents[1]
    require(sys.flags.isolated and (engine / 'python/python.exe').resolve() == Path(sys.executable).resolve()
            and all(Path(p).resolve().is_relative_to(engine) for p in sys.path), 'use engine-win64/python/python.exe')
    from backend import runtime
    from ml import extra_e03, extra_e03_service
    require(all(m.__file__ is not None and Path(m.__file__).resolve().is_relative_to(engine / 'app')
                for m in (runtime, extra_e03, extra_e03_service)),
            'module loaded outside the package')
    raw = read(args.manifest, 1024 * 1024)
    require(sha(raw) == MANIFEST, 'not the canonical E03 manifest')
    root = private(args.manifest).parent
    rows = [r for r in json.loads(raw)['rows'] if r['split'] == 'validation']
    require(Counter((r['participant'], r['label']) for r in rows) == {('05', 0): 10, ('05', 1): 10}, 'unexpected split')
    run = private(args.run)
    receipt_raw = read(run / 'receipt.json', 1024 * 1024)
    receipt = json.loads(receipt_raw)
    weights = sha(read(run / 'checkpoint.safetensors', 8 * 1024 * 1024))
    require(receipt['exercise'] == 'E03' and receipt['status'] == 'trained' and receipt['manifest_sha256'] == MANIFEST
            and receipt['weights_sha256'] == weights and receipt['source_sha256'] == sha(Path(extra_e03.__file__).read_bytes()),
            'fit provenance mismatch')
    predict = extra_e03_service.open_engine(run)
    version = 'rgb-motion-cnn-v1+' + weights[:12]
    require(receipt['version'] == version, 'wrong fit version')
    def score(sample):
        runtime.probe_frames(sample, 'video/mp4', str(engine / 'runtime/ffprobe.exe'))
        answer = predict(sample)
        require(answer['version'] == version, 'wrong runtime version')
        return answer
    cm = [[0, 0, 0], [0, 0, 0]]
    scratch = private(args.scratch)
    require(scratch.is_dir() and not scratch.is_relative_to(run) and not scratch.is_relative_to(engine)
            and not scratch.is_relative_to(root), 'scratch must be separate')
    with TemporaryDirectory(dir=scratch) as tmp:
        sample = Path(tmp) / 'input.mp4'
        for row in rows:
            path = root / row['path']
            require(path.parent == root / 'media', 'unexpected clip location')
            payload = read(path, 64 * 1024 * 1024)
            require(sha(payload) == row['sha256'], 'unexpected validation clip bytes')
            sample.write_bytes(payload)
            answer = score(sample)
            label = answer['label']
            require(label in (*CLASSES, 'uncertain'), 'unexpected classifier outcome')
            if label in CLASSES:
                confidence = answer['confidence']
                require(answer['reason'] is None and confidence is not None and math.isfinite(confidence) and 0.70 <= confidence <= 1,
                        'invalid scored outcome')
            cm[row['label']][(*CLASSES, 'uncertain').index(label)] += 1
        payload = read(args.blank, 64 * 1024 * 1024)
        require(sha(payload) == args.blank_sha256, 'unexpected blank bytes')
        sample.write_bytes(payload)
        answer = score(sample)
        blank_ok = answer['label'] == 'uncertain' and answer['reason'] == 'low_visibility' and answer['confidence'] is None
    return {'schema': 'qolda-e03-portable-validation-v1', 'engine_inventory_sha256': sha((engine/'manifest.json').read_bytes()),
            'model_version': version, 'weights_sha256': weights, 'training_receipt_sha256': sha(receipt_raw),
            'manifest_sha256': MANIFEST, 'class_order': list(CLASSES), 'columns': [*CLASSES, 'uncertain'], 'confusion': cm,
            'blank_uncertain_low_visibility': blank_ok, 'temporary_directory_removed': not Path(tmp).exists(),
            'subjects': 1, 'independent_evaluation': False, 'approval_written': False, 'ui_exercised': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'run', 'blank', 'scratch'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--blank-sha256', required=True)
    result = receive(parser.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    sys.exit(0 if result['blank_uncertain_low_visibility'] and result['temporary_directory_removed'] else 2)
