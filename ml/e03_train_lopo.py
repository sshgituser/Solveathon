"""Four leave-one-person-out folds on the E03 development TRAIN partition only."""
import argparse
import contextlib
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from ml import extra_e03 as e

PIN = '125abc289062c15a26a490bbaf7306d73767fcaddbea6c62fff81282a0d2b7ea'
SOURCE = 'ed3cac5dd48aa0afe9b8a13d48afc5c617b69b2b010216ef8293c8dac3df2a43'
METHODS = ('cnn_selective', 'dynamic_selective', 'cnn_raw', 'dynamic_raw', 'static_probe', 'cnn_shuffled_time')


def combine(folds, method):
    """Raw numerators/denominators, not an unweighted average of clip-level folds."""
    entries = [f['methods'][method] for f in folds]
    totals = [sum(m['class_total'][i] for m in entries) for i in range(2)]
    cm = [[sum(m['confusion'][i][j] for m in entries) for j in range(2)] for i in range(2)]
    decided = sum(map(sum, cm))
    return {'n':sum(totals),'scored':decided,'coverage':decided / sum(totals),'confusion':cm,
            'reject_by_class':[totals[i]-sum(cm[i]) for i in range(2)],
            'ba_with_rejections':mean(cm[i][i]/totals[i] for i in range(2)),
            'mean_person_ba_with_rejections':mean(m['unconditional_balanced_recall'] for m in entries)}


def evaluate(manifest):
    manifest = manifest.resolve(strict=True)
    if e._file_sha(Path(e.__file__)) != SOURCE:
        raise ValueError('Unrecognized E03 recipe')
    if any((p/'.git').exists() for p in manifest.parents):
        raise ValueError('Private manifest must be outside Git')
    rows, _ = e._load_manifest(manifest, PIN)
    train = [row for row in rows if row.split == 'train']
    if len(train) != 81 or {r.participant for r in train} != e.TRAIN_IDS:
        raise ValueError('Unexpected original training population')
    e._check_local_files(manifest.parent, train)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import cv2
    import numpy as np
    import torch
    device = e._cuda_only(torch)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(1)
    clips = [e._load_one(r, cv2, np) for r in train]
    if any(x is None for x in clips):
        raise ValueError('Incomplete decode; do not silently drop a training clip')
    people_by_sample = {}
    for r, x in zip(train, clips, strict=True):
        if people_by_sample.setdefault(e._sha(x.tobytes()), r.participant) != r.participant:
            raise ValueError('Identical sampled inputs across participants')
    folds = []
    for idx, subject in enumerate(sorted(e.TRAIN_IDS), 1):
        fold = [replace(r, split='validation') if r.participant == subject else r for r in train]
        torch.manual_seed(e.SEED); torch.cuda.manual_seed_all(e.SEED)
        receipt = {'source_sha256':SOURCE}
        with TemporaryDirectory(dir=manifest.parent, prefix='e03-lopo-') as tmp:
            # Unmodified recipe reloads its own frozen checkpoint and re-decodes the excluded person's files.
            with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                e._fit_and_validate(fold, clips, Path(tmp), receipt, torch, np, device)
        evidence = receipt['reload']; audit = evidence['cuda_audit']
        if (not evidence['ok'] or audit['non_cuda_tensor_operands_or_outputs'] != 0 or
                set(receipt['training']['participants_exposed']) != e.TRAIN_IDS - {subject}):
            raise RuntimeError('Invalid fold exposure, placement or replay')
        folds.append({'fold':idx,'fit_people':3,'held_n':evidence['n_full_validation_videos'],
                      'steps':receipt['training']['steps'],'reload_mismatches':evidence['decision_mismatches'],
                      'cuda_ops':audit['graph_operator_calls'],'methods':{k:receipt['validation'][k] for k in METHODS}})
    if sum(f['held_n'] for f in folds) != len(train):
        raise RuntimeError('Evaluation did not cover the original train split exactly once')
    return {'schema':'qolda-e03-development-lopo-v1','source_sha256':SOURCE,'manifest_sha256':PIN,
            'seed':e.SEED,'epochs':e.EPOCHS,'confidence_floor':e.MARGIN,'gpu':torch.cuda.get_device_name(0),
            'original_validation_media_read':False,'serving_assets_changed':False,'folds':folds,
            'pooled':{k:combine(folds,k) for k in METHODS}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    print(json.dumps(evaluate(ap.parse_args().manifest), indent=2, allow_nan=False))
