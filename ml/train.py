"""Fit on participants 01–04; choose epochs/thresholds on 05; score 08 once."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np
from safetensors import __version__ as safetensors_version
import torch
from torch.nn import functional as F

from ml.api import CONFIDENCE_BAND, Predictor, load_artifact, save_artifact, tensor_batch, threshold_confidence
from ml.data import (COUNTS, FOREARM_NTU, FOREARM_POLICY, POSE_POLICIES, RTMPOSE_SHA, SCHEMA, STRICT_POLICY,
                     dump_json, pose_sequence, read_manifest, sha256)
from ml.extract import ENGINES, cache_dir, run_extraction
from ml.models import ModelA, ModelB, SLIM_CODE, SLIM_REVISION, SLIM_SHA, base_encoder, trainable
from ml.scoring import metrics, open_test_once, select_threshold

SEED = 601
VERSION = 'seriq-e02-1.0.0'
FOREARM_VERSION = 'seriq-e02-forearm-1.0.0'
FOREARM_ADMISSION = dict(class_scored_min=8, balanced_accuracy_min=.70,
                         unconditional_balanced_recall_min=.65, per_class_recall_min=.60)
HYPER = {'A': dict(epochs=36, batch=8, lr=.002), 'B': dict(epochs=20, batch=8, lr=.00015)}
SOURCES = ('ml/api.py','ml/data.py','ml/extract.py','ml/models.py','ml/scoring.py','ml/train.py')


def subset(rows, split):
    return [r for r in rows if r['split'] == split]


def cached(cache, rows, policy=STRICT_POLICY):
    out = []
    for row in rows:
        with np.load(cache / (row['sha256']+'.npz'), allow_pickle=False) as data:
            xyz, mask = data['xyz'], data['mask']
            p = pose_sequence(xyz, mask, policy)
            reason = 'decode_failure' if not len(xyz) else p.reason
        out.append((row['label'], p, reason))
    return out


def filtered(items):
    return [p for y,p,r in items if r is None], [y for y,p,r in items if r is None]


def scores(items, probs, threshold):
    seq = iter(probs)
    report = metrics([y for y,p,r in items], [next(seq) if r is None else None for y,p,r in items],
                     [r for y,p,r in items], threshold)
    if next(seq, None) is not None:
        raise ValueError('received surplus probabilities')
    return report


def mask_control(train, val):
    """A train-fit, class-weighted ridge baseline using only missingness."""
    def x_of(poses):
        x = np.stack([p.mask[:, FOREARM_NTU[0]] for p in poses])
        # Four phase-window counts, plus the number of changes in validity.
        return np.column_stack((np.ones(len(poses)), x.reshape(-1,4,16).mean(axis=2),
                                (x[:, 1:] != x[:, :-1]).mean(axis=1)))
    p_train,y_train = filtered(train)
    p_val,y_val = filtered(val)
    x, y = x_of(p_train), np.asarray(y_train)
    weights = len(y) / (2 * np.bincount(y)[y])
    penalty = np.diag([0, .01, .01, .01, .01, .01])
    beta = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty, x.T @ (weights * y))
    v = (x_of(p_val) @ beta).clip(0, 1).tolist()
    t = select_threshold(y_val, v)
    return dict(threshold=t, variant='class-weighted ridge on four coverage fractions and transitions',
                train=scores(train, (x @ beta).clip(0,1).tolist(), t), validation=scores(val,v,t))


def forearm_admission(report):
    baseline = report['mask_only']['validation']['unconditional_balanced_recall']
    result = {}
    for k in ('A', 'B'):
        m = report['models'][k]['validation']
        rules = FOREARM_ADMISSION
        checks = {
            'class_coverage': all(n >= rules['class_scored_min'] for n in m['class_scored']),
            'conditional_balanced_accuracy': m['balanced_accuracy'] >= rules['balanced_accuracy_min'],
            'unconditional_balanced_recall': m['unconditional_balanced_recall'] >= rules['unconditional_balanced_recall_min'],
            'per_class_recall': all(m['confusion'][c][c]/m['class_total'][c] >= rules['per_class_recall_min'] for c in (0,1)),
            'mask_control': m['unconditional_balanced_recall'] > baseline,
        }
        result[k] = [name for name, passed in checks.items() if not passed]
    return result


def verify_forearm_dev(rows, root, cache):
    """Reuse v1 dev cache only when there could be no repeated native frames."""
    expected = {r['sha256'] for r in rows}
    if {p.stem for p in cache.glob('*.npz')} != expected:
        raise ValueError('forearm fit expects exactly the development cache and no other poses')
    h = hashlib.sha256()
    for row in rows:
        path = root / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('development source hash mismatch')
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 64:
                raise ValueError('v1 cache not admissible: development clip has fewer than 64 native frames')
        finally:
            cap.release()
        h.update(bytes.fromhex(sha256(cache / (row['sha256'] + '.npz'))))
    return dict(n_verified=len(rows), all_native_frames_at_least_64=True, cache_sha256=h.hexdigest())


def track_audit(cache, train_rows):
    """Train-only diagnostics before fitting, aggregated without identifiers."""
    observed, relative_overlap, scales, jitter, ranges = [], [], [], [], []
    for r in train_rows:
        with np.load(cache / (r['sha256'] + '.npz'), allow_pickle=False) as f:
            xy, mask = f['xyz'][..., :2], f['mask']
        mask = mask & np.isfinite(xy).all(axis=-1)
        right, left = xy[:, [14,16]], xy[:, [13,15]]
        m = mask[:, [14,16]].all(axis=1)
        length = np.linalg.norm(right[:,1]-right[:,0], axis=-1)
        m &= length > 1e-5
        co = m & mask[:, [13,15]].all(axis=1)
        relative_overlap.extend((np.linalg.norm(right[co]-left[co], axis=-1).max(axis=1)/length[co]).tolist())
        observed.append(int(m.sum()))
        if m.any():
            scale = float(np.median(length[m]))
            scales.append(scale)
            ranges.append(float(np.linalg.norm(np.ptp(right[m], axis=0),axis=-1).max()/scale))
            adjacent = m[1:] & m[:-1]
            if adjacent.any():
                step = np.linalg.norm(np.diff(right,axis=0),axis=-1).max(axis=1)[adjacent]
                jitter.append(float(np.quantile(step/scale,.90)))
    def summary(values):
        return dict(zip(('min','p10','median','p90','max'), np.quantile(values,[0,.1,.5,.9,1]).tolist())) if values else {}
    return dict(n=len(train_rows), right_pair_ge_40=sum(n>=40 for n in observed),
                right_pair_observed_frames=summary(observed), right_projected_median_length=summary(scales),
                endpoint_span_relative_to_length=summary(ranges), adjacent_step_p90_per_clip=summary(jitter),
                co_observed_left_right_slots=len(relative_overlap),
                separated_above_point45_slots=sum(d>.45 for d in relative_overlap),
                left_right_separation_relative_to_right_length=summary(relative_overlap))


def confidence_audit(items, probs, threshold):
    labels = [y for y,_,r in items if r is None]
    accepted = [(y,p) for y,p in zip(labels,probs,strict=True)
                if threshold_confidence(p,threshold) >= CONFIDENCE_BAND]
    return dict(cutoff=CONFIDENCE_BAND, n_at_or_above=len(accepted), n_below=len(labels)-len(accepted),
                false_complete_at_or_above=sum(y == 0 and p >= threshold for y,p in accepted))


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def current_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but unavailable; check installed PyTorch wheel and NVIDIA driver')
    return torch.device(name)


@torch.no_grad()
def make_prefix(model, data):
    model.encoder.eval()
    return torch.cat([model.encoder.prefix(data[0][i:i+8], data[1][i:i+8])
                      for i in range(0,len(data[0]),8)])


def fit_one(kind, train, val, base, pose, device, target, context):
    started = time.monotonic()
    seed = SEED if kind == 'A' else SEED + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    p_train, y_train = filtered(train)
    p_val, y_val = filtered(val)
    if set(y_train) != {0,1} or set(y_val) != {0,1}:
        raise ValueError(f'both classes must be scorable on train and validation; '
                         f'train={Counter(y_train)} validation={Counter(y_val)}')
    data_train, data_val = tensor_batch(p_train, device), tensor_batch(p_val, device)
    policy = pose.get('policy', STRICT_POLICY)
    model = (ModelA(policy) if kind == 'A' else ModelB(base_encoder(base))).to(device)
    params = trainable(model, kind)
    initial = {name: tensor_digest(value) for name, value in model.named_parameters()}
    frozen = {name: initial[name] for name in initial if name not in params}
    hp = HYPER[kind]
    groups = [{'params': list(params.values()), 'lr': hp['lr']}]
    if kind == 'B':
        groups = [{'params':[p for k,p in params.items() if k.startswith('encoder.')], 'lr':hp['lr']},
                  {'params':[p for k,p in params.items() if k.startswith('head.')], 'lr':hp['lr']*3}]
        # No stochastic layers or trainable tensors in the frozen prefix.
        cached_train, cached_val = make_prefix(model, data_train), make_prefix(model, data_val)
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    y_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)
    weight = torch.tensor([len(y_train)/(2*y_train.count(label)) for label in (0,1)],
                          dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    best, steps = None, 0
    for epoch in range(1, hp['epochs']+1):
        model.train()
        for idx in torch.randperm(len(y_train), generator=generator).split(hp['batch']):
            idx = idx.to(device)
            opt.zero_grad(set_to_none=True)
            logit = (model(*(a[idx] for a in data_train)) if kind == 'A'
                     else model.from_prefix(cached_train[idx]))
            y = y_gpu[idx]
            loss = F.binary_cross_entropy_with_logits(logit, y, weight=weight[y.long()])
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 5.)
            opt.step()
            steps += 1
        model.eval()
        with torch.inference_mode():
            logits = model(*data_val) if kind == 'A' else model.from_prefix(cached_val)
            probs = logits.float().sigmoid().cpu().tolist()
            loss_val = float(F.binary_cross_entropy_with_logits(logits.float(),
                                      torch.tensor(y_val, dtype=torch.float32, device=device)))
        t = select_threshold(y_val, probs)
        perf = scores(val, probs, t)
        rank = (perf['balanced_accuracy'], perf['f1_macro'], -loss_val, -epoch)
        if best is None or rank > best['rank']:
            best = dict(rank=rank, epoch=epoch, steps=steps, t=t, perf=perf, loss=loss_val,
                        state={n:v.detach().cpu().clone() for n,v in params.items()})
        if epoch in (1, hp['epochs']) or epoch % 5 == 0:
            print(f'{kind} epoch {epoch}/{hp["epochs"]}: steps={steps} val_bacc={perf["balanced_accuracy"]:.4f}', flush=True)
    assert best is not None
    model.load_state_dict({**model.state_dict(), **best['state']}, strict=True)
    after = {n:tensor_digest(p) for n,p in model.named_parameters()}
    if not all(after[n] == v for n,v in frozen.items()):
        raise RuntimeError('frozen tensors changed')
    updated = [n for n in params if after[n] != initial[n]]
    witness = 'encoder.blocks.7.attn.qkv.weight' if kind == 'B' else 'conv.0.weight'
    if witness not in updated or not any(n.startswith('head.') for n in updated):
        raise RuntimeError('selected checkpoint did not update encoder/backbone and head')
    record = dict(kind=kind, threshold=best['t'], seed=seed, version=context.get('version', VERSION), pose_policy=policy,
                  epoch=best['epoch'], steps=best['steps'], pose_engine=pose['engine'],
                  pose_model_sha256=pose['sha256'], manifest_sha256=context['manifest_sha256'])
    if kind == 'B':
        record.update(slim_base_sha256=SLIM_SHA, slim_revision=SLIM_REVISION, slim_code=SLIM_CODE)
    before = Predictor(model, record, device).predict_many(p_val)
    save_artifact(target, best['state'], record)
    digest = sha256(target)
    after_reload = load_artifact(target, base, device, digest).predict_many(p_val)
    np.testing.assert_allclose(before, after_reload, atol=1e-7, rtol=1e-6)
    if kind == 'B':
        with torch.inference_mode():
            cached_probs = model.from_prefix(cached_val).sigmoid().cpu().tolist()
        np.testing.assert_allclose(before, cached_probs, atol=1e-6, rtol=1e-5)
    if [p >= best['t'] for p in before] != [p >= best['t'] for p in after_reload]:
        raise RuntimeError('reloaded validation predictions differ')
    train_probs = Predictor(model, record, device).predict_many(p_train)
    train_report = scores(train, train_probs, best['t'])
    val_report = scores(val, after_reload, best['t'])
    if policy == FOREARM_POLICY:
        train_report['confidence_audit'] = confidence_audit(train, train_probs, best['t'])
        val_report['confidence_audit'] = confidence_audit(val, after_reload, best['t'])
    if val_report['confusion'] != best['perf']['confusion']:
        raise RuntimeError('selected validation confusion changed on reload')
    return dict(**record, artifact=target.name, sha256=digest, bytes=target.stat().st_size,
                train=train_report, validation=val_report,
                reload=dict(n_compared=len(before), max_abs_probability_diff=max(abs(a-b) for a,b in zip(before,after_reload)),
                            labels_match=True),
                training=dict(hyperparameters=hp, steps_executed=steps, objective='class-balanced BCE, AdamW',
                              trainable_tensors=len(params), updated_tensors=updated, frozen_tensors_verified=len(frozen),
                              parameters_total=sum(p.numel() for p in model.parameters()),
                              parameters_trainable=sum(p.numel() for p in params.values()),
                              initial_tensor_sha256=initial[witness], selected_tensor_sha256=after[witness], tensor=witness,
                              seconds=round(time.monotonic()-started,3),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None))


def report_md(report, path):
    lines = ['# SerIQ E02 — оценка по участникам', '',
             'Прототип распознавания одного локтевого цикла по позе; не медицинская валидация.', '',
             f"Извлечение: `{report['pose']['engine']}`; схема `{SCHEMA}`; политика `{report['pose']['policy']}`. "
             'По одному участнику на validation и test; результаты не показывают переносимость на популяцию.', '',
             '| Модель | Split | Scorable / n | Balanced acc. | F1 macro | F1 complete | Confusion |',
             '|---|---|---:|---:|---:|---:|---|']
    def fmt(x):
        return '—' if x is None else f'{x:.4f}'
    for model, r in report['models'].items():
        for s in ('train','validation','test'):
            if s in r:
                m = r[s]
                lines.append(f"| {model} | {s} | {m['n_scored']} / {m['n']} | {fmt(m['balanced_accuracy'])} | "
                             f"{fmt(m['f1_macro'])} | {fmt(m['f1_complete'])} | `{m['confusion']}` |")
    if 'mask_only' in report:
        c = report['mask_only']['validation']
        lines.append(f"| Mask-only | validation | {c['n_scored']} / {c['n']} | {fmt(c['balanced_accuracy'])} | "
                     f"{fmt(c['f1_macro'])} | {fmt(c['f1_complete'])} | `{c['confusion']}` |")
    lines.extend(['', 'Порядок строк и столбцов: `[incomplete, complete]`. Метрики таблицы условны на покрытии; '
                  'в JSON также есть полноразмерный denominator, reasons и balanced recall с пропусками как ошибками.',
                  'При `no_scorable` balanced accuracy и F1 не определены.', '',
                  'A: небольшой TCN с нуля. B: полный SLiM NTU60 XSub, обучены последний блок и бинарная голова. '
                  'Сравнение на общей позе без RGB embeddings и метаданных. Для 2D-экстрактора z нулевой, '
                  'отсутствующие суставы маскируются, а не выдаются за наблюдаемые. Видимость не гарантирует правильность позы.',
                  '', f"Статус fit: `{report.get('fit_status')}`; sealed test: `{report.get('test_status','not_opened')}`.", '',
                  'Контрольные суммы и доказательство обновления весов приведены в JSON. Источники и лицензии: '
                  '`artifacts/seriq/ATTRIBUTION.md`.', ''])
    path.write_text('\n'.join(lines), encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['fit','test'])
    parser.add_argument('--private', required=True, type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--pose-model', required=True, type=Path)
    parser.add_argument('--engine', required=True, choices=ENGINES)
    parser.add_argument('--pose-policy', choices=POSE_POLICIES, default=STRICT_POLICY)
    parser.add_argument('--slim-base', required=True, type=Path)
    parser.add_argument('--manifest', default='ml/manifest.json', type=Path)
    parser.add_argument('--artifacts', default='artifacts/seriq', type=Path)
    parser.add_argument('--reports', default='reports/ml', type=Path)
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    args = parser.parse_args()
    if args.private.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error('private data must be outside checkout')
    is_forearm = args.pose_policy == FOREARM_POLICY
    if is_forearm and args.artifacts.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error('forearm artifacts must stay outside checkout')
    if is_forearm and args.root is None:
        parser.error('forearm requires --root to verify development provenance or evaluate test')
    device = current_device(args.device)
    manifest = read_manifest(args.manifest)
    manifest_sha = sha256(args.manifest)
    pose = dict(engine=args.engine, sha256=sha256(args.pose_model), policy=args.pose_policy)
    if is_forearm and (pose['engine'], pose['sha256']) != ('rtmpose', RTMPOSE_SHA):
        parser.error('forearm policy requires pinned RTMPose')
    if sha256(args.slim_base) != SLIM_SHA:
        parser.error('official NTU60 XSub SLiM safetensors SHA-256 mismatch')
    cache = cache_dir(args.private, args.engine, args.pose_model)
    outfile = args.reports/'experiment.json'
    freeze_path = args.reports/'freeze.json'
    lock = args.private/f'sealed-test-seriq-e02-{manifest_sha}.json'
    legacy_lock = args.private/f'sealed-test-{VERSION}-{manifest_sha}.json'
    context = dict(version=FOREARM_VERSION if is_forearm else VERSION, manifest_sha256=manifest_sha,
                   pose=pose, split_counts=COUNTS,
                   sources={p:sha256(p) for p in SOURCES},
                   runtime=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                device=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
                                numpy=np.__version__, opencv=cv2.__version__, safetensors=safetensors_version))
    names = {'A':args.artifacts/'A-tcn.safetensors','B':args.artifacts/'B-slim-last-block.safetensors'}
    if args.stage == 'fit':
        if freeze_path.exists() or lock.exists() or legacy_lock.exists():
            raise FileExistsError('frozen experiment cannot be re-fit in place')
        if is_forearm:
            if outfile.exists() or any(path.exists() for path in names.values()):
                raise FileExistsError('choose new forearm outputs; never overwrite a development receipt or artifact')
            context['development_provenance'] = verify_forearm_dev(
                [r for r in manifest if r['split'] in ('train','validation')], args.root, cache)
        train, val = [cached(cache, subset(manifest,s), args.pose_policy) for s in ('train','validation')]
        scorable = {split: dict(Counter(y for y,_,reason in items if reason is None))
                    for split, items in (('train',train),('validation',val))}
        report = dict(**context, models={}, fit_scorable_by_class=scorable, test_status='not_opened')
        if is_forearm:
            report['train_track_audit'] = track_audit(cache,subset(manifest,'train'))
        report['fit_denominators'] = {'train':len(train), 'validation':len(val)}
        report['fit_abstain_reasons'] = {
            split:dict(Counter(reason for _,_,reason in items if reason is not None))
            for split, items in (('train',train),('validation',val))}
        if any(set(counts) != {0,1} for counts in scorable.values()):
            report['fit_status'] = 'blocked_pose_coverage'
            dump_json(outfile,report)
            print(f'blocked_pose_coverage: {scorable}', flush=True)
            raise SystemExit(2)
        if is_forearm:
            report['mask_only'] = mask_control(train,val)
        for model, path in names.items():
            report['models'][model] = fit_one(model,train,val,args.slim_base,pose,device,path,context)
            report['fit_status'] = 'trained_both' if len(report['models']) == 2 else 'trained_A_only'
            dump_json(outfile,report)
            report_md(report,args.reports/'experiment.md')
        if is_forearm:
            failures = forearm_admission(report)
            report['sealed_admission'] = dict(rules=FOREARM_ADMISSION, failures=failures, passed=not any(failures.values()))
            if not report['sealed_admission']['passed']:
                report['fit_status'] = 'trained_both_blocked_validation'
                report['test_status'] = 'not_opened_blocked_validation'
            dump_json(outfile,report)
            report_md(report,args.reports/'experiment.md')
            if not report['sealed_admission']['passed']:
                print(f'sealed test remains closed: {failures}', flush=True)
                raise SystemExit(2)
        dump_json(freeze_path, dict(**context, sealed_admission=report.get('sealed_admission'),
                                    artifacts={k:dict(sha256=r['sha256'], threshold=r['threshold'])
                                                         for k,r in report['models'].items()}))
        print(f'frozen A and B: {freeze_path}', flush=True)
    else:
        if args.root is None:
            parser.error('test needs --root')
        if lock.exists() or legacy_lock.exists():
            raise FileExistsError('sealed test has already been opened')
        if not freeze_path.is_file():
            parser.error('sealed test requires a successful pre-test freeze')
        freeze = json.loads(freeze_path.read_text(encoding='utf8'))
        if any(freeze[k] != context[k] for k in ('version','manifest_sha256','pose','sources')):
            raise ValueError('sealed test requires exact frozen manifest, extractor and source code')
        report = json.loads(outfile.read_text(encoding='utf8'))
        if is_forearm and (not freeze.get('sealed_admission', {}).get('passed') or any(forearm_admission(report).values())):
            raise ValueError('sealed test requires passing forearm validation for both models')
        loaded = {k:load_artifact(path,args.slim_base,device,freeze['artifacts'][k]['sha256']) for k,path in names.items()}
        if any((m.metadata['threshold'], m.metadata['pose_policy'], m.metadata['kind']) !=
               (freeze['artifacts'][k]['threshold'], args.pose_policy, k) for k,m in loaded.items()):
            raise ValueError('artifact metadata differs from freeze')
        open_test_once(lock, freeze)
        test = subset(manifest,'test')
        if is_forearm:
            cache = args.private / f'pose-sealed-{FOREARM_POLICY}-{RTMPOSE_SHA}'
        report['test_pose'] = run_extraction(test,args.root,args.engine,args.pose_model,cache,policy=args.pose_policy)
        frozen_test = cached(cache,test,args.pose_policy)
        scorable,_ = filtered(frozen_test)
        for k,predictor in loaded.items():
            probs = predictor.predict_many(scorable)
            t = predictor.metadata['threshold']
            m = scores(frozen_test,probs,t)
            if is_forearm:
                m['confidence_audit'] = confidence_audit(frozen_test,probs,t)
            report['models'][k]['test'] = m
        report['test_status'] = 'scored_once' if scorable else 'no_scorable'
        dump_json(outfile, report)
        report_md(report,args.reports/'experiment.md')
        print('sealed test:', {k:m['test'] for k,m in report['models'].items()}, flush=True)


if __name__ == '__main__':
    main()
