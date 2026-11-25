"""Development-only strict-GPU training entrypoint for SerIQ E02.

Owner decision: training runs on CUDA only. There is no ``--device cpu`` and no
path that silently falls back: the pose extractor comes from ``ml/gpu.py``
(CUDA EP only, CPU fallback disabled, full-delegation proven from a profile),
and the classifier is asserted to live on ``cuda:0`` before and after fitting.

Variants are experiment identifiers, not safetensors ``kind`` values: A is a
standalone TCN (``kind='A'``), while B and C are the same SLiM fine-tune over the
same pinned base with different corpora (``kind='B'`` plus ``variant`` and
``corpus_scope`` in the record). The historical test split is never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 601
A_RELEASE_MANIFEST = '8c73b2a35584639f3a2ea057878d39cf105c52ffa4d77a4a41639c2606134931'
VERSION = 'seriq-e02-forearm-1.0.0'
HYPER = {'A': dict(epochs=36, batch=8, lr=.002), 'B': dict(epochs=20, batch=8, lr=.00015),
         'C': dict(epochs=20, batch=8, lr=.00015)}
SPLITS = {'train', 'validation', 'quarantine'}
CACHE_TAG = 'ort127-cuda-b1'
SCOPES = {'B': 'qolda-owned-e02', 'C': 'qolda-owned-e02+approved-extension'}
ROW_V2 = {'path', 'split', 'label', 'sha256', 'source_id', 'family', 'origin'}
HEX64 = __import__('re').compile('[a-f0-9]{64}')
TOKEN = __import__('re').compile(r'[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,95}')
PROTECTED = __import__('re').compile(r'(?:^|[._ /\\-])(?:08|(?:tests?|holdouts?|quarantined?)[0-9]*)(?:$|[._ /\\-])',
                                     __import__('re').IGNORECASE)


def sha256(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n',
                   encoding='utf8')
    tmp.replace(path)


def load_manifest(path, variant=None, expected_sha256=None):
    """Validate a manifest before any GPU, media or output work.

    A keeps the historical row schema. B/C require the v2 derived manifest from
    ``tools/model_intake.py``: the pinned digest of the exact bytes, schema, variant,
    corpus scope, owner pins, allowed splits, exact row schema and the intake path
    validator are checked here, so a renamed or edited public E02 cannot be trained.
    """
    raw = Path(path).read_bytes()
    if variant == 'A' and expected_sha256 is None:
        expected_sha256 = A_RELEASE_MANIFEST
    actual = hashlib.sha256(raw).hexdigest()
    if variant in ('A', 'B', 'C') and (not expected_sha256 or not HEX64.fullmatch(expected_sha256) or actual != expected_sha256):
        raise ValueError('manifest bytes do not match the required release pin')
    doc = json.loads(raw.decode('utf8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('rows'), list) or not doc['rows']:
        raise ValueError('empty or malformed manifest')
    rows = doc['rows']
    if variant in ('B', 'C'):
        if doc.get('schema') != 'qolda-e02-train-gpu-manifest-v2':
            raise ValueError('B/C require the v2 derived manifest from tools/model_intake.py')
        if doc.get('variant') != variant:
            raise ValueError(f'manifest variant does not match --variant {variant}')
        if doc.get('corpus_scope') != SCOPES[variant]:
            raise ValueError('manifest corpus_scope does not match the variant')
        pins = ['owned_metadata_sha256'] + (['expanded_metadata_sha256'] if variant == 'C' else [])
        for key in pins:
            if not HEX64.fullmatch(str(doc.get(key, ''))):
                raise ValueError(f'{key} must be a pinned 64-hex digest')
        origins = {'owned'} if variant == 'B' else {'owned', 'expanded'}
        for row in rows:
            if not isinstance(row, dict) or set(row) != ROW_V2:
                raise ValueError('B/C row schema must be exactly path/split/label/sha256/source_id/family/origin')
            if row['split'] not in ('train', 'validation'):
                raise ValueError(f'manifest admits a forbidden split: {row["split"]}')
            if row['origin'] not in origins:
                raise ValueError(f'origin {row["origin"]} is not admitted for {variant}')
            if not HEX64.fullmatch(str(row['sha256'])) or row['sha256'] == '0' * 64:
                raise ValueError('invalid clip digest')
            if not TOKEN.fullmatch(str(row['source_id'])) or not TOKEN.fullmatch(str(row['family'])):
                raise ValueError('invalid source_id or family token')
            if type(row['label']) is not int or row['label'] not in (0, 1):
                raise ValueError('invalid label')
            from tools.model_intake import valid_clip_path
            if not valid_clip_path(row['path']):
                raise ValueError('manifest path is not the canonical intake clip path')
        if variant == 'B' and any(row['origin'] != 'owned' for row in rows):
            raise ValueError('B must not contain extension rows')
        if variant == 'C':
            if not any(r['origin'] == 'expanded' and r['split'] == 'train' for r in rows):
                raise ValueError('C requires at least one admitted expanded train row')
            if any(r['origin'] == 'expanded' and r['split'] != 'train' for r in rows):
                raise ValueError('expanded rows are train only; validation stays shared')
    else:
        bad = {r.get('split') for r in rows} - SPLITS
        if bad:
            raise ValueError(f'manifest admits a forbidden split: {sorted(bad)}')
        required = {'path', 'label', 'split', 'sha256'}
        for row in rows:
            if not required <= set(row):
                raise ValueError('manifest row is incomplete')
    for split in ('train', 'validation'):
        labels = Counter(r['label'] for r in rows if r['split'] == split)
        if set(labels) != {0, 1}:
            raise ValueError(f'{split} must contain both classes: {dict(labels)}')
    return rows, actual


def validation_rows_digest(rows):
    """Digest of the shared validation rows: clip digest *and* label, sorted.

    Sorting only clip SHA-256 would make a label swap invisible to review.
    """
    return hashlib.sha256(''.join(sorted(f'{digest}:{label}' for digest, label in rows)).encode()).hexdigest()


def prepare_pose(model_path, workdir):
    from ml.gpu import open_cuda_pose, prepare_static_graph
    workdir.mkdir(parents=True, exist_ok=True)
    graph = workdir / 'end2end.b1.onnx'
    receipt = None
    if not graph.exists():
        receipt = prepare_static_graph(model_path, graph)
    pose = open_cuda_pose(graph)
    return pose, dict(graph=graph, graph_sha256=sha256(graph), transform=receipt,
                      delegation=pose.delegation)


def extract(rows, root, pose, cache, *, limit=None):
    from ml.data import FOREARM_POLICY, pose_sequence
    from tools.verify_gpu_engine import clip_frames
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if r['split'] in ('train', 'validation')]
    if limit:
        todo = todo[:limit]
    started, done = time.monotonic(), 0
    for row in todo:
        target = cache / (row['sha256'] + '.npz')
        if target.is_file():
            continue
        path = root / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('clip bytes do not match the admitted digest')
        frame_list = clip_frames(path)
        xyz, masks = [], []
        for frame in frame_list:
            coords, observed = pose(frame)
            xyz.append(coords)
            masks.append(observed)
        tmp = target.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, xyz=np.stack(xyz), mask=np.stack(masks))
        tmp.replace(target)
        done += 1
        if done % 10 == 0:
            rate = done / (time.monotonic() - started)
            print(f'pose: {done}/{len(todo)} ({rate:.2f} clips/s)', flush=True)
    missing = [r['sha256'] for r in todo if not (cache / (r['sha256'] + '.npz')).is_file()]
    if missing:
        raise ValueError(f'pose cache incomplete: {len(missing)} clips')
    return dict(clips=len(todo), cache=str(cache.name), policy=FOREARM_POLICY)


def cached(cache, rows, policy):
    from ml.data import pose_sequence
    out = []
    for row in rows:
        if row['split'] == 'quarantine':
            continue
        with np.load(cache / (row['sha256'] + '.npz'), allow_pickle=False) as data:
            pose = pose_sequence(data['xyz'], data['mask'], policy)
            reason = 'decode_failure' if not len(data['xyz']) else pose.reason
        has_provenance = {'source_id', 'family', 'origin'} <= set(row)
        meta = {key: row[key] for key in ('source_id', 'family', 'origin', 'sha256')} if has_provenance else None
        out.append((row['split'], row['label'], pose, reason, meta) if meta
                   else (row['split'], row['label'], pose, reason))
    return out


def fit_variant(variant, items, base, pose_info, target, context, budget=None):
    """Fit A by epochs, or B/C by one pre-agreed step budget with step-based evaluation.

    ``budget`` is required for B/C: ``steps`` and ``eval_every`` are fixed before the
    pair runs, so B and C cannot be compared on different amounts of compute. Exposure
    is tracked per row family: a checkpoint that has not seen every required family is
    recorded as ``<variant>-partial`` instead of being reported as a full variant.
    """
    import math
    import torch
    from torch.nn import functional as F
    from ml.api import Predictor, load_artifact, save_artifact, tensor_batch
    from ml.data import FOREARM_POLICY
    from ml.gpu import assert_model_on_cuda
    from ml.models import ModelA, ModelB, base_encoder, trainable
    from ml.scoring import metrics, select_threshold
    from ml.train import confidence_audit, filtered

    kind = 'A' if variant == 'A' else 'B'
    if kind == 'B' and budget is None:
        raise ValueError('B/C require an explicit step budget')
    seed = SEED if kind == 'A' else SEED + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats()

    def meta_of(item):
        return item[4] if len(item) > 4 else None

    train = [i for i in items if i[0] == 'train']
    val = [i for i in items if i[0] == 'validation']
    p_train, y_train = filtered([(y, p, r) for _, y, p, r, *_ in train])
    p_val, y_val = filtered([(y, p, r) for _, y, p, r, *_ in val])
    if set(y_train) != {0, 1} or set(y_val) != {0, 1}:
        raise ValueError(f'both classes must be scorable: train={Counter(y_train)} val={Counter(y_val)}')

    data_train, data_val = tensor_batch(p_train, device), tensor_batch(p_val, device)
    policy = pose_info['policy']
    model = (ModelA(policy) if kind == 'A' else ModelB(base_encoder(base))).to(device)
    assert_model_on_cuda(model, device)
    params = trainable(model, kind)
    initial = {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
               for name, value in model.named_parameters()}
    frozen = {name: initial[name] for name in initial if name not in params}
    hp = HYPER[variant]
    groups = [{'params': list(params.values()), 'lr': hp['lr']}]
    if kind == 'B':
        groups = [{'params': [p for k, p in params.items() if k.startswith('encoder.')], 'lr': hp['lr']},
                  {'params': [p for k, p in params.items() if k.startswith('head.')], 'lr': hp['lr'] * 3}]
        with torch.no_grad():
            model.encoder.eval()
            cached_train = torch.cat([model.encoder.prefix(data_train[0][i:i + 8], data_train[1][i:i + 8])
                                      for i in range(0, len(data_train[0]), 8)])
            cached_val = torch.cat([model.encoder.prefix(data_val[0][i:i + 8], data_val[1][i:i + 8])
                                    for i in range(0, len(data_val[0]), 8)])
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    y_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)
    weight = torch.tensor([len(y_train) / (2 * y_train.count(label)) for label in (0, 1)],
                          dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    best, steps, checks = None, 0, 0

    families = [meta_of(i)['family'] if meta_of(i) else None for i in train if i[3] is None]
    # Every family present in an admitted train row is required, even if all of its rows
    # turn out unusable: otherwise a family could disappear from the denominator.
    admitted = {}
    for item in train:
        meta = meta_of(item)
        if meta:
            admitted.setdefault(meta['family'], set()).add(meta['origin'])
    scorable_families = {meta_of(i)['family'] for i in train if meta_of(i) and i[3] is None}
    required = set(admitted)
    # A family is unusable only when it has no scorable row at all; partial families count.
    unusable = {family for family in admitted if family not in scorable_families}
    expanded_required = {family for family, origins in admitted.items() if 'expanded' in origins}

    def evaluate(step_label):
        nonlocal best, checks
        model.eval()
        with torch.inference_mode():
            logits = model(*data_val) if kind == 'A' else model.from_prefix(cached_val)
            probs = logits.float().sigmoid().cpu().tolist()
            loss_val = float(F.binary_cross_entropy_with_logits(
                logits.float(), torch.tensor(y_val, dtype=torch.float32, device=device)))
        threshold = select_threshold(y_val, probs)
        perf = metrics(y_val, probs, [None] * len(y_val), threshold)
        rank = (perf['balanced_accuracy'], perf['f1_macro'], -loss_val, -step_label)
        checks += 1
        if best is None or rank > best['rank']:
            best = dict(rank=rank, epoch=step_label, steps=steps, threshold=threshold, perf=perf,
                        loss=loss_val, state={n: v.detach().cpu().clone() for n, v in params.items()})

    first_step = {}
    if kind == 'A':
        for epoch in range(1, hp['epochs'] + 1):
            model.train()
            for idx in torch.randperm(len(y_train), generator=generator).split(hp['batch']):
                idx = idx.to(device)
                opt.zero_grad(set_to_none=True)
                logit = model(*(a[idx] for a in data_train))
                loss = F.binary_cross_entropy_with_logits(logit, y_gpu[idx], weight=weight[y_gpu[idx].long()])
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params.values(), 5.)
                opt.step()
                steps += 1
            evaluate(epoch)
            if epoch in (1, hp['epochs']) or epoch % 5 == 0:
                print(f'{variant} epoch {epoch}/{hp["epochs"]}: steps={steps} '
                      f'val_bacc={best["perf"]["balanced_accuracy"]:.4f}', flush=True)
    else:
        total = budget['steps']
        eval_every = budget['eval_every']
        full_pass = math.ceil(len(y_train) / hp['batch'])
        if total < full_pass:
            raise ValueError(f'step budget {total} is below one full pass ({full_pass})')
        order, cursor = torch.randperm(len(y_train), generator=generator), 0
        eval_points = set(range(eval_every, total + 1, eval_every)) | {total}
        for step in range(1, total + 1):
            model.train()
            # Keep the short tail batch instead of dropping it, then reshuffle.
            if cursor >= len(y_train):
                order, cursor = torch.randperm(len(y_train), generator=generator), 0
            take = min(hp['batch'], len(y_train) - cursor)
            idx = order[cursor:cursor + take]
            cursor += take
            for local in idx.tolist():
                family = families[local]
                if family is not None and family not in first_step:
                    first_step[family] = step
            idx = idx.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model.from_prefix(cached_train[idx])
            loss = F.binary_cross_entropy_with_logits(logit, y_gpu[idx], weight=weight[y_gpu[idx].long()])
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 5.)
            opt.step()
            steps = step
            if step in eval_points:
                evaluate(step)
                print(f'{variant} step {step}/{total}: val_bacc={best["perf"]["balanced_accuracy"]:.4f} '
                      f'checks={checks}', flush=True)
    if best is None:
        raise RuntimeError('no checkpoint selected')
    model.load_state_dict({**model.state_dict(), **best['state']}, strict=True)
    assert_model_on_cuda(model, device)
    after = {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
             for n, p in model.named_parameters()}
    if not all(after[n] == v for n, v in frozen.items()):
        raise RuntimeError('frozen tensors changed')
    updated = [n for n in params if after[n] != initial[n]]
    witness = 'encoder.blocks.7.attn.qkv.weight' if kind == 'B' else 'conv.0.weight'
    if witness not in updated or not any(n.startswith('head.') for n in updated):
        raise RuntimeError('selected checkpoint did not update backbone and head')

    exposed = {family for family, step in first_step.items() if step <= best['steps']}
    missing = sorted(required - exposed)
    # A full C needs every required family and at least one update on an added source.
    expanded_exposed = {family for family in expanded_required if family in exposed}
    complete = not missing and (variant != 'C' or bool(expanded_exposed))
    claim = variant if complete else variant + '-partial'
    # Labels are part of the digest: the same clip digests with swapped labels must not look equal.
    has_provenance = any(meta_of(i) for i in val)
    validation_sha = (validation_rows_digest([(meta_of(i)['sha256'], i[1]) for i in val if meta_of(i)])
                      if has_provenance else None)
    per_origin = {}
    for origin in sorted({o for origins in admitted.values() for o in origins}):
        origin_families = {f for f, origins in admitted.items() if origin in origins}
        per_origin[origin] = dict(admitted=len(origin_families),
                                  scorable=len(origin_families & scorable_families),
                                  exposed=len(origin_families & exposed))
    # Source tokens stay in private training receipts, never in model metadata or stdout.
    by_source = {}
    for item in train:
        meta = meta_of(item)
        if meta:
            by_source.setdefault(meta['source_id'], set()).add(meta['family'])
    per_source = {source: dict(admitted=len(group), scorable=len(group & scorable_families),
                               exposed=len(group & exposed)) for source, group in sorted(by_source.items())}

    record = dict(kind=kind, variant=claim, threshold=best['threshold'], seed=seed, version=VERSION,
                  pose_policy=policy, epoch=best['epoch'], steps=best['steps'],
                  budget_steps=budget['steps'] if budget else None,
                  eval_every=budget['eval_every'] if budget else None,
                  exposure_complete=complete,
                  required_families=len(required), exposed_families=len(exposed),
                  unusable_families=len(unusable),
                  expanded_required_families=len(expanded_required),
                  expanded_exposed_families=len(expanded_exposed),
                  exposure_by_origin=per_origin,
                  pose_engine='rtmpose', pose_model_sha256=context['pose_original_sha256'],
                  pose_graph_sha256=context['pose_graph_sha256'],
                  manifest_sha256=context['manifest_sha256'],
                  corpus_scope=context['corpus_scope'],
                  runtime_device='cuda:0')
    if kind == 'B':
        from ml.models import SLIM_CODE, SLIM_REVISION, SLIM_SHA
        record.update(slim_base_sha256=SLIM_SHA, slim_revision=SLIM_REVISION, slim_code=SLIM_CODE)
    before = Predictor(model, record, 'cuda:0').predict_many(p_val)
    save_artifact(target, best['state'], record)
    digest = sha256(target)
    reloaded = load_artifact(target, base, 'cuda:0', digest).predict_many(p_val)
    np.testing.assert_allclose(before, reloaded, atol=1e-7, rtol=1e-6)
    if [p >= best['threshold'] for p in before] != [p >= best['threshold'] for p in reloaded]:
        raise RuntimeError('reloaded validation predictions differ')
    train_probs = Predictor(model, record, 'cuda:0').predict_many(p_train)
    val_probs = reloaded

    def full(items, probs):
        values, queue = [], iter(probs)
        for item in items:
            values.append(next(queue) if item[3] is None else None)
        return values

    train_report = metrics([i[1] for i in train], full(train, train_probs),
                           [i[3] for i in train], best['threshold'])
    val_report = metrics([i[1] for i in val], full(val, val_probs),
                         [i[3] for i in val], best['threshold'])
    train_report['confidence_audit'] = confidence_audit(
        [(y, p, None) for y, p in zip(y_train, p_train)], train_probs, best['threshold'])
    val_report['confidence_audit'] = confidence_audit(
        [(y, p, None) for y, p in zip(y_val, p_val)], val_probs, best['threshold'])
    return dict(record=record, artifact=target.name, sha256=digest, bytes=target.stat().st_size,
                train=train_report, validation=val_report,
                training=dict(hyperparameters=hp, steps_executed=steps,
                              budget_steps=budget['steps'] if budget else None,
                              eval_every=budget['eval_every'] if budget else None,
                              validation_checks=checks, seed=seed,
                              validation_rows_sha256=validation_sha,
                              class_weights=weight.tolist(),
                              checkpoint_step=best['steps'],
                              required_families=len(required), exposed_families_at_checkpoint=len(exposed),
                              missing_families=len(missing), unusable_families=len(unusable),
                              expanded_required_families=len(expanded_required),
                              expanded_exposed_families=len(expanded_exposed),
                              exposure_by_origin=per_origin,
                              exposure_by_source=per_source,
                              variant_claim=claim,
                              trainable_tensors=len(params), updated_tensors=updated,
                              frozen_tensors_verified=len(frozen),
                              parameters_total=sum(p.numel() for p in model.parameters()),
                              parameters_trainable=sum(p.numel() for p in params.values()),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                              device=str(next(model.parameters()).device)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', required=True, choices=('A', 'B', 'C'))
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--manifest-sha256', default=None,
                    help='approved manifest SHA-256: A defaults to the canonical development set; B/C require explicit derived pins')
    ap.add_argument('--root', required=True, type=Path, help='directory the manifest paths are relative to')
    ap.add_argument('--pose-model', required=True, type=Path)
    ap.add_argument('--private', required=True, type=Path, help='private work dir outside the checkout')
    ap.add_argument('--out', required=True, type=Path, help='new run directory outside the checkout')
    ap.add_argument('--slim-base', type=Path)
    ap.add_argument('--corpus-scope', default=None)
    ap.add_argument('--limit', type=int, default=None, help='debug only: A only, never a B/C release run')
    ap.add_argument('--steps', type=int, default=None,
                    help='B/C: one positive update budget agreed before the pair runs')
    ap.add_argument('--eval-every', type=int, default=None,
                    help='B/C: the same step-based evaluation grid for both variants')
    args = ap.parse_args(argv)

    private, out = args.private.resolve(), args.out.resolve()
    if private.is_relative_to(ROOT) or out.is_relative_to(ROOT):
        ap.error('private work dir and run directory must be outside the checkout')
    if out.exists() and any(out.iterdir()):
        ap.error('run directory already exists and is not empty')
    if args.variant != 'A' and not args.slim_base:
        ap.error('variants B and C require --slim-base')
    if args.variant == 'A':
        if args.steps is not None or args.eval_every is not None:
            ap.error('--steps/--eval-every are B/C only; A keeps its fixed recipe')
    else:
        if args.steps is None or args.eval_every is None:
            ap.error('B/C require --steps and --eval-every, fixed before the pair runs')
        if not args.manifest_sha256:
            ap.error('B/C require --manifest-sha256 pinned at release time')
        if args.steps <= 0 or args.eval_every <= 0:
            ap.error('--steps and --eval-every must be positive')
        if args.limit is not None:
            ap.error('--limit is not allowed for B/C')
        if args.corpus_scope not in (None, SCOPES[args.variant]):
            ap.error(f'--corpus-scope must be {SCOPES[args.variant]}')

    # Manifest validation happens before CUDA init, media access or any output directory.
    try:
        rows, manifest_sha = load_manifest(args.manifest, args.variant, args.manifest_sha256)
    except FileNotFoundError:
        ap.error('manifest is missing; no training started')
    except ValueError as exc:
        ap.error(str(exc))
    counts = Counter((r['split'], r['label']) for r in rows)
    print(f'manifest rows={len(rows)} counts={dict(counts)}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    from ml.gpu import cuda_stack, enable_cuda_dlls
    enable_cuda_dlls()
    stack = cuda_stack()


    pose, pose_info = prepare_pose(args.pose_model, private / 'graphs')
    budget = dict(steps=args.steps, eval_every=args.eval_every) if args.variant != 'A' else None
    receipt = dict(schema='seriq-train-gpu-receipt-v1', variant=args.variant,
                   generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   stack=stack, manifest_sha256=manifest_sha,
                   manifest_counts={f'{k[0]}:{k[1]}': v for k, v in sorted(counts.items(), key=str)},
                   budget=budget,
                   pose=dict(original_sha256=sha256(args.pose_model), graph_sha256=pose_info['graph_sha256'],
                             delegation=pose_info['delegation'], transform=pose_info['transform']),
                   corpus_scope=args.corpus_scope or ('mendeley-e02-public-dev' if args.variant == 'A'
                                                      else SCOPES[args.variant]),
                   run_dir=out.name, status='running')
    dump(out / 'receipt.json', receipt)

    cache = private / 'pose-cache' / f'{CACHE_TAG}-{pose_info["graph_sha256"][:12]}'
    receipt['extraction'] = extract(rows, args.root, pose, cache, limit=args.limit)
    receipt['cache_dir'] = cache.name
    dump(out / 'receipt.json', receipt)

    from ml.data import FOREARM_POLICY
    work_rows = [r for r in rows if r['split'] in ('train', 'validation')]
    if args.limit:
        work_rows = work_rows[:args.limit]
    items = cached(cache, work_rows, FOREARM_POLICY)
    target = out / ('A-tcn.safetensors' if args.variant == 'A' else f'{args.variant}-slim-last-block.safetensors')
    context = dict(pose_original_sha256=sha256(args.pose_model), pose_graph_sha256=pose_info['graph_sha256'],
                   manifest_sha256=manifest_sha, corpus_scope=receipt['corpus_scope'])
    result = fit_variant(args.variant, items, args.slim_base, dict(policy=FOREARM_POLICY, engine='rtmpose'),
                         target, context, budget)
    receipt.update(result)
    receipt['status'] = 'trained' if result['record'].get('exposure_complete') else 'trained_partial'
    dump(out / 'receipt.json', receipt)
    dump(out / 'report.json', dict(variant=args.variant, artifact=result['artifact'],
                                   sha256=result['sha256'], train=result['train'],
                                   validation=result['validation'], training=result['training']))
    print(json.dumps(dict(variant=args.variant, artifact=result['artifact'], sha256=result['sha256'],
                          bacc=result['validation']['balanced_accuracy'],
                          selection=result['record']['threshold'],
                          steps=result['training']['steps_executed'],
                          checks=result['training']['validation_checks'],
                          variant_claim=result['training']['variant_claim']), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
