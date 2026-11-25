"""E01 development robustness: leave one original train participant out; never open validation media."""
import argparse
import json
from pathlib import Path
from statistics import mean

from ml import extra_e01 as e

# This diagnostic describes a single reviewed recipe and intake, not a data admission tool.
MANIFEST = '287285d9386f26f7d147a9821fdd46b38f32673f801ea7c7df336ddefe7015eb'
RECIPE = '592963d86c9b54b7246bd716374aa17faffa116d21c27d4bf024b3a466d5a475'


def evaluate(manifest):
    if e._sha(Path(e.__file__)) != RECIPE:
        raise ValueError('E01 source is not the approved frozen recipe')
    rows, pin = e._load_manifest(manifest, MANIFEST)
    root = manifest.parent.resolve(strict=True)
    train = [r for r in rows if r.split == 'train']
    groups = sorted({r.participant for r in train})
    if len(groups) != 4 or len(train) != 82 or set(groups) & {r.participant for r in rows if r.split == 'validation'}:
        raise ValueError('unexpected participant partition')
    graph = e._resource(root / 'run', e.POSE)
    if e._sha(graph) != e.POSE_SHA:
        raise ValueError('pose graph mismatch')
    stack = e.cuda_stack()
    pose = e._Pose(graph)
    try:
        # Only frozen, per-clip pose features are reused. Never persist trajectories.
        data = [(r, e._motion(e._verified_clip(root, r), pose)) for r in train]
        folds, truth, tcn_all, base_all = [], [], [], []
        for group in groups:
            fitting = [(r, m) for r, m in data if r.participant != group]
            unseen = [(r, m) for r, m in data if r.participant == group]
            if {r.label for r, _ in unseen} != {0, 1}:
                raise ValueError('fold missing a class')
            # Normalization, pos_weight, baseline threshold and the entire head are refit only on the other people.
            model, fit = e._fit(fitting)
            threshold = e._baseline(fitting)
            engine = e._Engine(pose, model, e.VERSION)
            y, tcn, baseline = [], [], []
            for r, motion in unseen:
                p = engine._probability(motion) if motion.reason is None else None
                label = e._decision(p, motion.reason, e.VERSION)['label']
                y.append(r.label)
                tcn.append({'incomplete': 0, 'complete': 1}.get(label))
                baseline.append(int(motion.lift_return >= threshold) if motion.lift_return is not None else None)
            audit = engine.evidence['head']
            if not audit or audit['non_cuda_tensor_ops'] != 0:
                raise RuntimeError('fold did not pass the real tensor-device audit')
            folds.append({'fold': len(folds) + 1, 'fit_n': len(fitting), 'held_n': len(y),
                          'steps': fit['optimizer_steps'], 'head_cuda_ops': audit['tensor_ops'],
                          'tcn': e._metrics(y, tcn), 'baseline': e._metrics(y, baseline)})
            truth.extend(y); tcn_all.extend(tcn); base_all.extend(baseline)
            del engine, model
        if len(truth) != len(train) or pose.profile['cpu_node_events'] != 0:
            raise RuntimeError('coverage or pose placement mismatch')
        return {'schema': 'qolda-e01-train-lopo-v1', 'manifest_sha256':pin,'source_sha256':RECIPE,
                'pose_sha256':e.POSE_SHA,'device':stack['device'],'pose_profile':pose.profile,
                'recipe':{'version':e.VERSION,'epochs':e.EPOCHS,'seed':e.SEED,'confidence_floor':e.MIN_CONFIDENCE},
                'validation_media_read':False,'fold_weights_exported':False,'folds':folds,
                'pooled':{'tcn':e._metrics(truth,tcn_all),'baseline':e._metrics(truth,base_all)},
                'participant_macro_ba_with_rejects':{k:mean(f[k]['all_with_abstention_penalty']['ba'] for f in folds) for k in ('tcn','baseline')}}
    finally:
        pose.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.manifest),indent=2,allow_nan=False))
