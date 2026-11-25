"""Local pose extraction; do not write decoded images or public skeletons."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ml.data import ARM, FOREARM_NTU, FOREARM_POLICY, STRICT_POLICY, dump_json, pose_sequence, read_manifest, sha256, visible


def phase_indices(n, repeat_short=True):
    count = 64 if repeat_short else min(n, 64)
    return np.rint(np.linspace(0, n - 1, count)).astype(int) if n > 0 else np.empty(0, int)


def smoke_rows(rows):
    strata = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            strata[row['participant'], row['label']].append(row)
    return [r for key in sorted(strata) for r in sorted(strata[key], key=lambda x: x['path'])[:3]]


def frames(path, repeat_short=True):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return
        cursor = -1
        frame = None
        for target in phase_indices(n, repeat_short=repeat_short):
            while cursor < target:
                cursor += 1
                if cursor == target:
                    ok, frame = cap.read()
                else:
                    ok = cap.grab()
                if not ok:
                    yield None
                    return
            yield frame
    finally:
        cap.release()


def blank():
    return np.zeros((33, 3), np.float32), np.zeros(33, bool)


class MediaPipe:
    def __init__(self, model):
        import mediapipe as mp
        self.mp = mp
        self.model = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1, min_pose_detection_confidence=.5, min_pose_presence_confidence=.5))

    def close(self):
        self.model.close()

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not out.pose_landmarks or not out.pose_world_landmarks:
            return blank()
        image, world = out.pose_landmarks[0], out.pose_world_landmarks[0]
        mask = visible([(p.x, p.y) for p in image], [p.visibility for p in image], [p.presence for p in image])
        xyz = np.array([(p.x, -p.y, p.z) for p in world], np.float32)
        return xyz, mask & np.isfinite(xyz).all(axis=-1)


class MoveNet:
    # COCO 17 -> subset of MediaPipe 33. Unmatched joints remain masked.
    INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, model):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        self.model = Interpreter(model_path=str(model), num_threads=2)
        self.model.allocate_tensors()
        self.inp = self.model.get_input_details()[0]
        self.out = self.model.get_output_details()[0]['index']

    def close(self):
        pass

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        side = int(self.inp['shape'][1])
        gain = side / max(h, w)
        nh, nw = round(h * gain), round(w * gain)
        top, left = (side - nh) // 2, (side - nw) // 2
        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[top:top+nh, left:left+nw] = cv2.cvtColor(cv2.resize(bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
        self.model.set_tensor(self.inp['index'], canvas[None].astype(self.inp['dtype']))
        self.model.invoke()
        y, x, score = self.model.get_tensor(self.out)[0, 0].T
        x, y = (x * side - left) / nw, (y * side - top) / nh
        xyz, mask = blank()
        xyz[self.INDICES] = np.stack((x * w / max(h, w), -y * h / max(h, w), np.zeros_like(x)), axis=1)
        mask[self.INDICES] = visible(np.stack((x, y), axis=1), score, score, .35)
        return xyz, mask


class RTMPose:
    """Official OpenMMLab body7 Gaussian SimCC ONNX, single-person frame box."""
    def __init__(self, model):
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.mean = np.asarray([123.675, 116.28, 103.53], np.float32)
        self.std = np.asarray([58.395, 57.12, 57.375], np.float32)

    def close(self):
        pass

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(192/w, 256/h) / 1.25
        left, top = (192 - scale*w)/2, (256 - scale*h)/2
        transform = np.array([[scale, 0, left], [0, scale, top]], np.float32)
        image = cv2.warpAffine(bgr, transform, (192,256), flags=cv2.INTER_LINEAR)
        rgb = (image[:,:,::-1].astype(np.float32) - self.mean) / self.std
        self.net.setInput(np.ascontiguousarray(rgb.transpose(2,0,1)[None]))
        sx, sy = (a[0] for a in self.net.forward(('simcc_x', 'simcc_y')))
        # Argmax and min(axis maxima) match the official SimCC decoder.
        x = (sx.argmax(axis=1)/2 - left)/(scale*w)
        y = (sy.argmax(axis=1)/2 - top)/(scale*h)
        score = np.minimum(sx.max(axis=1), sy.max(axis=1))
        xyz, mask = blank()
        xyz[MoveNet.INDICES] = np.stack((x*w/max(h,w), -y*h/max(h,w), np.zeros_like(x)), axis=1)
        mask[MoveNet.INDICES] = visible(np.stack((x, y), axis=1), score, score, .50)
        return xyz, mask


ENGINES = {'mediapipe': MediaPipe, 'movenet': MoveNet, 'rtmpose': RTMPose}


def extract_one(path, extractor, repeat_short=True):
    xyz, mask = [], []
    prior = None
    for frame in frames(path, repeat_short=repeat_short):
        if frame is None:
            return np.empty((0, 33, 3), np.float32), np.empty((0, 33), bool)
        # Very short clips repeat nearest frame; reuse inference only for that frame.
        if frame is not prior:
            prediction = extractor(frame)
            prior = frame
        x, m = prediction
        xyz.append(x)
        mask.append(m)
    return np.asarray(xyz, np.float32).reshape(-1, 33, 3), np.asarray(mask, bool).reshape(-1, 33)


def coverage(xyz, mask, policy=STRICT_POLICY):
    p = pose_sequence(xyz, mask, policy)
    out = dict(reason='decode_failure' if not len(xyz) else p.reason,
                valid_arm_frames=p.angle_mask.sum(axis=0).tolist(),
                valid_joint_frames={str(k): int(mask[:, k].sum()) for side in ARM for k in side})
    if policy == FOREARM_POLICY:
        out.update(valid_pair_frames=p.pair_frames, usable_pair_frames=int(p.mask[:, FOREARM_NTU[0]].sum()),
                   slots=max(1, len(xyz)))
    return out


def aggregate(items, policy=STRICT_POLICY):
    reasons = Counter(x['reason'] or 'scorable' for x in items)
    classes = {}
    for label in (0, 1):
        group = [x for x in items if x['label'] == label]
        classes[str(label)] = dict(n=len(group), scorable=sum(x['reason'] is None for x in group))
    slots = sum(x['slots'] for x in items) if policy == FOREARM_POLICY else 64 * len(items)
    out = dict(n=len(items), scorable=reasons['scorable'], reasons=dict(reasons), by_class=classes,
                mean_joint_visibility={str(k): sum(x['valid_joint_frames'][str(k)] for x in items) / slots
                                       for side in ARM for k in side} if items else {})
    if policy == FOREARM_POLICY:
        out['observed_pair_frames_total'] = [sum(x['valid_pair_frames'][s] for x in items) for s in (0, 1)]
        out['usable_right_frames_total'] = sum(x['usable_pair_frames'] for x in items)
    return out


def run_extraction(rows, root, engine, model, cache=None, policy=STRICT_POLICY):
    cv2.setNumThreads(1)
    extractor = ENGINES[engine](model)
    items = []
    try:
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        for row in rows:
            p = cache / (row['sha256'] + '.npz') if cache else None
            if p and p.is_file():
                with np.load(p, allow_pickle=False) as saved:
                    xyz, mask = saved['xyz'], saved['mask']
            else:
                path = root / row['path']
                if sha256(path) != row['sha256']:
                    raise ValueError(f'source changed: {row["path"]}')
                xyz, mask = extract_one(path, extractor, repeat_short=policy != FOREARM_POLICY)
                if p:
                    tmp = p.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp, xyz=xyz, mask=mask)
                    tmp.replace(p)
            items.append(dict(label=row['label'], **coverage(xyz, mask, policy)))
            if len(items) % 10 == 0:
                print(f'{engine}: {len(items)}/{len(rows)}', flush=True)
    finally:
        extractor.close()
    return aggregate(items, policy)


def cache_dir(private, engine, model):
    return Path(private) / f'pose-v1-{engine}-{sha256(model)}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'dev'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--pose-model', required=True, type=Path)
    parser.add_argument('--engine', choices=ENGINES, default='mediapipe')
    parser.add_argument('--manifest', default='ml/manifest.json')
    parser.add_argument('--private', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.private.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('pose cache must be outside checkout')
    cache = cache_dir(args.private, args.engine, args.pose_model)
    report = dict(engine=args.engine, pose_model_sha256=sha256(args.pose_model))
    splits = {'smoke': smoke_rows(rows)} if args.stage == 'smoke' else {
        s: [r for r in rows if r['split'] == s] for s in ('train', 'validation')}
    for name, subset in splits.items():
        report[name] = run_extraction(subset, args.root, args.engine, args.pose_model, cache)
        print(name, report[name], flush=True)
    dump_json(args.out, report)


if __name__ == '__main__':
    main()
