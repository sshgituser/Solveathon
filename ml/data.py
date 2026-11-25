"""Participant split and common, fixed-length pose representation."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np

SPLIT = {'01': 'train', '02': 'train', '03': 'train', '04': 'train',
         '05': 'validation', '08': 'test'}
COUNTS = {'train': 77, 'validation': 20, 'test': 20, 'quarantine': 10}
SCHEMA = 'seriq-e02-pose-v1'
FRAMES = 64
ARM = ((11, 13, 15), (12, 14, 16))
MIN_ARM_FRAMES = 20  # at least 31.25% of normalized cycle on the SAME side
STRICT_POLICY = 'seriq-e02-arm-triplet-v1'
FOREARM_POLICY = 'seriq-e02-forearm-2d-v1'
POSE_POLICIES = (STRICT_POLICY, FOREARM_POLICY)
RTMPOSE_SHA = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
FOREARM_NTU = (9, 10)
FOREARM_MIN_FRAMES = 40
FOREARM_MIN_QUARTER = 6
# Kinect v2 indices 1..25. None means no well-matched measured MP joint.
# The tuple for SpineMid/SpineShoulder/SpineBase averages observed landmarks.
NTU_MAP = ((23, 24), (11, 12, 23, 24), None, None,
           (11,), (13,), (15,), (19,), (12,), (14,), (16,), (20,),
           (23,), (25,), (27,), (31,), (24,), (26,), (28,), (32,),
           (11, 12), None, (21,), None, (22,))


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_training_inputs(metadata, *, manifest, original, graph):
    """Bind runtime inputs to digests recorded by training inside the weights."""
    for field, supplied in (('manifest_sha256', manifest), ('pose_model_sha256', original),
                            ('pose_graph_sha256', graph)):
        if (not isinstance(supplied, str) or not re.fullmatch('[0-9a-f]{64}', supplied)
                or metadata.get(field) != supplied):
            raise ValueError(f'{field} does not match training checkpoint metadata')


def build_manifest(root):
    root = Path(root)
    rows, seen = [], set()
    for path in sorted((p for p in root.rglob('*') if p.suffix.lower() == '.mp4'),
                       key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        match = re.fullmatch(r'(\d{2})_02_(00|01)_\d{2}', path.stem)
        if not match or len(relative.parts) != 3:
            raise ValueError(f'unexpected E02 file: {relative}')
        participant, code = match.groups()
        label = int(code)
        directory, folder, _ = relative.parts
        split = SPLIT.get(participant)
        if not split or folder != ('Test' if split == 'test' else 'Train'):
            raise ValueError(f'unexpected participant/folder: {relative}')
        if (directory, folder, participant, code) == ('Complete', 'Train', '04', '00'):
            split, label = 'quarantine', None
        elif directory != ('Complete' if label else 'Incomplete'):
            raise ValueError(f'unexpected label conflict: {relative}')
        digest = sha256(path)
        if digest in seen:
            raise ValueError(f'duplicate bytes: {relative}')
        seen.add(digest)
        rows.append(dict(path=relative.as_posix(), participant=participant, label=label,
                         filename_code=code, split=split, sha256=digest,
                         reason='folder_filename_conflict' if label is None else None))
    return rows


def verify_counts(rows):
    actual = Counter(r['split'] for r in rows)
    classes = Counter((r['split'], r['label']) for r in rows)
    expected_classes = {('train', 0): 47, ('train', 1): 30, ('validation', 0): 10,
                        ('validation', 1): 10, ('test', 0): 10, ('test', 1): 10,
                        ('quarantine', None): 10}
    if actual != COUNTS or classes != expected_classes:
        raise ValueError(f'unexpected split counts: {actual}, {classes}')
    if len({r['sha256'] for r in rows}) != sum(COUNTS.values()):
        raise ValueError('duplicate hash in manifest')
    for r in rows:
        if r['split'] == 'quarantine':
            if (r['participant'], r['filename_code'], r['reason']) != ('04', '00', 'folder_filename_conflict'):
                raise ValueError('unexpected quarantine item')
        elif SPLIT.get(r['participant']) != r['split']:
            raise ValueError('participant split mismatch')


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                              allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def read_manifest(path):
    rows = json.loads(Path(path).read_text(encoding='utf8'))
    verify_counts(rows)
    return rows


def visible(xy, visibility, presence, confidence=.50):
    xy = np.asarray(xy)
    return (np.isfinite(xy).all(axis=-1) & (xy >= .01).all(axis=-1) &
            (xy <= .99).all(axis=-1) & (np.asarray(visibility) >= confidence) &
            (np.asarray(presence) >= confidence))


@dataclass
class Pose:
    coords: np.ndarray
    mask: np.ndarray
    angles: np.ndarray
    angle_mask: np.ndarray
    reason: str | None
    pair_frames: tuple[int, int] | None = None


def pose_sequence(xyz, mask, policy=STRICT_POLICY):
    """Input [T,33,3] (x right, y up, z away; coherent scale), [T,33] boolean."""
    xyz, mask = np.asarray(xyz, np.float32), np.asarray(mask, bool)
    if xyz.ndim != 3 or xyz.shape[1:] != (33, 3) or mask.shape != xyz.shape[:2]:
        raise ValueError('expected coordinates [T,33,3] and mask [T,33]')
    if policy == FOREARM_POLICY:
        return forearm_sequence(xyz, mask)
    if policy != STRICT_POLICY:
        raise ValueError('unsupported pose policy')
    if len(xyz):
        idx = np.rint(np.linspace(0, len(xyz) - 1, FRAMES)).astype(int)
        xyz, mask = xyz[idx], mask[idx] & np.isfinite(xyz[idx]).all(axis=-1)
    else:
        xyz, mask = np.zeros((FRAMES, 33, 3), np.float32), np.zeros((FRAMES, 33), bool)
    coords = np.zeros((FRAMES, 25, 3), np.float32)
    masks = np.zeros((FRAMES, 25), bool)
    for index, ids in enumerate(NTU_MAP):
        if ids is not None:
            good = mask[:, ids].all(axis=1)
            coords[good, index] = xyz[good][:, ids].mean(axis=1)
            masks[:, index] = good
    angles, angle_mask = np.zeros((FRAMES, 2), np.float32), np.zeros((FRAMES, 2), bool)
    lengths = []
    for side, (s, e, w) in enumerate(ARM):
        upper, lower = xyz[:, s] - xyz[:, e], xyz[:, w] - xyz[:, e]
        nu, nl = np.linalg.norm(upper, axis=-1), np.linalg.norm(lower, axis=-1)
        good = mask[:, [s,e,w]].all(axis=1) & (nu > 1e-5) & (nl > 1e-5)
        angle_mask[:, side] = good
        angles[good, side] = np.clip((upper[good] * lower[good]).sum(axis=-1) / (nu[good] * nl[good]), -1, 1)
        lengths.extend(nu[good].tolist())
    count = angle_mask.sum(axis=0)
    reason = ('no_visible_arm_triplet' if not count.max() else
              'insufficient_visible_arm_triplet' if count.max() < MIN_ARM_FRAMES else None)
    if lengths:
        side = int(count.argmax())
        first = np.flatnonzero(angle_mask[:, side])[0]
        # NTU joint index 1 (zero based), fallback to the first reliable shoulder.
        center = coords[first, 1].copy() if masks[first, 1] else xyz[first, ARM[side][0]].copy()
        # Nominal upper-arm length in metres controls camera/subject scale.
        coords[masks] = (coords[masks] - center) * (.30 / np.median(lengths))
    coords[~masks] = 0
    return Pose(coords, masks, angles, angle_mask, reason)


def forearm_sequence(xyz, mask):
    """Right-labelled image-plane pair only; neither gap filling nor side switching."""
    xy = np.zeros((FRAMES, 33, 2), np.float32)
    seen = np.zeros((FRAMES, 33), bool)
    if len(xyz):
        # Resample long streams; map short streams sparsely so a native point
        # can never count as several observed phase positions.
        src = np.rint(np.linspace(0, len(xyz) - 1, min(len(xyz), FRAMES))).astype(int)
        dst = np.rint(np.linspace(0, FRAMES - 1, len(src))).astype(int)
        xy[dst] = xyz[src, :, :2]
        seen[dst] = mask[src] & np.isfinite(xy[dst]).all(axis=-1)
    l, r = xy[:, [13, 15]], xy[:, [14, 16]]
    left, right = seen[:, [13, 15]].all(axis=1), seen[:, [14, 16]].all(axis=1)
    p = Pose(np.zeros((FRAMES, 25, 3), np.float32), np.zeros((FRAMES, 25), bool),
             np.zeros((FRAMES, 2), np.float32), np.zeros((FRAMES, 2), bool), None,
             (int(left.sum()), int(right.sum())))
    lengths = np.linalg.norm(r[:, 1] - r[:, 0], axis=1)
    keep = right & np.isfinite(lengths) & (lengths > 1e-5)
    if not right.any():
        p.reason = 'no_visible_forearm_pair'
    elif not keep.any():
        p.reason = 'degenerate_forearm_pair'
    elif right.sum() < FOREARM_MIN_FRAMES:
        p.reason = 'insufficient_visible_forearm_pair'
    else:
        scale = float(np.median(lengths[keep]))
        keep &= (lengths >= .60 * scale) & (lengths <= 1.45 * scale)
        # Geometric checks use projected lengths, not detector confidence. Near
        # coincident alternate labels never count as a second usable arm.
        left_length = np.linalg.norm(l[:, 1] - l[:, 0], axis=1)
        both = left & keep & (left_length >= .60 * scale) & (left_length <= 1.45 * scale)
        separation = np.linalg.norm(l[both] - r[both], axis=-1).max(axis=1)
        slots = np.flatnonzero(keep)
        steps = np.linalg.norm(np.diff(r[keep], axis=0), axis=-1).max(axis=1)
        jump_limit = scale * np.minimum(.20 * np.diff(slots), .55)
        if (right.sum() - keep.sum() > 6 or (steps > jump_limit).any()):
            p.reason = 'inconsistent_forearm_trajectory'
        elif (separation > .45 * scale).sum() >= 8:
            p.reason = 'ambiguous_forearm_tracks'
        elif len(slots) < FOREARM_MIN_FRAMES or keep.reshape(4, 16).sum(axis=1).min() < FOREARM_MIN_QUARTER:
            p.reason = 'incomplete_forearm_coverage'
        else:
            lo, hi = np.percentile(r[keep], (10, 90), axis=0)
            if np.linalg.norm(hi - lo, axis=-1).max() < .10 * scale:
                p.reason = 'static_forearm_trajectory'
            else:
                center = r[slots[0], 0]
                p.coords[np.ix_(slots, FOREARM_NTU, (0, 1))] = (r[slots] - center) * (.30 / scale)
                p.mask[np.ix_(slots, FOREARM_NTU)] = True
    return p
