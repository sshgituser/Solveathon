"""Load serialized SerIQ weights and classify a masked E02 pose cycle locally."""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from ml.data import FOREARM_NTU, FOREARM_POLICY, POSE_POLICIES, RTMPOSE_SHA, SCHEMA, STRICT_POLICY, pose_sequence, sha256
from ml.models import ModelA, ModelB, SLIM_SHA, base_encoder, trainable

ARTIFACT_FORMAT = 'seriq-safetensors-v1'
CONFIDENCE_BAND = .10
FOREARM_METADATA = dict(input_channels='right_elbow_xy,right_wrist_xy,pair_mask', pose_source_side='right',
                        pose_dimensions=2, phase_sampling='unique-native-to-64-v1',
                        confidence_kind='normalized_threshold_margin_uncalibrated',
                        low_confidence_below=CONFIDENCE_BAND)


def threshold_confidence(p, threshold):
    return (p-threshold) / (1-threshold) if p >= threshold else (threshold-p) / threshold


def save_artifact(path, weights, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    detached = {name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()}
    policy = record.get('pose_policy', STRICT_POLICY)
    fields = FOREARM_METADATA if policy == FOREARM_POLICY else {}
    save_file(detached, str(tmp), metadata={'seriq': json.dumps(
        {**record, **fields, 'format': ARTIFACT_FORMAT, 'pose_schema': SCHEMA, 'pose_policy': policy},
        sort_keys=True, allow_nan=False)})
    if tmp.stat().st_size > 90_000_000:
        tmp.unlink()
        raise ValueError('artifact exceeds 90 MB')
    tmp.replace(path)


def tensor_batch(poses, device):
    return tuple(torch.as_tensor(np.stack([getattr(p, k) for p in poses]), device=device)
                 for k in ('coords','mask','angles','angle_mask'))


class Predictor:
    def __init__(self, model, metadata, device):
        self.model = model.eval()
        self.metadata = metadata
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict_many(self, poses, batch_size=8):
        if any(p.reason is not None for p in poses):
            raise ValueError('predict_many expects scorable poses only')
        values = []
        for start in range(0, len(poses), batch_size):
            data = tensor_batch(poses[start:start+batch_size], self.device)
            values.extend(self.model(*data).float().sigmoid().cpu().tolist())
        return values

    def predict_pose(self, xyz, mask):
        """Classify [T,33,3] (x-right,y-up,z-away) with bool validity [T,33]."""
        policy = self.metadata.get('pose_policy', STRICT_POLICY)
        pose = pose_sequence(xyz, mask, policy)
        n_arm = pose.angle_mask.sum(axis=0).tolist()
        fields = {}
        if policy == FOREARM_POLICY:
            fields = dict(pose_policy=policy, valid_pair_frames=list(pose.pair_frames),
                          usable_pair_frames=int(pose.mask[:, FOREARM_NTU[0]].sum()), confidence=None)
        if pose.reason:
            return dict(label=None, p_complete=None, reason=pose.reason, valid_arm_frames=n_arm, **fields)
        p = self.predict_many([pose])[0]
        t = self.metadata['threshold']
        if policy == FOREARM_POLICY:
            fields['confidence'] = threshold_confidence(p, t)
        return dict(label='complete' if p >= t else 'incomplete',
                    p_complete=p, reason=None, valid_arm_frames=n_arm, **fields)


def load_artifact(path, slim_base=None, device='cpu', expected_sha256=None):
    """Artifact A is standalone; B overlays the block/head on the pinned official base."""
    path = Path(path)
    if expected_sha256 is not None and expected_sha256 != sha256(path):
        raise ValueError('artifact SHA-256 mismatch')
    with safe_open(str(path), framework='pt') as f:
        record = json.loads((f.metadata() or {}).get('seriq', '{}'))
    if record.get('format') != ARTIFACT_FORMAT or record.get('pose_schema') != SCHEMA:
        raise ValueError('unsupported artifact or pose format')
    policy = record.setdefault('pose_policy', STRICT_POLICY)
    if policy not in POSE_POLICIES:
        raise ValueError('unsupported pose policy')
    if policy == FOREARM_POLICY:
        if (record.get('pose_engine'), record.get('pose_model_sha256')) != ('rtmpose', RTMPOSE_SHA):
            raise ValueError('forearm policy requires the pinned RTMPose extractor')
        if any(record.get(k) != v for k, v in FOREARM_METADATA.items()):
            raise ValueError('inconsistent forearm pose metadata')
    if not 0 < record.get('threshold', -1) < 1:
        raise ValueError('invalid threshold')
    state = load_file(str(path))
    kind = record.get('kind')
    if kind == 'A':
        model = ModelA(policy)
        model.load_state_dict(state, strict=True)
    elif kind == 'B':
        if slim_base is None or record.get('slim_base_sha256') != SLIM_SHA:
            raise ValueError('B requires the pinned official NTU60 XSub encoder file')
        model = ModelB(base_encoder(slim_base))
        expected = trainable(model, kind)
        if set(state) != set(expected):
            raise ValueError('B artifact must contain exactly the final encoder block and binary head')
        model.load_state_dict({**model.state_dict(), **state}, strict=True)
    else:
        raise ValueError('unknown model kind')
    return Predictor(model.to(device), record, device)
