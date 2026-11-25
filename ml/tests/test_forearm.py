import numpy as np
import pytest

from ml.data import pose_sequence

POLICY = 'seriq-e02-forearm-2d-v1'


def observed_pair(n=64):
    xyz = np.zeros((n, 33, 3), np.float32)
    mask = np.zeros((n, 33), bool)
    phase = np.linspace(0, 1, n, dtype=np.float32)
    xyz[:, 14, 0] = .3 + .1 * phase
    xyz[:, 14, 1] = -.4
    xyz[:, 16, :2] = xyz[:, 14, :2] + [.2, 0]
    mask[:, [14, 16]] = True
    return xyz, mask


def test_observed_pair_survives_single_similarity_transform_without_shoulder_angle_or_depth():
    xyz, mask = observed_pair()
    p = pose_sequence(xyz, mask, POLICY)
    assert p.reason is None
    assert p.mask.sum() == 128
    assert p.mask[:, [9, 10]].all()
    np.testing.assert_allclose(p.coords[:, [9, 10], :2], (xyz[:, [14, 16], :2] - xyz[0, 14, :2]) * 1.5, atol=1e-6)
    assert np.count_nonzero(p.coords[..., 2]) == 0
    assert not p.angle_mask.any() and not p.angles.any()
    assert pose_sequence(xyz, mask).reason == 'no_visible_arm_triplet'


@pytest.mark.parametrize('missing', ['elbow', 'all', 'right'])
def test_cannot_score_wrist_only_pose_missing_or_fallback_to_other_side(missing):
    xyz, mask = observed_pair()
    if missing == 'elbow':
        mask[:, 14] = False
    elif missing == 'all':
        xyz, mask = xyz[:0], mask[:0]
    else:
        xyz[:, [13, 15]], mask[:, [13, 15]] = xyz[:, [14, 16]], True
        mask[:, [14, 16]] = False
    assert pose_sequence(xyz, mask, POLICY).reason == 'no_visible_forearm_pair'


def test_duplicate_left_and_other_values_neither_copy_nor_change_the_two_observations():
    xyz, mask = observed_pair()
    p = pose_sequence(xyz, mask, POLICY)
    xyz[:, [13, 15]] = xyz[:, [14, 16]]  # two labels for the same apparent limb
    xyz[:, [12, 24]] = 4
    xyz[:, [14, 16], 2] = np.nan  # no use of depth even if supplied
    mask[:, [12, 13, 15, 24]] = True
    other = pose_sequence(xyz, mask, POLICY)
    assert other.reason is None
    np.testing.assert_array_equal(other.coords, p.coords)
    np.testing.assert_array_equal(other.mask, p.mask)


def test_gap_not_filled_even_when_duplicate_left_pair_remains_observable():
    xyz, mask = observed_pair()
    xyz[:, [13, 15]], mask[:, [13, 15]] = xyz[:, [14, 16]], True
    mask[18:22, 14] = False
    p = pose_sequence(xyz, mask, POLICY)
    assert p.reason is None
    assert not p.mask[18:22].any() and not p.coords[18:22].any()
    assert p.mask[[17, 22]][:, [9, 10]].all()


@pytest.mark.parametrize(('fault', 'reason'), [
    ('collapsed', 'degenerate_forearm_pair'),
    ('jump', 'inconsistent_forearm_trajectory'),
    ('static', 'static_forearm_trajectory'),
    ('ambiguous', 'ambiguous_forearm_tracks'),
    ('one_end_missing', 'incomplete_forearm_coverage'),
    ('too_short', 'insufficient_visible_forearm_pair'),
])
def test_reject_unusable_forearm_trajectory(fault, reason):
    xyz, mask = observed_pair()
    if fault == 'collapsed':
        xyz[:, 16] = xyz[:, 14]
    if fault == 'jump':
        xyz[31, [14, 16], 1] += .5
    if fault == 'static':
        xyz[:] = xyz[0]
    if fault == 'ambiguous':
        xyz[:, [13, 15]] = xyz[:, [14, 16]] + [.0, .3, .0]
        mask[:, [13, 15]] = True
    if fault == 'one_end_missing':
        mask[-15:, 16] = False
    if fault == 'too_short':
        xyz, mask = xyz[:24], mask[:24]
    assert pose_sequence(xyz, mask, POLICY).reason == reason


def test_native_short_track_does_not_repeat_observations():
    xyz, mask = observed_pair(48)
    p = pose_sequence(xyz, mask, POLICY)
    assert p.reason is None
    assert p.mask[:, 9].sum() == 48
    assert not p.coords[~p.mask].any()


def test_artifact_uses_forearm_policy_and_exposes_score_diagnostics(tmp_path):
    from ml.api import load_artifact, save_artifact
    from ml.data import RTMPOSE_SHA
    from ml.models import ModelA
    record = dict(kind='A', threshold=.5, pose_policy=POLICY, pose_engine='rtmpose', pose_model_sha256=RTMPOSE_SHA)
    path = tmp_path/'forearm.safetensors'
    save_artifact(path, ModelA(POLICY).state_dict(), record)
    loaded = load_artifact(path)
    assert loaded.metadata['pose_policy'] == POLICY
    assert loaded.metadata['input_channels'] == 'right_elbow_xy,right_wrist_xy,pair_mask'
    xyz, mask = observed_pair()
    result = loaded.predict_pose(xyz, mask)
    assert result['reason'] is None and result['label'] in {'complete', 'incomplete'}
    assert result['valid_arm_frames'] == [0, 0] and result['valid_pair_frames'] == [0, 64]
    assert result['confidence'] == pytest.approx(2 * abs(result['p_complete'] - .5))
    mask[:, 14] = False
    result = loaded.predict_pose(xyz, mask)
    assert result['reason'] == 'no_visible_forearm_pair'
    assert result['p_complete'] is None and result['confidence'] is None and result['label'] is None


def test_artifact_rejects_unpinned_forearm_extractor(tmp_path):
    from ml.api import load_artifact, save_artifact
    from ml.models import ModelA
    record = dict(kind='A', threshold=.5, pose_policy=POLICY, pose_engine='rtmpose', pose_model_sha256='a'*64)
    path = tmp_path/'forearm.safetensors'
    save_artifact(path, ModelA(POLICY).state_dict(), record)
    with pytest.raises(ValueError, match='pinned RTMPose'):
        load_artifact(path)
