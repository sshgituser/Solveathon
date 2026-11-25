from pathlib import Path
import hashlib
import pytest
import numpy as np


def test_manifest_quarantines_conflicts_without_collapsing_stems(tmp_path):
    from ml.data import build_manifest, verify_counts
    paths = ['Complete/Train/04_02_00_01.mp4', 'Incomplete/Train/04_02_00_01.mp4',
             'Complete/Train/05_02_01_01.mp4', 'Incomplete/Test/08_02_00_01.mp4']
    for i, name in enumerate(paths):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([i]))
    rows = {r['path']: r for r in build_manifest(tmp_path)}
    assert [rows[p]['split'] for p in paths] == ['quarantine', 'train', 'validation', 'test']
    assert rows[paths[0]]['label'] is None
    assert rows[paths[1]]['sha256'] == hashlib.sha256(b'\x01').hexdigest()
    with pytest.raises(ValueError, match='counts'):
        verify_counts(list(rows.values()))


def test_missing_pose_abstains_and_bad_coordinates_are_not_filled():
    from ml.data import pose_sequence
    xyz = np.zeros((64, 33, 3), np.float32)
    mask = np.zeros((64, 33), bool)
    assert pose_sequence(xyz, mask).reason == 'no_visible_arm_triplet'
    xyz[:, 11] = [0.3, 0.1, 0.1]
    xyz[:, 13] = [0.6, 0.1, 0.1]
    xyz[:, 15] = [0.6, 0.4, 0.1]
    mask[:, [11, 13, 15]] = True
    result = pose_sequence(xyz, mask)
    assert result.reason is None
    assert result.coords.shape == (64, 25, 3)
    assert result.mask[:, 4:7].all() and not result.mask[:, 8:11].any()
    assert np.all(result.coords[~result.mask] == 0)
    np.testing.assert_allclose(result.angles[:, 0], 0, atol=1e-6)
    assert not result.angle_mask[:, 1].any()


def test_out_of_frame_and_low_confidence_are_masked():
    from ml.data import visible
    p = np.array([[.2, .7], [-.01, .5], [.4, .5], [.3, .5]])
    np.testing.assert_array_equal(visible(p, [.9, .9, .2, .9], [.9, .9, .9, .2]), [1, 0, 0, 0])
