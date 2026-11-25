import os
from pathlib import Path

import numpy as np
import pytest


def test_synthetic_gpu_fit_updates_and_reloads_both_formats(tmp_path, monkeypatch):
    import torch
    from ml.data import pose_sequence
    from ml.train import HYPER, fit_one

    base = os.environ.get('SERIQ_SLIM_BASE')
    if not base or os.environ.get('SERIQ_CHECK_DEVICE') != 'cuda':
        pytest.skip('requires pinned SLiM and explicit CUDA test opt-in')
    assert torch.cuda.is_available()
    rng = np.random.default_rng(601)
    samples = []
    for k in range(12):
        label = k % 2
        xyz = np.zeros((64,33,3),np.float32)
        mask = np.zeros((64,33),bool)
        mask[:,[11,13,15]] = True
        xyz[:,11] = [0.,0.,0.]
        xyz[:,13] = [.3,0.,0.]
        phi = np.sin(np.linspace(0,np.pi,64))* (2.5 if label else 1.2) + rng.normal(0,.01,64)
        xyz[:,15] = xyz[:,13] + np.column_stack((-.24*np.cos(phi),.24*np.sin(phi),np.zeros(64)))
        samples.append((label,pose_sequence(xyz,mask),None))
    for kind in ('A','B'):
        monkeypatch.setitem(HYPER[kind], 'epochs', 2)
        artifact = tmp_path/f'{kind}.safetensors'
        actual = fit_one(kind, samples[:8], samples[8:], Path(base),
                         {'engine':'rtmpose','sha256':'a'*64},torch.device('cuda'),
                         artifact,{'manifest_sha256':'b'*64})
        assert actual['reload']['labels_match'] and actual['reload']['n_compared'] == 4
        assert actual['training']['steps_executed'] == 2
        assert actual['steps'] > 0
        assert actual['training']['initial_tensor_sha256'] != actual['training']['selected_tensor_sha256']
        if kind == 'B':
            assert actual['training']['frozen_tensors_verified'] == 92
            assert sum(k.startswith('encoder.blocks.7.') for k in actual['training']['updated_tensors']) == 12


def test_sealed_admission_requires_both_classes_and_both_models_above_mask_control():
    from ml.train import forearm_admission
    good = dict(class_total=[10,10], class_scored=[9,9], confusion=[[8,1],[1,8]],
                balanced_accuracy=8/9, unconditional_balanced_recall=.8)
    bad = dict(good, class_scored=[9,7], confusion=[[8,1],[2,5]],
               balanced_accuracy=.80, unconditional_balanced_recall=.65)
    report = {'models': {'A': {'validation': good}, 'B': {'validation': bad}},
              'mask_only': {'validation': {'unconditional_balanced_recall': .65}}}
    assert forearm_admission(report)['A'] == []
    assert set(forearm_admission(report)['B']) == {'class_coverage', 'per_class_recall', 'mask_control'}


def test_mask_control_never_uses_coordinates():
    from ml.data import pose_sequence
    from ml.train import mask_control
    from ml.tests.test_forearm import observed_pair, POLICY
    left = []
    for i in range(8):
        xyz, mask = observed_pair()
        if i % 2:
            mask[4:10, 16] = False
        p = pose_sequence(xyz, mask, POLICY)
        left.append((i % 2, p, p.reason))
    before = mask_control(left[:4], left[4:])
    for _, pose, _ in left:
        pose.coords[:] = float('nan')
    assert mask_control(left[:4], left[4:]) == before
    assert before['validation']['balanced_accuracy'] == 1.
