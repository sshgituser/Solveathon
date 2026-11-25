import json

import numpy as np
import pytest


def test_safetensors_round_trip_and_abstention(tmp_path):
    from ml.api import load_artifact, save_artifact
    from ml.models import ModelA
    record = {'kind':'A','threshold':.5,'seed':601,'pose_engine':'rtmpose','pose_model_sha256':'a'*64}
    target = tmp_path/'a.safetensors'
    save_artifact(target, ModelA().eval().state_dict(), record)
    model = load_artifact(target)
    xyz = np.zeros((64,33,3), np.float32)
    mask = np.zeros((64,33), bool)
    missing = model.predict_pose(xyz,mask)
    assert missing['p_complete'] is None and missing['reason'] == 'no_visible_arm_triplet'
    xyz[:,[11,13,15],0] = [1,2,3]
    mask[:,[11,13,15]] = True
    result = model.predict_pose(xyz,mask)
    assert result == load_artifact(target).predict_pose(xyz,mask)
    assert result['label'] in {'incomplete','complete'} and result['reason'] is None
