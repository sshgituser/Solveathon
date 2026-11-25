import os
from pathlib import Path

import numpy as np
import pytest


def test_report_keeps_abstains_out_of_confusion_but_in_denominator():
    from ml.scoring import metrics, select_threshold
    report = metrics([0,0,1,1,1], [.1,.45,.7,.3,None], [None]*4+['no_pose'], .5)
    assert report['confusion'] == [[2,0],[1,1]]
    assert report['n'] == 5 and report['n_scored'] == 4
    assert report['coverage'] == .8 and report['balanced_accuracy'] == .75
    assert report['f1_complete'] == pytest.approx(2/3)
    assert report['abstain_reasons'] == {'no_pose': 1}
    assert report['unconditional_balanced_recall'] == pytest.approx(2/3)
    assert select_threshold([0,1], [.1,.9]) == .5
    assert metrics([0,1], [None,None], ['no_pose']*2,.5)['status'] == 'no_scorable'


def test_freeze_lock_cannot_be_opened_twice(tmp_path):
    from ml.scoring import open_test_once
    target = tmp_path/'sealed.json'
    open_test_once(target, {'a': 'selected'})
    with pytest.raises(FileExistsError):
        open_test_once(target, {'a': 'different'})
    assert 'selected' in target.read_text()
