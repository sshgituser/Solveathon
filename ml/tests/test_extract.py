import numpy as np
import pytest


def test_phase_indices_sample_start_and_end_without_exposing_duration():
    from ml.extract import phase_indices
    for n in [21, 200]:
        idx = phase_indices(n)
        assert len(idx) == 64
        assert idx[0] == 0 and idx[-1] == n - 1
        assert (np.diff(idx) >= 0).all()
    assert not len(phase_indices(0))


def test_smoke_is_train_only_class_and_subject_stratified():
    from ml.extract import smoke_rows
    rows = [{'split': split, 'participant': subject, 'label': label, 'path': str(i)}
            for split, subject in [('train','01'), ('train','02'), ('validation','05'), ('test','08')]
            for label in (0, 1) for i in range(4)]
    smoke = smoke_rows(rows)
    assert len(smoke) == 12
    assert {row['split'] for row in smoke} == {'train'}
    assert len({(row['participant'], row['label']) for row in smoke}) == 4


def test_forearm_extraction_never_duplicates_short_native_frames():
    from ml.extract import phase_indices
    assert phase_indices(43, repeat_short=False).tolist() == list(range(43))
    assert np.array_equal(phase_indices(123, repeat_short=False), phase_indices(123))
