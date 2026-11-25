"""Aggregate binary metrics; abstention is not an incomplete prediction."""
import json
from pathlib import Path

import numpy as np


def metrics(labels, probabilities, reasons, threshold):
    if len(labels) != len(probabilities) or len(labels) != len(reasons):
        raise ValueError('one probability or abstention per clip required')
    cm = np.zeros((2, 2), dtype=int)
    totals = np.zeros(2, dtype=int)
    abstain = {}
    for y, p, reason in zip(labels, probabilities, reasons, strict=True):
        if y not in (0, 1):
            raise ValueError('expected binary label')
        totals[y] += 1
        if p is None:
            if reason is None:
                raise ValueError('missing abstention reason')
            abstain[reason] = abstain.get(reason, 0) + 1
        else:
            if reason is not None or not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('invalid scored sample')
            cm[y, int(p >= threshold)] += 1
    n, n_scored = int(totals.sum()), int(cm.sum())
    support = cm.sum(axis=1)
    diag = cm.diagonal()
    denom = support + cm.sum(axis=0)
    f1 = np.divide(2 * diag, denom, out=np.zeros(2), where=denom > 0) if n_scored else [None, None]
    both = bool((support > 0).all())
    return dict(n=n, n_scored=n_scored, n_abstain=n-n_scored, coverage=n_scored/n if n else None,
                status='ok' if both else 'single_scorable_class' if n_scored else 'no_scorable',
                abstain_reasons=abstain, class_order=['incomplete', 'complete'], confusion=cm.tolist(),
                class_total=totals.tolist(), class_scored=support.tolist(),
                balanced_accuracy=float((diag/support).mean()) if both else None,
                f1_macro=float(np.mean(f1)) if n_scored else None,
                f1_complete=float(f1[1]) if n_scored else None,
                unconditional_balanced_recall=float((diag/totals).mean()) if (totals > 0).all() else None)


def select_threshold(labels, probs):
    if set(labels) != {0, 1}:
        raise ValueError('threshold selection requires both scorable validation classes')
    thresholds = [.5, *[x / 20 for x in range(2, 19) if x != 10]]
    def key(t):
        m = metrics(labels, probs, [None]*len(labels), t)
        return (m['balanced_accuracy'], m['f1_macro'], -abs(t-.5))
    return max(thresholds, key=key)


def open_test_once(path, freeze):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(freeze, f, indent=2, sort_keys=True)
        f.write('\n')
