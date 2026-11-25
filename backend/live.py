"""One explicitly delimited E02 camera attempt. Never retain RGB frames or record media."""
from __future__ import annotations

import secrets
from threading import Lock, Timer
from time import monotonic

from backend.contract import validate_result

LIVE_VERSION = 'seriq.live.v1'
CAPTURE_SECONDS = 30
FINISH_GRACE = 5
MAX_FRAMES = 360
MAX_SIDE = 480
PREVIEW_MIN = 40  # same lower bound as ml.data.FOREARM_MIN_FRAMES


class LiveError(Exception):
    def __init__(self, status, code):
        super().__init__(code)
        self.status, self.code = status, code


class CameraAttempt:
    def __init__(self, registry, extractor, classifier, version):
        self.registry = registry
        self.extractor = extractor
        self.classifier = classifier
        self.version = version
        self.lock = Lock()
        self.token = None
        self.timer = None
        self.samples = []
        self.started = 0.
        self.last_preview = None
        self.visible_frames = 0
        self.scored_at = 0

    def _clear(self):
        if self.timer:
            self.timer.cancel()
        self.timer = self.token = self.last_preview = None
        self.samples.clear()
        self.visible_frames = self.scored_at = 0

    def _expire(self, token):
        with self.lock:
            if self.token == token:
                self._clear()

    def _check(self, token):
        if not self.token or token != self.token:
            raise LiveError(410, 'session_expired')
        if monotonic() - self.started > CAPTURE_SECONDS + FINISH_GRACE:
            self._clear()
            raise LiveError(410, 'session_expired')

    def start(self):
        with self.lock:
            if self.token and monotonic() - self.started < CAPTURE_SECONDS + FINISH_GRACE:
                raise LiveError(503, 'engine_busy')
            self._clear()
            self.token = secrets.token_urlsafe(24)
            self.started = monotonic()
            self.timer = Timer(CAPTURE_SECONDS + FINISH_GRACE, self._expire, args=(self.token,))
            self.timer.daemon = True
            self.timer.start()
            return {'version': LIVE_VERSION, 'session': self.token, 'max_seconds': CAPTURE_SECONDS}

    def _score(self):
        import numpy as np
        from backend.runtime import wire_result
        # The frozen policy reads only the projected elbows/wrists (13..16).
        # Restore its input layout for scoring; do not retain a full-body trace.
        n = len(self.samples)
        xyz = np.zeros((n, 33, 3), dtype=np.float32)
        mask = np.zeros((n, 33), dtype=bool)
        xy, seen = zip(*self.samples)
        xyz[:, 13:17, :2] = np.stack(xy)
        mask[:, 13:17] = np.stack(seen)
        result = self.classifier.predict_pose(xyz, mask)
        return validate_result(wire_result('scratch', self.version, result, self.classifier.metadata['low_confidence_below']))

    def _gpu_error(self, exc):
        self._clear()
        self.registry.mark_dead('scratch', 'live inference failed')
        raise LiveError(503, 'model_unavailable') from exc

    def frame(self, token, seq, w, h, raw):
        import numpy as np
        with self.lock:
            self._check(token)
            if seq != len(self.samples) + 1:
                raise LiveError(409, 'frame_sequence')
            if seq > MAX_FRAMES or monotonic() - self.started > CAPTURE_SECONDS:
                raise LiveError(422, 'duration_limit')
            if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE and len(raw) == w*h*3):
                raise LiveError(422, 'invalid_frame')
            # The browser supplies RGB; the same strict extractor as file analysis expects BGR.
            bgr = np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 3)[:, :, ::-1])
            try:
                xyz, mask = self.extractor(bgr)
                seen = mask[13:17].copy()
                xy = xyz[13:17, :2].copy()
                xy[~seen] = 0
                self.samples.append((xy, seen))
                visible = bool(mask[14] and mask[16])
                self.visible_frames += int(visible)
                if not visible:
                    self.last_preview = None
                elif self.visible_frames >= PREVIEW_MIN and seq - self.scored_at >= 4:
                    self.scored_at = seq
                    candidate = self._score()
                    # A quality refusal can contain an 'uncertain' attempt;
                    # only a classifier output has a numeric confidence.
                    scored = any(a['confidence'] is not None for a in candidate['attempts'])
                    self.last_preview = candidate if scored else None
            except Exception as exc:
                self._gpu_error(exc)
            return {'version': LIVE_VERSION, 'frames': seq, 'visible': visible,
                    'phase': 'ready' if self.last_preview is not None else 'collecting', 'preview': self.last_preview}

    def finish(self, token):
        with self.lock:
            self._check(token)
            try:
                if not self.samples:
                    raise LiveError(422, 'empty_session')
                return self._score()
            except LiveError:
                raise
            except Exception as exc:
                self._gpu_error(exc)
            finally:
                self._clear()

    def cancel(self, token):
        with self.lock:
            self._check(token)
            self._clear()
            return {'closed': True}
