"""Backend adapter for the frozen E03 RGB model; approval belongs to the registry."""
from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Callable

from ml import extra_e03 as frozen


class E03Unavailable(RuntimeError):
    """The CUDA engine failed at runtime and must be removed from the registry."""


def open_engine(model_dir: Path) -> Callable[[Path], frozen.Result]:
    """Load one validated CUDA graph; let dependency and bundle failures raise."""
    import cv2 as cv
    import numpy as np
    import torch

    net, version, device = frozen._open(Path(model_dir))
    failed = Event()

    def predict(path: Path) -> frozen.Result:
        if failed.is_set():
            raise E03Unavailable('E03 CUDA engine has been retired')

        try:
            rgb = frozen._decode(Path(path), cv, np)
        except (OSError, TypeError, ValueError, cv.error):
            rgb = None
        if rgb is None:
            result = frozen._result(version, 'uncertain', reason='unscorable')
        else:
            try:
                with torch.inference_mode():
                    frozen._assert_cuda(net, device)
                    residual, delta, _ = frozen._motion(torch.as_tensor(rgb, device=device).unsqueeze(0), torch)
                    enough, global_motion = (bool(x.item()) for x in frozen._motion_quality(residual))
                    if not enough or global_motion:
                        result = frozen._decide(version, (0, 0), enough, global_motion)
                    else:
                        inputs = frozen._stack_motion(residual, delta, torch)
                        views = frozen._views(net, inputs, torch)[0].tolist()
                        result = frozen._decide(version, views)
                if result['label'] is None:
                    raise RuntimeError('invalid E03 model scores')
            except RuntimeError as exc:
                failed.set()
                raise E03Unavailable('E03 CUDA inference failed') from exc

        if failed.is_set():
            raise E03Unavailable('E03 CUDA engine has been retired')
        return result

    return predict
