import os
from pathlib import Path

import numpy as np
import pytest


def checkpoint():
    name = os.environ.get('SERIQ_SLIM_BASE')
    if not name:
        pytest.skip('set SERIQ_SLIM_BASE to exercise pinned encoder')
    return Path(name)


def test_slim_strict_load_and_adapted_last_block():
    import torch
    from ml.models import ModelB, base_encoder, trainable
    torch.set_num_threads(1)
    m = ModelB(base_encoder(checkpoint()))
    params = trainable(m, 'B')
    assert len(m.encoder.blocks) == 8
    assert sum(k.startswith('encoder.blocks.7.') for k in params) == 12
    assert all(k.startswith(('encoder.blocks.7.', 'head.')) for k in params)
    xyz = torch.randn(2, 64, 25, 3)
    mask = torch.ones_like(xyz[..., 0], dtype=torch.bool)
    old = params['encoder.blocks.7.attn.qkv.weight'].detach().clone()
    opt = torch.optim.SGD(params.values(), lr=.01)
    y = m(xyz, mask, None, None)
    assert y.shape == (2,)
    y.square().mean().backward()
    opt.step()
    assert not torch.equal(params['encoder.blocks.7.attn.qkv.weight'], old)
    assert m.encoder.blocks[0].attn.qkv.weight.grad is None


def test_omitted_coordinates_do_not_affect_predictions():
    import torch
    from ml.models import ModelA, ModelB, base_encoder
    torch.set_num_threads(1)
    coords = torch.randn(1, 64, 25, 3)
    mask = torch.zeros_like(coords[...,0], dtype=torch.bool)
    mask[:,:,4:7] = True
    altered = coords.masked_fill(~mask[...,None], 1000.0)
    angles = torch.randn(1,64,2)
    valid = torch.ones_like(angles, dtype=torch.bool)
    for model in (ModelA(), ModelB(base_encoder(checkpoint()))):
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(coords, mask, angles, valid), model(altered, mask, angles, valid), atol=0, rtol=0)
