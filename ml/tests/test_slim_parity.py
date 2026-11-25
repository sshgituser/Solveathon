import os
from pathlib import Path

import pytest


def test_full_slim_matches_upstream_dense_tokens_with_reference_attention(monkeypatch):
    import importlib
    import torch
    from safetensors.torch import load_file
    from ml.models import base_encoder

    base, source = os.environ.get('SERIQ_SLIM_BASE'), os.environ.get('SERIQ_SLIM_SRC')
    if not base or not source:
        pytest.skip('set SERIQ_SLIM_BASE and SERIQ_SLIM_SRC to verify the upstream equations')
    monkeypatch.syspath_prepend(source)
    monkeypatch.setenv('XFORMERS_DISABLED', '1')
    attention = importlib.import_module('slim.layers.attention')
    upstream = importlib.import_module('slim.hub').SLiMEncoder.from_config({})
    upstream.load_state_dict(load_file(base), strict=True)

    def dense(q, k, v, attn_bias=None):
        assert attn_bias is None
        q,k,v = (x.transpose(1,2) for x in (q,k,v))
        return ((q @ k.transpose(-2,-1) / q.size(-1)**.5).softmax(-1) @ v).transpose(1,2).contiguous()
    monkeypatch.setattr(attention, 'XFORMERS_AVAILABLE', True)
    monkeypatch.setattr(attention, 'unbind', torch.unbind, raising=False)
    monkeypatch.setattr(attention, 'memory_efficient_attention', dense, raising=False)
    dev = os.environ.get('SERIQ_CHECK_DEVICE', 'cpu')
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(42)
    points = (torch.randn(2,64,25,3,generator=gen) * .2).to(dev)
    mask = torch.ones((2,64,25), dtype=torch.bool, device=dev)
    mask[:,8:32,12:20] = False
    x = points.masked_fill(~mask[...,None], 0)
    p_mask = ~mask.reshape(2,8,8,25).any(dim=2).flatten(1,2)
    reference = upstream.eval().to(dev)
    actual = base_encoder(Path(base)).eval().to(dev)
    with torch.inference_mode():
        result = reference.forward_features(x.permute(0,3,1,2)[...,None], p_mask)
        tokens = actual.last(actual.prefix(points,mask))
    expected = torch.cat((result['x_norm_clstoken'][:,None],result['x_norm_regtokens'],result['x_norm_patchtokens']),dim=1)
    assert tuple(tokens.shape) == (2,205,256)
    torch.testing.assert_close(tokens, expected, rtol=5e-5, atol=5e-5)
