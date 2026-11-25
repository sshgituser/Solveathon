"""SerIQ classifiers. SLiM inference equations adapted from KAIST-VICLab/SLiM.

SLiM and original DINOv2 code: Apache-2.0, portions Copyright (c) Meta
Platforms, Inc. and affiliates. Changes: dense single-person inference path,
PyTorch SDPA instead of xFormers, and a binary classification head.
"""
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ml.data import FOREARM_NTU, FOREARM_POLICY, POSE_POLICIES, STRICT_POLICY, sha256

SLIM_REVISION = 'b934cfdbce135479b1df5abe6d1495fcee05e4f5'
SLIM_CODE = '3aaab213883a1e8b06d08fcfa854da50f23889c4'
SLIM_SHA = 'c25c70db9ca5c1ca3715b80acbd21851fa04f4b3178e4c82f96d45492774b5bf'


class ModelA(nn.Module):
    """A shared-arm TCN, with masked range/peak summaries and symmetric pooling."""
    def __init__(self, pose_policy=STRICT_POLICY):
        super().__init__()
        if pose_policy not in POSE_POLICIES:
            raise ValueError('unsupported pose policy')
        self.pose_policy = pose_policy
        self.conv = nn.Sequential(nn.Conv1d(5 if pose_policy == FOREARM_POLICY else 10, 16, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(16, 16, 5, padding=4, dilation=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(36, 16), nn.GELU(), nn.Dropout(.15), nn.Linear(16, 1))

    def forward(self, coords, mask, angles, angle_mask):
        if self.pose_policy == FOREARM_POLICY:
            xy = coords[:, :, FOREARM_NTU, :2].flatten(2)
            valid = mask[:, :, FOREARM_NTU].all(dim=-1, keepdim=True)
            xy = xy.masked_fill(~valid, 0)
            features = torch.cat((xy, valid.to(xy.dtype)), dim=-1).transpose(1, 2)
            encoded = self.conv(features).transpose(1, 2)
            count = valid.sum(dim=1)
            mean = (encoded * valid).sum(dim=1) / count.clamp_min(1)
            peak = encoded.masked_fill(~valid, -1e4).amax(dim=1).masked_fill(count == 0, 0)
            ranges = (xy.masked_fill(~valid, -1e4).amax(dim=1) -
                      xy.masked_fill(~valid, 1e4).amin(dim=1)).masked_fill(count == 0, 0)
            return self.head(torch.cat((mean, peak, ranges), dim=1)).squeeze(-1)
        ids = [4,5,6,8,9,10]
        arm = coords[:, :, ids].reshape(coords.shape[0], 64, 2, 3, 3)
        m = mask[:, :, ids].reshape(coords.shape[0], 64, 2, 3)
        upper = (arm[..., 1, :] - arm[..., 0, :]) * (m[..., 1] & m[..., 0])[..., None]
        lower = (arm[..., 2, :] - arm[..., 1, :]) * (m[..., 2] & m[..., 1])[..., None]
        upper = torch.stack((upper[:,:,0], upper[:,:,1] * upper.new_tensor([-1,1,1])), dim=2)
        lower = torch.stack((lower[:,:,0], lower[:,:,1] * lower.new_tensor([-1,1,1])), dim=2)
        features = torch.cat((upper, lower, angles.masked_fill(~angle_mask, 0)[...,None], m.float()), dim=-1)
        features = features.permute(0,2,3,1).reshape(-1, 10, 64)
        encoded = self.conv(features).reshape(coords.shape[0], 2, 16, 64)
        valid = angle_mask.transpose(1,2).unsqueeze(2)
        count = valid.sum(dim=-1)
        mean = (encoded * valid).sum(dim=-1) / count.clamp_min(1)
        peak = encoded.masked_fill(~valid, -1e4).amax(dim=-1)
        peak = peak.masked_fill(count == 0, 0)
        min_angle = angles.masked_fill(~angle_mask, 1).amin(dim=1)
        max_angle = angles.masked_fill(~angle_mask, -1).amax(dim=1)
        # Elbow extension is low cosine; consider both arms without privileging a side.
        active = count.squeeze(-1) > 0
        extension = -min_angle.masked_fill(~active, 1).amin(dim=1)
        flexion = max_angle.masked_fill(~active, -1).amax(dim=1)
        span = (max_angle - min_angle).masked_fill(~active, 0).amax(dim=1)
        stability = (min_angle.masked_fill(~active, 0).sum(dim=1) / active.sum(dim=1).clamp_min(1))
        summary = torch.stack((extension, flexion, span, stability), dim=1)
        return self.head(torch.cat((mean.amax(dim=1), peak.amax(dim=1), summary), dim=1)).squeeze(-1)


def rotated(x, cos, sin):
    a, b = x.float().chunk(2, dim=-1)
    return (x.float() * cos + torch.cat((-b,a), dim=-1) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 768)
        self.proj = nn.Linear(256, 256)
        inv = 1. / (10000 ** (torch.arange(0, 32, 2).float() / 32.))
        self.register_buffer('inv_freq', inv, persistent=False)

    def forward(self, x):
        b, n, c = x.shape
        q,k,v = self.qkv(x).reshape(b,n,3,8,32).unbind(dim=2)
        frequencies = torch.outer(torch.arange(8, device=x.device, dtype=torch.float32), self.inv_freq)
        freqs = torch.cat((frequencies, frequencies), dim=1).repeat_interleave(25, dim=0)[None,:,None,:]
        cos, sin = freqs.cos().to(x.dtype).float(), freqs.sin().to(x.dtype).float()
        q = torch.cat((q[:,:5], rotated(q[:,5:], cos, sin)), dim=1)
        k = torch.cat((k[:,:5], rotated(k[:,5:], cos, sin)), dim=1)
        attn = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
        return self.proj(attn.transpose(1,2).reshape(b,n,c))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w12 = nn.Linear(256, 1376)
        self.w3 = nn.Linear(688, 256)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256, eps=1e-6)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(256, eps=1e-6)
        self.mlp = SwiGLU()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SLiM(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 256, (8, 1), stride=(8, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, 256))
        self.pos_embed = nn.Parameter(torch.zeros(1, 26, 256))
        self.mask_token = nn.Parameter(torch.zeros(1, 256))
        self.blocks = nn.ModuleList(Block() for _ in range(8))
        self.norm = nn.LayerNorm(256, eps=1e-6)

    def embed(self, coords, mask):
        b = coords.shape[0]
        x = self.patch_embed.proj(coords.masked_fill(~mask[...,None], 0).permute(0,3,1,2))
        x = x.permute(0,2,3,1)
        visible = mask.reshape(b,8,8,25).any(dim=2)
        x = torch.where(visible[...,None], x, self.mask_token.to(x.dtype))
        x = (x + self.pos_embed[:,None,1:]).flatten(1,2)
        return torch.cat(((self.cls_token+self.pos_embed[:,:1]).expand(b,-1,-1),
                          self.register_tokens.expand(b,-1,-1), x), dim=1)

    def prefix(self, coords, mask):
        x = self.embed(coords, mask)
        for block in list(self.blocks)[:-1]:
            x = block(x)
        return x

    def last(self, x):
        return self.norm(self.blocks[-1](x))


def base_encoder(path):
    if sha256(path) != SLIM_SHA:
        raise ValueError('official NTU60 XSub SLiM safetensors SHA-256 mismatch')
    model = SLiM()
    model.load_state_dict(load_file(str(path)), strict=True)
    return model


class ModelB(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(256, 1))

    def from_prefix(self, data):
        return self.head(self.encoder.last(data)[:,0]).squeeze(-1)

    def forward(self, coords, mask, angles, angle_mask):
        return self.from_prefix(self.encoder.prefix(coords, mask))


def trainable(model, kind):
    if kind not in ('A','B'):
        raise ValueError('expected A or B')
    result = {}
    for name, value in model.named_parameters():
        value.requires_grad_(kind == 'A' or name.startswith(('encoder.blocks.7.', 'head.')))
        if value.requires_grad:
            result[name] = value
    return result
