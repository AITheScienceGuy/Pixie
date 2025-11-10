#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def build_rope_cache(L: int, dim: int, base: float, device, dtype):
    # returns cos,sin with shape [L, dim/2]
    assert dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(L, device=device, dtype=dtype)
    freqs = torch.einsum('l,d->ld', t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def rot_half(x):
    # (..., 2m) -> rotate pairs (x0,x1),(x2,x3)...
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)

def apply_rope(x, cos, sin):
    # x: [..., d] (even), cos/sin: [..., d/2]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos + rot_half(x) * sin

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class AdaRMSNorm(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int, eps: float = 1e-6, zero_init: bool = True):
        super().__init__()
        self.core = RMSNorm(embed_dim, eps)
        self.to_hidden = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * embed_dim)
        )
        self.to_gamma = nn.Linear(4 * embed_dim, embed_dim)
        self.to_beta  = nn.Linear(4 * embed_dim, embed_dim)
        if zero_init:  
            nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
            nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None):
        if cond is None:
            return self.core(x)
        B, T, P, E = x.shape
        h = self.to_hidden(cond)          # [B,T,4E]
        gamma = self.to_gamma(h)          # [B,T,E]
        beta  = self.to_beta(h)           # [B,T,E]
        gamma = gamma.unsqueeze(2)        # [B,T,1,E]
        beta  = beta.unsqueeze(2)
        y = self.core(x)
        return y * (1.0 + gamma) + beta

class SwiGLUFFN(nn.Module):
    def __init__(self, embedding_dim, inter_dim):
        super().__init__()
        h = math.floor(2 * inter_dim / 3)
        self.w_v = nn.Linear(embedding_dim, h)
        self.w_g = nn.Linear(embedding_dim, h)
        self.w_o = nn.Linear(h, embedding_dim)

    def forward(self, x, conditioning=None):
        v = F.silu(self.w_v(x)) # [B, T, P, h]
        g = self.w_g(x) # [B, T, P, h]
        out = self.w_o(v * g) # [B, T, P, E]
        return out # [B, T, P, E]

class SpatialAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, inter_dim, Hp=None, Wp=None, rope_base=10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.inter_dim = inter_dim

        assert inter_dim % num_heads == 0, "inter_dim must be divisible by num_heads"
        self.head_dim = inter_dim // num_heads

        assert self.head_dim % 2 == 0, "head_dim must be even for 2D RoPE"
        self.Dhx = self.head_dim // 2
        self.Dhy = self.head_dim - self.Dhx
        
        self.q_weight = nn.Linear(self.embedding_dim, self.inter_dim)
        self.k_weight = nn.Linear(self.embedding_dim, self.inter_dim)
        self.v_weight = nn.Linear(self.embedding_dim, self.inter_dim)
        self.proj_out = nn.Linear(inter_dim, embedding_dim, bias=False)
        self.softmax_scale = (self.head_dim) ** -0.5

        self.rope_base = rope_base
        self.Hp, self.Wp = Hp, Wp 
        if (Hp is not None) and (Wp is not None):
            yy, xx = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing='ij')
            P = Hp * Wp
            self.register_buffer("patch_y_index", yy.reshape(P), persistent=False)
            self.register_buffer("patch_x_index", xx.reshape(P), persistent=False)
        self._rope_xy_cache = {}

    def _rope_xy(self, device, dtype):
        key = (device, dtype)
        if key not in self._rope_xy_cache:
            cos_x, sin_x = build_rope_cache(self.Wp, self.Dhx, self.rope_base, device, dtype)
            cos_y, sin_y = build_rope_cache(self.Hp, self.Dhy, self.rope_base, device, dtype)
            self._rope_xy_cache[key] = (cos_x, sin_x, cos_y, sin_y)
        return self._rope_xy_cache[key]

    def forward(self, x):
        B, T, P, E = x.shape
        
        q = self.q_weight(x).view(B, T, P, self.num_heads, self.head_dim)
        k = self.k_weight(x).view(B, T, P, self.num_heads, self.head_dim)
        v = self.v_weight(x).view(B, T, P, self.num_heads, self.head_dim)

        cos_x, sin_x, cos_y, sin_y = self._rope_xy(x.device, x.dtype)
        cosx = cos_x[self.patch_x_index].view(1,1,P,1,self.Dhx//2)
        sinx = sin_x[self.patch_x_index].view(1,1,P,1,self.Dhx//2)
        cosy = cos_y[self.patch_y_index].view(1,1,P,1,self.Dhy//2)
        siny = sin_y[self.patch_y_index].view(1,1,P,1,self.Dhy//2)

        qx, qy = q[...,:self.Dhx], q[...,self.Dhx:]
        kx, ky = k[...,:self.Dhx], k[...,self.Dhx:]
        qx = apply_rope(qx, cosx, sinx);  kx = apply_rope(kx, cosx, sinx)
        qy = apply_rope(qy, cosy, siny);  ky = apply_rope(ky, cosy, siny)
        q = torch.cat([qx,qy], dim=-1);   k = torch.cat([kx,ky], dim=-1)

        scores = torch.einsum('btihd,btjhd->bthij', q, k) * self.softmax_scale
        scores = scores.softmax(dim=-1)

        y = torch.einsum('bthij,btjhd->btihd', scores, v).reshape(B, T, P, self.inter_dim)
        output = self.proj_out(y)

        return output

class TemporalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, inter_dim, causal=True, rope_base=10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.inter_dim = inter_dim

        assert inter_dim % num_heads == 0, "inter_dim must be divisible by num_heads"
        self.head_dim = inter_dim // num_heads
        
        self.q_weight = nn.Linear(self.embedding_dim, self.inter_dim)
        self.k_weight = nn.Linear(self.embedding_dim, self.inter_dim)
        self.v_weight = nn.Linear(self.embedding_dim, self.inter_dim)
        self.proj_out = nn.Linear(inter_dim, embedding_dim, bias=False)
        self.softmax_scale = (self.head_dim) ** -0.5
        self.causal = causal

        self.rope_base = rope_base
        self._rope_t_cache = {}

    def _rope_t(self, T, device, dtype):
        key = (T, device, dtype)
        if key not in self._rope_t_cache:
            cos_t, sin_t = build_rope_cache(T, self.head_dim, self.rope_base, device, dtype)
            self._rope_t_cache[key] = (cos_t, sin_t)  # [T, Dh/2]
        return self._rope_t_cache[key]

    def forward(self, x):
        B, T, P, E = x.shape
        
        q = self.q_weight(x).view(B, T, P, self.num_heads, self.head_dim)
        k = self.k_weight(x).view(B, T, P, self.num_heads, self.head_dim)
        v = self.v_weight(x).view(B, T, P, self.num_heads, self.head_dim)

        cos_t, sin_t = self._rope_t(T, x.device, x.dtype)
        cos_t = cos_t.view(1,T,1,1,self.head_dim//2)   # broadcast to [B,T,P,H,Dh/2]
        sin_t = sin_t.view(1,T,1,1,self.head_dim//2)
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)

        scores = torch.einsum('btphd,buphd->bphtu', q, k) * self.softmax_scale  # [B,P,T,T]

        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, -torch.inf)
        
        scores = scores.softmax(dim=-1)

        y = torch.einsum('bphtu,buphd->btphd', scores, v).reshape(B, T, P, self.inter_dim)
        output = self.proj_out(y)

        return output

class STBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, inter_dim, Hp, Wp, causal=True, rope_base=10000.0, action_dim: int = None):
        super().__init__()
        self.spatial_attention = SpatialAttention(embedding_dim, num_heads, inter_dim, Hp=Hp, Wp=Wp, rope_base=10000.0)
        self.temporal_attention = TemporalAttention(embedding_dim, num_heads, inter_dim, causal, rope_base=10000.0)
        self.ffn = SwiGLUFFN(embedding_dim, inter_dim)
        if action_dim is None:
            self.spatialNorm  = RMSNorm(embedding_dim)
            self.temporalNorm = RMSNorm(embedding_dim)
            self.expertNorm   = RMSNorm(embedding_dim)
            self._cond_optional = True
        else:
            self.spatialNorm  = AdaRMSNorm(embedding_dim, cond_dim=action_dim)
            self.temporalNorm = AdaRMSNorm(embedding_dim, cond_dim=action_dim)
            self.expertNorm   = AdaRMSNorm(embedding_dim, cond_dim=action_dim)
            self._cond_optional = False

    def forward(self, x, cond=None):
        if self._cond_optional:
            x = x + self.spatial_attention(self.spatialNorm(x))
            x = x + self.temporal_attention(self.temporalNorm(x))
            x = x + self.ffn(self.expertNorm(x))
        else:
            x = x + self.spatial_attention(self.spatialNorm(x, cond))
            x = x + self.temporal_attention(self.temporalNorm(x, cond))
            x = x + self.ffn(self.expertNorm(x, cond))
        return x

class STTransformer(nn.Module):
    def __init__(self, num_blocks, embedding_dim, num_heads, inter_dim, Hp, Wp, causal=True, rope_base=10000.0, action_dim: int = None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_id in range(num_blocks):
            self.layers.append(STBlock(embedding_dim, num_heads, inter_dim, Hp=Hp, Wp=Wp, causal=causal, rope_base=rope_base, action_dim=action_dim))

    def forward(self, x, cond=None):
        for layer in self.layers:
            x = layer(x, cond)
        return x
