#!/usr/bin/env python

import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embedding_dim=128, input_channels=3):
        super().__init__()
        H, W = frame_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp

        self.proj = nn.Conv2d(input_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = rearrange(frames, 'b t c h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) e hp wp -> b t (hp wp) e', b=B, t=T)

        return x
