#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from STTransformer import STTransformer, RMSNorm
from PatchEmbedding import PatchEmbedding
from FSQ import FiniteScalarQuantizer
from einops import rearrange

class ActionEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embedding_dim=128, input_channels=3, num_blocks=4, num_heads=6, inter_dim=256, action_dim=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(frame_size, patch_size, embedding_dim, input_channels)
        self.num_patches = self.patch_embedding.num_patches
        Hp, Wp = self.patch_embedding.Hp, self.patch_embedding.Wp
        self.transformer = STTransformer(num_blocks, embedding_dim, num_heads, inter_dim, Hp, Wp, causal=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.attention_pool = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.action_head = nn.Sequential(
            RMSNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, 4 * action_dim),
            nn.GELU(),
            nn.Linear(4 * action_dim, action_dim)
        )

    def forward(self, frames):
        batch_size, seq_len, _, _, _ = frames.shape

        embeddings = self.patch_embedding(frames)
        transformed = self.transformer(embeddings)
        reshaped_transformed = transformed.view(batch_size * seq_len, self.num_patches, -1)
        query = self.pool_query.expand(batch_size * seq_len, -1, -1)
        pooled, _ = self.attention_pool(query=query, key=reshaped_transformed, value=reshaped_transformed)
        pooled = pooled.squeeze(1).view(batch_size, seq_len, -1)

        pooled_current = pooled[:, :-1]
        pooled_next = pooled[:, 1:]

        combined_features = torch.cat([pooled_current, pooled_next], dim=2)
        actions = self.action_head(combined_features)

        return actions

class ActionDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embedding_dim=128, input_channels=3, num_blocks=4, num_heads=6, inter_dim=256, conditioning_dim=5):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(frame_size, patch_size, embedding_dim, input_channels)
        Hp, Wp = self.patch_embedding.Hp, self.patch_embedding.Wp
        self.transformer = STTransformer(num_blocks, embedding_dim, num_heads, inter_dim, Hp, Wp, causal=True, action_dim=conditioning_dim)
        
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, self.input_channels * patch_size * patch_size)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_dim))

    def forward(self, frames, actions):
        B, T, C, H, W = frames.shape
        frames = frames[:, :-1]
        embeddings = self.patch_embedding(frames)
        
        assert actions.shape[:2] == (B, T-1), "actions must be [B, T-1, A]"
        transformed = self.transformer(embeddings, cond=actions)
        patches = self.frame_head(transformed)
        
        patches = rearrange(patches, 'b t p (c p1 p2) -> b t c p p1 p2', c=self.input_channels, p1=self.patch_size, p2=self.patch_size)
        pred_frames = rearrange(patches, 'b t c (h w) p1 p2 -> b t c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size)
        return pred_frames

class LatentActionModel(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embedding_dim=128, input_channels=3, num_blocks=4, num_heads=6, inter_dim=256, action_dim=5, conditioning_dim=5):
        super().__init__()
        self.action_encoder = ActionEncoder(frame_size, patch_size, embedding_dim, input_channels, num_blocks, num_heads, inter_dim, action_dim)
        self.quantizer = FiniteScalarQuantizer(action_dim, num_bins=3)
        self.action_decoder = ActionDecoder(frame_size, patch_size, embedding_dim, input_channels, num_blocks, num_heads, inter_dim, conditioning_dim)

    def forward(self, frames, return_latents=False):
        action_latents = self.action_encoder(frames)
        quantized_actions = self.quantizer(action_latents)
        out = self.action_decoder(frames, quantized_actions)

        if return_latents:
            return out, action_latents, quantized_actions
        return out

    def encode(self, frames):
        action_latents = self.action_encoder(frames)
        quantized_actions = self.quantizer(action_latents)
        return quantized_actions
