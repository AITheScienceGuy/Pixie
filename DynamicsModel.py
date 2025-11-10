#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from STTransformer import STTransformer
from einops import rearrange
from FSQ import FiniteScalarQuantizer

class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(128,128), patch_size=8, embedding_dim=128,
                 num_blocks=4, num_heads=6, inter_dim=256,
                 latent_dim=5, num_bins=4, action_dim=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bins = num_bins

        sub_dim = math.ceil(embedding_dim / latent_dim)
        self.latent_embedders = nn.ModuleList(
            [nn.Embedding(num_bins, sub_dim) for _ in range(latent_dim)]
        )
        embed_cat_dim = sub_dim * latent_dim
        self.embedding_proj = nn.Linear(embed_cat_dim, embedding_dim)

        H, W = frame_size
        Hp, Wp = H // patch_size, W // patch_size
        self.transformer = STTransformer(num_blocks, embedding_dim, num_heads, inter_dim,
                                         Hp, Wp, causal=True, action_dim=action_dim)

        self.output_head = nn.Linear(embedding_dim, latent_dim * num_bins)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embedding_dim) * 0.02)

    def _embed_indices(self, idx): # idx: [B,T,P,L]
        parts = [self.latent_embedders[l](idx[..., l]) for l in range(self.latent_dim)]
        x = torch.cat(parts, dim=-1)
        x = self.embedding_proj(x)
        return x

    def forward(self, discrete_indices, actions=None, context_mask_ratio: float = 0.0):
        """Forward pass for the dynamics model.

        Args:
            discrete_indices: [B, T, P, L] integer digits in [0, num_bins-1]
            actions:          [B, T-1, action_dim] or None
        Returns:
            logits: [B, T-1, P, L, num_bins]
            z_tgt:  [B, T-1, P, L]
        """
        # Context / targets in token space
        z_ctx = discrete_indices[:, :-1]   # [B, T-1, P, L]
        z_tgt = discrete_indices[:, 1:]    # [B, T-1, P, L]
        B, Tm1, P, _ = z_ctx.shape

        # Align actions with context length if provided
        if actions is not None:
            if actions.size(1) != Tm1:
                # If actions are too long, truncate; if too short, pad last action
                if actions.size(1) > Tm1:
                    a_ctx = actions[:, :Tm1]
                else:
                    pad = actions[:, -1:].expand(-1, Tm1 - actions.size(1), -1)
                    a_ctx = torch.cat([actions, pad], dim=1)
            else:
                a_ctx = actions
        else:
            a_ctx = None

        # Embed discrete digits to continuous tokens
        x = self._embed_indices(z_ctx)  # [B, T-1, P, E]

        # Optional context masking
        if self.training and context_mask_ratio > 0.0:
            mask = (torch.rand(B, Tm1, P, device=x.device) < context_mask_ratio)
            mask[:, 0] = False  # always keep the first timestep
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        # Spatiotemporal transformer with optional conditioning
        h = self.transformer(x, cond=a_ctx)  # [B, T-1, P, E]

        # Project to factorized logits, using reshape (not view) for safety
        logits = self.output_head(h).reshape(B, Tm1, P, self.latent_dim, self.num_bins)
        return logits, z_tgt
