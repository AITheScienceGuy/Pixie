#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from STTransformer import STTransformer, RMSNorm
from PatchEmbedding import PatchEmbedding
from FSQ import FiniteScalarQuantizer

class EmbeddingsToPixels(nn.Module):
    def __init__(self, embedding_dim, out_chans, patch_size):
        super().__init__()
        self.deproj = nn.ConvTranspose2d(
            in_channels=embedding_dim, out_channels=out_chans,
            kernel_size=patch_size, stride=patch_size
        )
    def forward(self, tokens, Hp, Wp, B, T):  
        x = tokens.reshape(B*T, Hp*Wp, -1)  
        x = x.transpose(1, 2).reshape(B*T, -1, Hp, Wp)   
        x = self.deproj(x) 
        x = x.reshape(B, T, x.size(1), x.size(2), x.size(3)) 
        return x

class VideoTokenizerEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embedding_dim=128, input_channels=3, num_blocks=4, num_heads=6, inter_dim=256, causal=True, rope_base=10000.0, latent_dim=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(frame_size, patch_size, embedding_dim, input_channels)
        Hp, Wp = self.patch_embedding.Hp, self.patch_embedding.Wp
        self.transformer = STTransformer(num_blocks, embedding_dim, num_heads, inter_dim, Hp, Wp, causal, rope_base)
        self.latent_head = nn.Sequential(
            RMSNorm(embedding_dim),
            nn.Linear(embedding_dim, latent_dim)
        )

    def forward(self, frames):
        patch_embeddings = self.patch_embedding(frames)
        output = self.transformer(patch_embeddings)
        latents = self.latent_head(output)

        return latents

class VideoTokenizerDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embedding_dim=128, input_channels=3, num_blocks=4, num_heads=6, inter_dim=256, causal=True, rope_base=10000.0, latent_dim=5):
        super().__init__()
        self.patch_size = patch_size
        H, W = frame_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.latent_to_embedding = nn.Linear(latent_dim, embedding_dim)
        self.transformer = STTransformer(num_blocks, embedding_dim, num_heads, inter_dim, Hp=self.Hp, Wp=self.Wp, causal=causal, rope_base=rope_base)
        self.embedding_to_pixels = EmbeddingsToPixels(embedding_dim, input_channels, patch_size)

    def forward(self, latents):
        B, T, P, _ = latents.shape
        embeddings = self.latent_to_embedding(latents)
        output = self.transformer(embeddings)
        pixels = self.embedding_to_pixels(output, self.Hp, self.Wp, B, T)

        return pixels

class VideoTokenizer(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8,
                 embedding_dim=128, input_channels=3,
                 num_blocks=4, num_heads=6, inter_dim=256,
                 causal=False, rope_base=10_000.0,
                 latent_dim=5, num_bins=4):
        super().__init__()
        self.encoder = VideoTokenizerEncoder(
            frame_size=frame_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            inter_dim=inter_dim,
            causal=causal,
            rope_base=rope_base,
            latent_dim=latent_dim,
        )
        self.quantizer = FiniteScalarQuantizer(latent_dim=latent_dim, num_bins=num_bins)
        self.decoder = VideoTokenizerDecoder(
            frame_size=frame_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            inter_dim=inter_dim,
            causal=causal,
            rope_base=rope_base,
            latent_dim=latent_dim,
        )

    def forward(self, frames):
        latents = self.encoder(frames)
        quantized_latents = self.quantizer(latents)
        x_hat = self.decoder(quantized_latents)
        return x_hat

    def encode(self, frames):
        latents = self.encoder(frames)
        quantized_latents = self.quantizer(latents)
        return quantized_latents

    def decode(self, latents):
        x_hat = self.decoder(latents)
        return x_hat

    # NEW: factorized discrete interface for the dynamics model
    def encode_to_indices(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Returns digit indices in [0, num_bins-1] per latent dimension.

        Shape:
            frames: [B, T, C, H, W]
            return: [B, T, P, latent_dim]
        """
        with torch.no_grad():
            latents = self.encoder(frames)                 # [B, T, P, latent_dim]
            quantized = self.quantizer(latents)           # same shape, but snapped to scalar levels

            # Map quantized latents back to digit bins 0..num_bins-1
            digits = torch.round(
                self.quantizer.scale_and_shift(quantized)
            ).clamp(0, self.quantizer.num_bins - 1)

        return digits.long()

    def decode_from_indices(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Takes digit indices [0, num_bins-1] and decodes back to RGB frames.

        Shape:
            digits: [B, T, P, latent_dim]
            return: [B, T, C, H, W]
        """
        # Map digits back to latent space (inverse of scale_and_shift)
        latents = self.quantizer.unscale_and_unshift(digits.float())
        x_hat = self.decoder(latents)
        return x_hat
