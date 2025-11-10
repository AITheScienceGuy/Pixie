#!/usr/bin/env python

import torch
import torch.nn as nn
from einops import rearrange


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, latent_dim=5, num_bins=4):
        super().__init__()
        self.num_bins = num_bins # D
        self.levels_np = torch.tensor(latent_dim * [num_bins])
        self.codebook_size = num_bins**latent_dim # L^D
        self.register_buffer('basis', (num_bins**torch.arange(latent_dim, dtype=torch.long)))

    def scale_and_shift(self, z):
        return 0.5 * (z + 1) * (self.num_bins - 1)

    def unscale_and_unshift(self, z):
        return 2 * z / (self.num_bins - 1) - 1

    def forward(self, z):
        tanh_z = torch.tanh(z)
        bounded_z = self.scale_and_shift(tanh_z)
        rounded_z = torch.round(bounded_z)
        quantized_z = bounded_z + (rounded_z - bounded_z).detach()
        quantized_z = self.unscale_and_unshift(quantized_z)

        return quantized_z

    def get_codebook_usage(self, quantized_z):
        unique_bins = torch.unique(quantized_z).shape[0]
        
        return unique_bins / self.num_bins

    def get_indices_from_latents(self, latents, dim=-1):
        digits = torch.round(self.scale_and_shift(latents)).clamp(0, self.num_bins-1)
        indices = torch.sum(digits * self.basis.to(latents.device), dim=dim).long() 
        
        return indices

    def get_latents_from_indices(self, indices, dim=-1):
        digits = (indices.unsqueeze(-1) // self.basis) % self.num_bins
        latents = self.unscale_and_unshift(digits) 
        
        return latents
