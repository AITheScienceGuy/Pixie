#!/usr/bin/env python

import os
import h5py
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

SUPPORTED_FILES = [
    "picodoom_frames.h5",
    "pole_position_frames.h5",
    "pong_frames.h5",
    "sonic_frames.h5",
    "zelda_frames.h5",
]

class H5GameClipsDataset(Dataset):
    """
    Loads fixed-length clips from multiple HDF5 gameplay files in ./data.
    Expects a dataset key 'frames' unless overridden.
    Returns a tensor of shape [T, C, H, W] in [0,1].
    All items are resized to `target_size=(H, W)` if provided to guarantee
    identical shapes for batching.
    """
    def __init__(
        self,
        data_dir: str = "./data",
        dataset_key: str = "frames",
        clip_len: int = 16,
        frame_stride: int = 1,          # sample every k-th frame inside a clip
        step_between_clips: int = 8,    # how far to move the clip window
        target_size: Optional[Tuple[int, int]] = None,  # (H, W) to enforce
        files: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_key = dataset_key
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.step_between_clips = step_between_clips
        self.target_size = target_size

        # Collect files
        self.files = [os.path.join(data_dir, f) for f in (files or SUPPORTED_FILES)]
        self.files = [p for p in self.files if os.path.exists(p)]
        if not self.files:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir} matching {SUPPORTED_FILES}")

        # Per-worker file handles (opened lazily)
        self._handles: Dict[str, h5py.File] = {}

        # Build clip index: (path, seq_id, start, T_total)
        self.index: List[Tuple[str, int, int, int]] = []
        for path in self.files:
            with h5py.File(path, "r") as f:
                if self.dataset_key not in f:
                    # fallback: use first dataset present
                    key = next((k for k, v in f.items() if isinstance(v, h5py.Dataset)), None)
                    if key is None:
                        raise KeyError(f"No dataset found in {path}")
                    ds = f[key]
                else:
                    ds = f[self.dataset_key]

                rank = ds.ndim
                shape = ds.shape
                # Normalize to sequences: treat [T,...] as N=1
                if rank == 4:  # [T,H,W,C] or [T,C,H,W]
                    N, T_total = 1, shape[0]
                elif rank == 5:  # [N,T,H,W,C] or [N,T,C,H,W]
                    N, T_total = shape[0], shape[1]
                else:
                    raise ValueError(f"Unsupported dataset rank {rank} in {path} with shape {shape}")

                # Frames sampled per clip with stride
                eff_clip = 1 + (self.clip_len - 1) * self.frame_stride
                if T_total < eff_clip:
                    continue

                # Sliding windows
                for n in range(N):
                    for start in range(0, T_total - eff_clip + 1, self.step_between_clips):
                        self.index.append((path, n, start, T_total))

        if not self.index:
            raise RuntimeError("No clips could be constructed with given parameters.")

    def __len__(self):
        return len(self.index)

    def _get_handle(self, path: str) -> h5py.File:
        # one handle per worker-process for each file
        h = self._handles.get(path)
        if h is None or not getattr(h, 'id', None):
            # swmr=True enables concurrent reads; good practice for multi-worker
            h = h5py.File(path, "r", swmr=True)
            self._handles[path] = h
        return h

    @staticmethod
    def _to_TCHW(x: np.ndarray) -> np.ndarray:
        # x: numpy array in [T,H,W,C] or [T,C,H,W]; return [T,C,H,W]
        if x.ndim != 4:
            raise ValueError(f"Expected 4D array per clip, got {x.shape}")
        # Heuristic: channels-last if last dim is 3/1; else already channels-first
        if x.shape[-1] in (1, 3):
            x = x.transpose(0, 3, 1, 2)  # THWC -> TCHW
        return x

    @staticmethod
    def _resize_tchw(tchw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        # tchw: [T,C,H,W] float in [0,1]; bilinear per-frame
        T, C, H, W = tchw.shape
        tchw = tchw.view(T * 1, C, H, W)  # merge batch=1
        tchw = F.interpolate(tchw, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return tchw.view(T, C, out_h, out_w)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, seq_id, start, T_total = self.index[idx]
        f = self._get_handle(path)
        ds = f[self.dataset_key] if self.dataset_key in f else next(iter(f.values()))

        # Frame indices within the clip (stride inside the window)
        eff_idx = start + np.arange(0, self.clip_len * self.frame_stride, self.frame_stride)
        if ds.ndim == 4:
            # [T, H, W, C] or [T, C, H, W]
            clip = ds[eff_idx, ...]
        else:
            # [N, T, H, W, C] or [N, T, C, H, W]
            clip = ds[seq_id, eff_idx, ...]

        # Ensure contiguous, owning NumPy storage to avoid collate resizing issues
        clip = np.ascontiguousarray(clip)
        clip = self._to_TCHW(clip)                 # numpy [T,C,H,W]
        clip = np.ascontiguousarray(clip)          # contiguous again after transpose
        clip = torch.from_numpy(clip).float()      # torch [T,C,H,W]

        if clip.max() > 1.0:
            clip = clip / 255.0

        if self.target_size is not None:
            H_out, W_out = self.target_size
            if clip.shape[2] != H_out or clip.shape[3] != W_out:
                clip = self._resize_tchw(clip, H_out, W_out)

        return clip  # [T,C,H,W] in [0,1]

def make_dataloader(
    data_dir="./data",
    dataset_key="frames",
    clip_len=16,
    frame_stride=1,
    step_between_clips=8,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    target_size: Optional[Tuple[int, int]] = None,  # (H,W) to enforce
):
    ds = H5GameClipsDataset(
        data_dir=data_dir,
        dataset_key=dataset_key,
        clip_len=clip_len,
        frame_stride=frame_stride,
        step_between_clips=step_between_clips,
        target_size=target_size,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
    )
    return ds, loader
