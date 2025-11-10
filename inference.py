#!/usr/bin/env python

import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import h5py

import pygame
from moviepy.editor import ImageSequenceClip  # moviepy==1.0.3 necessary

from VideoTokenizer import VideoTokenizer
from DynamicsModel import DynamicsModel
from FSQ import FiniteScalarQuantizer


# ----------------------------
# Encoders / Decoders (INDEX-BASED)
# ----------------------------

@torch.no_grad()
def encode_video_latents(vt: torch.nn.Module, clips: torch.Tensor) -> torch.Tensor:
    """
    Use the tokenizer's discrete interface:
      frames -> digit indices in [0..num_bins-1]
    Returns: [B, T, P, L] Long
    """
    vt.eval()
    return vt.encode_to_indices(clips)


@torch.no_grad()
def decode_latents_to_frames(vt: torch.nn.Module, indices: torch.Tensor) -> torch.Tensor:
    """
    Decode digit indices [B, T, P, L] back to RGB frames [B, T, C, H, W].
    """
    vt.eval()
    return vt.decode_from_indices(indices)


def load_video_tokenizer(args, device):
    vt = VideoTokenizer(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embedding_dim=args.vt_embedding_dim,
        input_channels=args.input_channels,
        num_blocks=args.vt_num_blocks,
        num_heads=args.vt_num_heads,
        inter_dim=args.vt_inter_dim,
        causal=False,
        rope_base=10_000.0,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
    ).to(device)

    if args.tokenizer_ckpt and os.path.exists(args.tokenizer_ckpt):
        ckpt = torch.load(args.tokenizer_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        vt.load_state_dict(state, strict=False)
        print(f"[info] loaded video tokenizer from {args.tokenizer_ckpt}")
    else:
        print("[warn] tokenizer checkpoint not found, using randomly initialized vt")

    vt.eval()
    for p in vt.parameters():
        p.requires_grad = False
    return vt


def load_dynamics_model(args, device):
    dyn = DynamicsModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        inter_dim=args.inter_dim,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
        action_dim=args.action_dim,
    ).to(device)

    if args.dynamics_ckpt and os.path.exists(args.dynamics_ckpt):
        ckpt = torch.load(args.dynamics_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        dyn.load_state_dict(state, strict=False)
        print(f"[info] loaded dynamics model from {args.dynamics_ckpt}")
    else:
        print("[warn] dynamics checkpoint not found, using randomly initialized dynamics model")

    dyn.eval()
    return dyn


# ----------------------------
# Load contiguous frames from a specific game
# ----------------------------

def load_context_clip_from_game(args, device):
    """
    Loads a contiguous sequence of frames from a chosen game .h5 file.

    - Picks ./data/<game>_frames.h5 (or args.data_dir).
    - Reads dataset 'frames'.
    - Converts to [1, T, C, H, W] float in [0,1].
    - Uses the first T = clip_len frames as context source.
    """
    game_filename = f"{args.game}_frames.h5"
    h5_path = os.path.join(args.data_dir, game_filename)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    print(f"[info] loading frames from {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "frames" not in f:
            raise KeyError(f"'frames' dataset not found in {h5_path}")
        arr = f["frames"][:]  # expect [N,H,W,3] or [N,3,H,W]

    if arr.ndim != 4:
        raise ValueError(f"Expected frames with 4 dims, got {arr.shape}")

    # Convert to [N,C,H,W]
    if arr.shape[-1] == 3:
        # [N,H,W,3] -> [N,3,H,W]
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.shape[1] == 3:
        # [N,3,H,W] already
        pass
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {arr.shape}")

    frames = torch.from_numpy(arr).float() / 255.0  # [N,3,H,W] in [0,1]

    # Pick a clip_len window starting at 0 (contiguous)
    N, C, H, W = frames.shape
    if (H != args.frame_size) or (W != args.frame_size):
        frames = F.interpolate(
            frames, size=(args.frame_size, args.frame_size),
            mode="bilinear", align_corners=False
        )
        H = W = args.frame_size
        print(f"[info] resized frames to {H}x{W} to match tokenizer")

    T = min(args.clip_len, N)
    frames = frames[:T]  # [T,3,H,W]

    clips = frames.unsqueeze(0).to(device)  # [1,T,3,H,W]
    print(f"[info] loaded context clip of shape {clips.shape} from game '{args.game}'")
    return clips


# ----------------------------
# Discrete action table via FSQ bin centers (for actions)
# ----------------------------

def build_action_table(action_q: FiniteScalarQuantizer,
                       action_dim: int,
                       device: torch.device) -> torch.Tensor:
    """
    Build a small set of discrete macro-actions using the SAME FSQ scheme as the
    action tokenizer (per-dimension bin centers via unscale_and_unshift).

    action_q: FiniteScalarQuantizer(latent_dim=action_dim, num_bins=action_num_bins)
    returns: [N_actions, action_dim] continuous latents
    """
    num_bins = action_q.num_bins
    mid = num_bins // 2
    low = 0
    high = num_bins - 1

    def make_latent(digit_overrides):
        digits = torch.full((action_dim,), mid, device=device, dtype=torch.float32)
        for d, val in digit_overrides.items():
            if 0 <= d < action_dim:
                digits[d] = float(val)
        # digits: [A] in [0..num_bins-1]
        latents = action_q.unscale_and_unshift(digits)  # [A]
        return latents

    # 0: neutral, 1–4: simple directions in first 2 dims
    a0 = make_latent({})
    a1 = make_latent({0: high})         # +X
    a2 = make_latent({0: low})          # -X
    a3 = make_latent({1: high}) if action_dim > 1 else a0.clone()
    a4 = make_latent({1: low})  if action_dim > 1 else a0.clone()

    ACTION_TABLE = torch.stack([a0, a1, a2, a3, a4], dim=0)  # [5,A]
    return ACTION_TABLE


def get_action_from_keys_discrete(keys, action_table: torch.Tensor) -> torch.Tensor:
    """
    keys: pygame.key.get_pressed()
    action_table: [N_actions, A]

    Mapping:
      1 -> neutral
      2 -> +X
      3 -> -X
      4 -> +Y
      5 -> -Y

    If no key is pressed, we default to neutral.
    """
    idx = 0
    if keys[pygame.K_1]:
        idx = 0
    elif keys[pygame.K_2]:
        idx = 1
    elif keys[pygame.K_3]:
        idx = 2
    elif keys[pygame.K_4]:
        idx = 3
    elif keys[pygame.K_5]:
        idx = 4

    return action_table[idx]  # [A]


# ----------------------------
# Pygame rollout
# ----------------------------

@torch.no_grad()
def rollout_pygame(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # models
    vt = load_video_tokenizer(args, device)
    dyn = load_dynamics_model(args, device)

    # FSQ for actions (must match ActionTokenizer spec)
    action_q = FiniteScalarQuantizer(
        latent_dim=args.action_dim,
        num_bins=args.action_num_bins,
    ).to(device)

    # load contiguous context frames from chosen game
    clips = load_context_clip_from_game(args, device)  # [1,T,3,H,W]
    B, T, C, H, W = clips.shape

    # encode video tokens as DISCRETE INDICES
    z_all = encode_video_latents(vt, clips)   # [1, T, P, L] Long
    _, Tz, P, L = z_all.shape

    # multi-frame context
    ctx_len = min(args.context_len, Tz)
    if ctx_len < 1:
        ctx_len = 1
    z_hist = z_all[:, :ctx_len]              # [1, ctx_len, P, L]

    # action history: zeros for the ctx transitions
    action_dim = args.action_dim
    if ctx_len > 1:
        a_hist = torch.zeros(1, ctx_len - 1, action_dim, device=device)
    else:
        a_hist = torch.empty(1, 0, action_dim, device=device)

    # decode last context frame for initial display
    ctx_indices = z_hist[:, -1:]                 # [1,1,P,L]
    ctx_frame = decode_latents_to_frames(vt, ctx_indices)  # [1,1,C,H,W]
    frame_tensor = ctx_frame[0, 0].detach().cpu().clamp(0, 1)  # [C,H,W]

    # discrete action table in FSQ latent space
    action_table = build_action_table(action_q, action_dim, device=device)

    # prepare output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- pygame setup ----------
    pygame.init()
    scale = args.display_scale
    win_size = (W * scale, H * scale)
    screen = pygame.display.set_mode(win_size)
    pygame.display.set_caption(f"Dynamics (game={args.game})")
    clock = pygame.time.Clock()

    def tensor_to_surface_and_array(t):
        """
        t: [C,H,W] in [0,1]
        returns:
          - pygame.Surface for display
          - HxWx3 uint8 numpy array for video writing
        """
        img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HxWx3
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))   # (W,H,3)
        if scale != 1:
            surf = pygame.transform.scale(surf, win_size)
        return surf, img

    current_surface, first_video_frame = tensor_to_surface_and_array(frame_tensor)
    video_frames = [first_video_frame]

    running = True
    step = 0
    max_steps = args.num_steps

    print()
    print(f"Interactive rollout from game '{args.game}':")
    print(f"  context_len = {ctx_len} frames")
    print(f"  generating up to {max_steps} additional frames.")
    print("Controls:")
    print("  1: neutral")
    print("  2: +X (e.g. right)")
    print("  3: -X (e.g. left)")
    print("  4: +Y (e.g. up)")
    print("  5: -Y (e.g. down)")
    print("  ESC / close window: quit")
    print()

    while running and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break
        if not running:
            break

        keys = pygame.key.get_pressed()
        a_vec = get_action_from_keys_discrete(keys, action_table)  # [A]
        a_vec = a_vec.view(1, 1, action_dim)  # [1,1,A]

        # append action to history
        if a_hist.size(1) == 0:
            a_hist = a_vec
        else:
            a_hist = torch.cat([a_hist, a_vec], dim=1)  # [1, Thist, A]

        Thist = z_hist.size(1)

        # ensure actions length matches transitions (Thist-1)
        # if actions are shorter, pad with zeros
        if a_hist.size(1) < Thist - 1:
            pad = (Thist - 1) - a_hist.size(1)
            pad_zeros = torch.zeros(1, pad, action_dim, device=device)
            a_hist = torch.cat([a_hist, pad_zeros], dim=1)
        # if somehow longer, truncate
        if a_hist.size(1) > Thist - 1:
            a_hist = a_hist[:, :Thist - 1]

        a_ctx = a_hist  # [1, Thist-1, A]
        assert a_ctx.size(1) == Thist - 1, f"a_ctx shape mismatch: {a_ctx.shape}, Thist={Thist}"

        # predict next discrete indices
        dyn.eval()
        logits, _ = dyn(z_hist, actions=a_ctx, context_mask_ratio=0.0)
        # logits: [1, Thist-1, P, L, K]
        logits_last = logits[:, -1]                 # [1, P, L, K]
        pred_digits = logits_last.argmax(dim=-1)    # [1, P, L] (Long indices)

        pred_indices = pred_digits.unsqueeze(1)     # [1,1,P,L]

        # extend history in index space
        z_hist = torch.cat([z_hist, pred_indices], dim=1)  # [1, Thist+1, P, L]

        # decode and display
        pred_frame = decode_latents_to_frames(vt, pred_indices)  # [1,1,C,H,W]
        frame_tensor = pred_frame[0, 0].detach().cpu().clamp(0, 1)
        current_surface, video_frame = tensor_to_surface_and_array(frame_tensor)
        video_frames.append(video_frame)

        screen.blit(current_surface, (0, 0))
        pygame.display.flip()

        step += 1
        clock.tick(args.fps)

    pygame.quit()
    print("[info] pygame rollout finished.")

    # write video
    if len(video_frames) > 1:
        video_path = os.path.join(args.out_dir, f"rollout_{args.game}.mp4")
        clip = ImageSequenceClip(video_frames, fps=args.fps)
        clip.write_videofile(video_path, codec="libx264")
        print(f"[info] saved rollout video to {video_path}")
    else:
        print("[warn] not enough frames to write a video.")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Real-time pygame inference for DynamicsModel "
                    "(FSQ-discrete actions, multi-frame context, per-game)"
    )

    # Paths
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--tokenizer_ckpt", type=str, default="./runs/unified/video_tokenizer/checkpoints/best.pth")
    p.add_argument("--dynamics_ckpt",  type=str, default="./runs/unified/dynamics/checkpoints/best.pth")
    p.add_argument("--out_dir", type=str, default="./dynamics_rollout_pygame")

    # Which game (maps to <game>_frames.h5)
    p.add_argument(
        "--game",
        type=str,
        default="picodoom",
        choices=["picodoom", "pole_position", "pong", "sonic", "zelda"],
        help="which game to use as context, expects <game>_frames.h5 in data_dir",
    )

    # Video tokenizer / dynamics hyperparams (must match training)
    p.add_argument("--frame_size",   type=int, default=128)
    p.add_argument("--patch_size",   type=int, default=4)
    p.add_argument("--input_channels", type=int, default=3)

    p.add_argument("--vt_embedding_dim", type=int, default=256)
    p.add_argument("--vt_num_blocks",    type=int, default=8)
    p.add_argument("--vt_num_heads",     type=int, default=8)
    p.add_argument("--vt_inter_dim",     type=int, default=512)

    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--num_blocks",   type=int, default=8)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--inter_dim",    type=int, default=512)
    p.add_argument("--latent_dim",   type=int, default=10)
    p.add_argument("--num_bins",     type=int, default=8)   # video FSQ bins
    p.add_argument("--action_dim",   type=int, default=4)

    # FSQ for actions (must match ActionTokenizer)
    p.add_argument(
        "--action_num_bins",
        type=int,
        default=3,
        help="number of FSQ bins per action dimension (match ActionTokenizer)",
    )

    # Rollout data
    p.add_argument("--clip_len",     type=int, default=16,
                   help="number of contiguous frames to load from the game file")
    p.add_argument("--context_len",  type=int, default=16,
                   help="how many of those frames to use as context")
    p.add_argument("--fps",          type=int, default=15,
                   help="pygame display & video fps")
    p.add_argument("--display_scale",type=int, default=4,
                   help="integer upscaling factor for the pygame window")
    p.add_argument("--num_steps",    type=int, default=200,
                   help="number of predicted frames to generate")

    args = p.parse_args()
    rollout_pygame(args)
