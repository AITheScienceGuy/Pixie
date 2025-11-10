#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import numpy as np
import random
from torch.amp import GradScaler, autocast
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import argparse

from VideoDataLoader import H5GameClipsDataset
from VideoTokenizer import VideoTokenizer
from ActionTokenizer import LatentActionModel
from DynamicsModel import DynamicsModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Video Tokenizer Training
# ----------------------------
def train_video_tokenizer(args, device):
    print("\n--- Starting Video Tokenizer Training ---")
    log_dir = os.path.join(args.log_dir, "video_tokenizer")
    model_save_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    latest_checkpoint_path = os.path.join(model_save_dir, "latest.pth")
    best_checkpoint_path = os.path.join(model_save_dir, "best.pth")
    args.tokenizer_ckpt = best_checkpoint_path

    full_dataset = H5GameClipsDataset(
        data_dir=args.data_dir,
        dataset_key="frames",
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        step_between_clips=args.step_between_clips,
        target_size=(args.frame_size, args.frame_size),
    )

    val_split = 0.1
    num_train = int((1 - val_split) * len(full_dataset))
    num_val = len(full_dataset) - num_train
    train_dataset, val_dataset = random_split(
        full_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True, 'persistent_workers': args.num_workers > 0}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    fixed_val_batch = next(iter(val_loader)).to(device)

    model = VideoTokenizer(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        input_channels=args.input_channels,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        inter_dim=args.inter_dim,
        causal=args.causal,
        rope_base=10_000.0,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.vt_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.vt_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")
    recon_loss_fn = nn.L1Loss()
    commitment_loss_fn = nn.MSELoss()

    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')
    if os.path.exists(latest_checkpoint_path):
        print("Resuming Video Tokenizer from latest checkpoint...")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.vt_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"VT Epoch {epoch}/{args.vt_epochs} TRAIN", leave=False)
        for clips in pbar:
            clips = clips.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                latents = model.encoder(clips)
                quantized_latents = model.quantizer(latents)
                x_hat = model.decoder(quantized_latents)
                recon_loss = recon_loss_fn(x_hat, clips)
                commitment_loss = commitment_loss_fn(latents, quantized_latents.detach())
                loss = recon_loss + args.commitment_beta * commitment_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            pbar.set_postfix(loss=float(loss), recon=float(recon_loss))
            writer.add_scalar("train/total_loss", float(loss), global_step)
            writer.add_scalar("train/recon_loss", float(recon_loss), global_step)
            writer.add_scalar("train/commitment_loss", float(commitment_loss), global_step)
            epoch_loss += float(loss)
            global_step += 1
        scheduler.step()
        writer.add_scalar("train/epoch_loss", epoch_loss / len(train_loader), epoch)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch)

        model.eval()
        val_loss, total_usage = 0.0, 0.0
        val_pbar = tqdm(val_loader, desc=f"VT Epoch {epoch}/{args.vt_epochs} VALIDATION", leave=False)
        with torch.no_grad():
            for clips in val_pbar:
                clips = clips.to(device, non_blocking=True)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    x_hat = model(clips)
                    loss = recon_loss_fn(x_hat, clips)
                    quantized_latents = model.encode(clips)

                val_loss += float(loss)
                total_usage += model.quantizer.get_codebook_usage(quantized_latents)
                val_pbar.set_postfix(loss=float(loss))
        avg_val_loss = val_loss / len(val_loader)
        avg_usage = total_usage / len(val_loader)
        writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
        writer.add_scalar("val/codebook_usage_percent", avg_usage * 100, epoch)

        checkpoint = {
            'epoch': epoch, 'global_step': global_step,
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, latest_checkpoint_path)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, best_checkpoint_path)
            tqdm.write(f"VT Epoch {epoch} | New best val_loss: {best_val_loss:.4f}, model saved.")

        with torch.no_grad():
            x_hat_fixed = model(fixed_val_batch)
            b = min(4, fixed_val_batch.size(0))
            x = fixed_val_batch[:b].detach().cpu().clamp(0, 1)
            y = x_hat_fixed[:b].detach().cpu().clamp(0, 1)
            grid_in0 = make_grid(x[:, 0], nrow=b)
            grid_out0 = make_grid(y[:, 0], nrow=b)
            writer.add_image("val_fixed/frame0_in", grid_in0, epoch)
            writer.add_image("val_fixed/frame0_out", grid_out0, epoch)
        tqdm.write(f"VT Epoch {epoch} | train_loss {epoch_loss/len(train_loader):.4f} | val_loss {avg_val_loss:.4f} | usage {avg_usage*100:.2f}%")
    writer.close()
    print("--- Video Tokenizer Training Finished ---")


# ----------------------------
# Action Tokenizer Training
# ----------------------------
def train_action_tokenizer(args, device):
    print("\n--- Starting Action Tokenizer Training ---")
    log_dir = os.path.join(args.log_dir, "action_tokenizer")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    args.action_ckpt = best_ckpt

    full_dataset = H5GameClipsDataset(
        data_dir=args.data_dir, dataset_key="frames", clip_len=args.clip_len,
        frame_stride=args.frame_stride, step_between_clips=args.step_between_clips,
        target_size=(args.frame_size, args.frame_size),
    )
    val_split = args.val_split
    n_total = len(full_dataset)
    n_train = int((1 - val_split) * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True, 'persistent_workers': args.num_workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    fixed_val_batch = next(iter(val_loader)).to(device)

    model = LatentActionModel(
        frame_size=(args.frame_size, args.frame_size), patch_size=args.patch_size,
        embedding_dim=args.embedding_dim, input_channels=args.input_channels,
        num_blocks=args.num_blocks, num_heads=args.num_heads,
        inter_dim=args.inter_dim, action_dim=args.action_dim,
        conditioning_dim=args.action_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.at_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.at_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")
    recon_loss_fn = nn.L1Loss()
    commitment_loss_fn = nn.MSELoss()

    start_epoch, global_step, best_val = 1, 0, float("inf")
    if os.path.exists(latest_ckpt):
        print("Resuming Action Tokenizer from latest checkpoint...")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"]); scaler.load_state_dict(ckpt["scaler"])
        start_epoch, global_step, best_val = ckpt["epoch"] + 1, ckpt["global_step"], ckpt.get("best_val", best_val)
        print(f"→ resumed at epoch {start_epoch} (best val {best_val:.4f})")

    for epoch in range(start_epoch, args.at_epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"AT Epoch {epoch}/{args.at_epochs} TRAIN", leave=False)
        for clips in pbar:
            clips = clips.to(device, non_blocking=True)
            targets = clips[:, 1:]
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                latents = model.action_encoder(clips)
                quantized_latents = model.quantizer(latents)
                preds = model.action_decoder(clips, quantized_latents)

                recon_loss = recon_loss_fn(preds, targets)
                commitment_loss = commitment_loss_fn(latents, quantized_latents.detach())
                loss = recon_loss + args.action_commitment_beta * commitment_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss)
            pbar.set_postfix(loss=float(loss))
            writer.add_scalar("train/loss", float(loss), global_step)
            global_step += 1
        sched.step()
        epoch_loss = running / max(1, len(train_loader))
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], epoch)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"AT Epoch {epoch}/{args.at_epochs} VAL", leave=False)
            for clips in vbar:
                clips = clips.to(device, non_blocking=True)
                targets = clips[:, 1:]
                with autocast("cuda", enabled=(device.type == "cuda")):
                    preds = model(clips)
                    vloss = recon_loss_fn(preds, targets)
                val_sum += float(vloss)
                vbar.set_postfix(loss=float(vloss))
        val_avg = val_sum / max(1, len(val_loader))
        writer.add_scalar("val/epoch_loss", val_avg, epoch)

        ckpt = {"epoch": epoch, "global_step": global_step, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "best_val": best_val}
        torch.save(ckpt, latest_ckpt)
        if val_avg < best_val:
            best_val = val_avg
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_ckpt)
            tqdm.write(f"AT Epoch {epoch} | new best val {best_val:.4f} → saved")

        with torch.no_grad():
            preds_fixed = model(fixed_val_batch)
            b = min(4, fixed_val_batch.size(0))
            inp0 = fixed_val_batch[:b, 0].detach().cpu().clamp(0, 1)
            tgt1 = fixed_val_batch[:b, 1].detach().cpu().clamp(0, 1)
            pred1 = preds_fixed[:b, 0].detach().cpu().clamp(0, 1)
            grid_in = make_grid(inp0, nrow=b)
            grid_tgt = make_grid(tgt1, nrow=b)
            grid_out = make_grid(pred1, nrow=b)
            writer.add_image("val_fixed/frame_t0_input", grid_in, epoch)
            writer.add_image("val_fixed/frame_t1_target", grid_tgt, epoch)
            writer.add_image("val_fixed/frame_t1_pred", grid_out, epoch)
        tqdm.write(f"AT Epoch {epoch} | train {epoch_loss:.4f} | val {val_avg:.4f}")
    writer.close()
    print("--- Action Tokenizer Training Finished ---")


# ----------------------------
# Dynamics Model Training
# ----------------------------
@torch.no_grad()
def _log_decoded_predictions(
    writer, vt, epoch: int, clips: torch.Tensor,
    logits: torch.Tensor, z_tgt_idx: torch.Tensor,
    max_batch: int = 4, fps: int = 8,
):
    """
    Logs one-step predictions:
      logits:   [B, T-1, P, L, num_bins]
      z_tgt_idx:[B, T-1, P, L]
    """
    # Predicted digits from logits
    pred_idx_digits = torch.argmax(logits, dim=-1)          # [B, T-1, P, L]

    # Decode both predicted and target future tokens
    pred_rgb = vt.decode_from_indices(pred_idx_digits)      # [B, T-1, C, H, W]
    gt_rgb   = vt.decode_from_indices(z_tgt_idx)            # [B, T-1, C, H, W]

    # Match batch size
    b = min(
        max_batch,
        clips.size(0),
        pred_rgb.size(0),
        gt_rgb.size(0),
    )

    x_in = clips[:b, :-1].detach().float().clamp(0, 1).cpu()
    y_gt = gt_rgb[:b].detach().float().clamp(0, 1).cpu()
    y_pr = pred_rgb[:b].detach().float().clamp(0, 1).cpu()

    writer.add_video("val_fixed/clip_input_context", x_in, epoch, fps=fps)
    writer.add_video("val_fixed/clip_future_gt", y_gt, epoch, fps=fps)
    writer.add_video("val_fixed/clip_future_pred", y_pr, epoch, fps=fps)

@torch.no_grad()
def log_autoregressive_rollout(
    writer, vt, act_tok, dm, epoch: int, clips: torch.Tensor,
    context_frames: int, prediction_steps: int,
    max_batch: int = 4, fps: int = 8
):
    dm.eval()
    vt_device = next(vt.parameters()).device
    clips = clips.to(vt_device)
    b_size = min(max_batch, clips.size(0))

    # For visualization
    context_clips_rgb = clips[:b_size, :context_frames]
    future_clips_rgb  = clips[:b_size, context_frames : context_frames + prediction_steps]

    # Discrete video tokens and latent actions for the full clip
    z_indices_all = encode_video_indices(vt, clips[:b_size])        # [B, T, P, L]
    a_latents_all = encode_action_latents(act_tok, clips[:b_size])  # [B, T-1, A]

    # Start from context frames
    context_indices = z_indices_all[:, :context_frames]             # [B, k, P, L]

    predicted_indices_list = []
    current_indices = context_indices.clone()

    for i in range(prediction_steps):
        current_T = current_indices.size(1)
        if current_T <= 1:
            current_actions = None
        else:
            # Use one action per transition: length = current_T - 1
            current_actions = a_latents_all[:, : current_T - 1]     # [B, current_T-1, A]

        logits, _ = dm(current_indices, actions=current_actions, context_mask_ratio=0.0)

        # Take the prediction for the *last* timestep
        next_step_logits = logits[:, -1]                            # [B, P, L, num_bins]
        next_indices = torch.argmax(next_step_logits, dim=-1).unsqueeze(1)  # [B, 1, P, L]

        predicted_indices_list.append(next_indices)
        # Append predicted frame to the context for the next step
        current_indices = torch.cat([current_indices, next_indices], dim=1)

    predicted_indices = torch.cat(predicted_indices_list, dim=1)    # [B, pred_T, P, L]

    # Decode predicted and ground-truth future frames
    decoded_prediction_rgb = vt.decode_from_indices(predicted_indices)  # [B, pred_T, C, H, W]

    separator = torch.zeros_like(context_clips_rgb[:, :1])
    separator[:, :, 0] = 1.0  # red channel separator

    full_rollout_video = torch.cat(
        [context_clips_rgb, separator, decoded_prediction_rgb],
        dim=1,
    ).cpu().clamp(0, 1)

    ground_truth_video = torch.cat(
        [context_clips_rgb, future_clips_rgb],
        dim=1,
    ).cpu().clamp(0, 1)

    writer.add_video(
        f"rollout/imagined_future (k={context_frames})", full_rollout_video,
        epoch, fps=fps
    )
    writer.add_video(
        "rollout/ground_truth", ground_truth_video,
        epoch, fps=fps
    )

@torch.no_grad()
def encode_video_indices(vt: nn.Module, clips: torch.Tensor) -> torch.Tensor:
    return vt.encode_to_indices(clips)

@torch.no_grad()
def encode_action_latents(act_tok: nn.Module, clips: torch.Tensor) -> torch.Tensor:
    return act_tok.encode(clips)

@torch.no_grad()
def top1_factor_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == targets).float().mean().item()

def train_dynamics_model(args, device):
    print("\n--- Starting Dynamics Model Training ---")
    log_dir = os.path.join(args.log_dir, "dynamics")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt, best_ckpt = os.path.join(ckpt_dir, "latest.pth"), os.path.join(ckpt_dir, "best.pth")
    writer = SummaryWriter(log_dir)

    full_dataset = H5GameClipsDataset(
        data_dir=args.data_dir, dataset_key="frames", clip_len=args.clip_len,
        frame_stride=args.frame_stride, step_between_clips=args.step_between_clips,
        target_size=(args.frame_size, args.frame_size),
    )
    n_total = len(full_dataset)
    n_train = int((1.0 - args.val_split) * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True, 'persistent_workers': args.num_workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    fixed_val_batch = next(iter(val_loader)).to(device)

    vt_kwargs = {'frame_size':(args.frame_size, args.frame_size), 'patch_size':args.patch_size, 'embedding_dim':args.embedding_dim, 'input_channels':args.input_channels, 'num_blocks':args.num_blocks, 'num_heads':args.num_heads, 'inter_dim':args.inter_dim, 'causal':args.causal, 'rope_base':10_000.0, 'latent_dim':args.latent_dim, 'num_bins':args.num_bins}
    vt = VideoTokenizer(**vt_kwargs).to(device)
    if args.tokenizer_ckpt and os.path.exists(args.tokenizer_ckpt):
        vt_ckpt = torch.load(args.tokenizer_ckpt, map_location="cpu")
        state = vt_ckpt.get("model_state_dict", vt_ckpt.get("model", vt_ckpt))
        vt.load_state_dict(state)
        print(f"Loaded video tokenizer checkpoint: {args.tokenizer_ckpt}")
    else:
        print("WARNING: No video tokenizer checkpoint found. Using random weights.")
    vt.eval()
    for p in vt.parameters(): p.requires_grad = False

    at_kwargs = {'frame_size':(args.frame_size, args.frame_size), 'patch_size':args.patch_size, 'embedding_dim':args.embedding_dim, 'input_channels':args.input_channels, 'num_blocks':args.num_blocks, 'num_heads':args.num_heads, 'inter_dim':args.inter_dim, 'action_dim':args.action_dim, 'conditioning_dim':args.action_dim}
    act_tok = LatentActionModel(**at_kwargs).to(device)
    if args.action_ckpt and os.path.exists(args.action_ckpt):
        at_ckpt = torch.load(args.action_ckpt, map_location="cpu")
        state = at_ckpt.get("model_state_dict", at_ckpt.get("model", at_ckpt))
        act_tok.load_state_dict(state)
        print(f"Loaded action tokenizer checkpoint: {args.action_ckpt}")
    else:
        print("WARNING: No action tokenizer checkpoint found. Using random weights.")
    act_tok.eval()
    for p in act_tok.parameters(): p.requires_grad = False

    model = DynamicsModel(
        frame_size=(args.frame_size, args.frame_size), patch_size=args.patch_size,
        embedding_dim=args.embedding_dim, num_blocks=args.num_blocks,
        num_heads=args.num_heads, inter_dim=args.inter_dim,
        latent_dim=args.latent_dim, num_bins=args.num_bins,
        action_dim=args.action_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.dm_lr, betas=(0.9, 0.999), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.dm_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    start_epoch, global_step, best_val = 1, 0, float("inf")
    if os.path.exists(latest_ckpt):
        print("Resuming Dynamics Model from latest checkpoint...")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"]); scaler.load_state_dict(ckpt["scaler"])
        start_epoch, global_step, best_val = ckpt["epoch"] + 1, ckpt["global_step"], ckpt.get("best_val", best_val)
        print(f"→ resumed at epoch {start_epoch} (best val {best_val:.4f})")

    for epoch in range(start_epoch, args.dm_epochs + 1):
        model.train()
        running, acc_running = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"DM Epoch {epoch}/{args.dm_epochs} TRAIN", leave=False)
        for clips in pbar:
            clips = clips.to(device, non_blocking=True)
            with torch.no_grad():
                z_indices = encode_video_indices(vt, clips)
                a = encode_action_latents(act_tok, clips)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits, targets = model(z_indices, actions=a, context_mask_ratio=args.context_mask_ratio)
                logits_flat  = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                loss = criterion(logits_flat, targets_flat)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad(): acc = top1_factor_acc(logits, targets)
            running += float(loss); acc_running += acc
            pbar.set_postfix(loss=float(loss), acc=acc)
            writer.add_scalar("train/loss", float(loss), global_step); writer.add_scalar("train/acc_top1_factor", float(acc), global_step)
            global_step += 1
        sched.step()
        epoch_loss, epoch_acc = running/len(train_loader), acc_running/len(train_loader)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch); writer.add_scalar("train/epoch_acc", epoch_acc, epoch)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], epoch)

        model.eval()
        val_sum, val_acc = 0.0, 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"DM Epoch {epoch}/{args.dm_epochs} VAL", leave=False)
            for clips in vbar:
                clips = clips.to(device, non_blocking=True)
                z_indices = encode_video_indices(vt, clips)
                a = encode_action_latents(act_tok, clips)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits, targets = model(z_indices, actions=a, context_mask_ratio=0.0)
                    logits_flat  = logits.reshape(-1, logits.size(-1))
                    targets_flat = targets.reshape(-1)
                    vloss = criterion(logits_flat, targets_flat)
                val_sum += float(vloss)
                val_acc += top1_factor_acc(logits, targets)
                vbar.set_postfix(loss=float(vloss))
        val_avg, val_acc_avg = val_sum/len(val_loader), val_acc/len(val_loader)
        writer.add_scalar("val/epoch_loss", val_avg, epoch)
        writer.add_scalar("val/epoch_acc", val_acc_avg, epoch)
        
        with torch.no_grad():
            z_indices_fixed = encode_video_indices(vt, fixed_val_batch)
            a_fixed = encode_action_latents(act_tok, fixed_val_batch)
            logits_fixed, targets_fixed = model(z_indices_fixed, actions=a_fixed, context_mask_ratio=0.0)
            _log_decoded_predictions(writer, vt, epoch, fixed_val_batch, logits_fixed, targets_fixed)

            if epoch % args.rollout_freq == 0 or epoch == args.dm_epochs:
                log_autoregressive_rollout(
                    writer, vt, act_tok, model, epoch, fixed_val_batch,
                    context_frames=args.context_frames,
                    prediction_steps=args.prediction_steps
                )

        ckpt = {"epoch": epoch, "global_step": global_step, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best_val": best_val}
        torch.save(ckpt, latest_ckpt)
        if val_avg < best_val:
            best_val = val_avg
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_ckpt)
            tqdm.write(f"DM Epoch {epoch} | new best val {best_val:.4f} → saved")

        tqdm.write(f"DM Epoch {epoch} | train {epoch_loss:.4f} (acc {epoch_acc:.3f}) | val {val_avg:.4f} (acc {val_acc_avg:.3f})")
    writer.close()
    print("--- Dynamics Model Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Script for Video, Action, and Dynamics Models")
    # Shared Paths & Reproducibility
    parser.add_argument('--log_dir', type=str, default='./runs/unified')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    # Shared Model Parameters
    parser.add_argument('--frame_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--inter_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--num_bins', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=4)

    # Shared Data Parameters
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--step_between_clips', type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--vt_lr', type=float, default=1e-4, help="Learning rate for Video Tokenizer")
    parser.add_argument('--at_lr', type=float, default=1e-4, help="Learning rate for Action Tokenizer")
    parser.add_argument('--dm_lr', type=float, default=5e-5, help="Learning rate for Dynamics Model")

    # Stage-specific Epochs
    parser.add_argument('--vt_epochs', type=int, default=50, help="Epochs for Video Tokenizer training")
    parser.add_argument('--at_epochs', type=int, default=50, help="Epochs for Action Tokenizer training")
    parser.add_argument('--dm_epochs', type=int, default=100, help="Epochs for Dynamics Model training")

    # Model-specific Parameters
    parser.add_argument('--causal', action='store_true', default=False, help="Use causal attention for Video Tokenizer Decoder")
    parser.add_argument('--commitment_beta', type=float, default=0.25, help="Weight for the commitment loss in Video Tokenizer")
    parser.add_argument('--action_commitment_beta', type=float, default=0.25, help="Weight for the commitment loss in Action Tokenizer")
    parser.add_argument("--context_mask_ratio", type=float, default=0.4, help="Context mask ratio for Dynamics Model during training")
    
    # Autoregressive Rollout Parameters
    parser.add_argument('--rollout_freq', type=int, default=5, help="How often (in epochs) to log an autoregressive rollout video")
    parser.add_argument('--context_frames', type=int, default=16, help="Number of context frames for the rollout")
    parser.add_argument('--prediction_steps', type=int, default=32, help="Number of steps to predict in the rollout")

    # Checkpoint paths (will be set automatically)
    parser.add_argument("--tokenizer_ckpt", type=str, default=None, help="Path to video tokenizer checkpoint (set automatically)")
    parser.add_argument("--action_ckpt", type=str, default=None, help="Path to action tokenizer checkpoint (set automatically)")

    args = parser.parse_args()
    
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Execute Training Stages Sequentially ---
    if args.vt_epochs > 0:
        train_video_tokenizer(args, device)
    else:
        print("Skipping Video Tokenizer training as --vt_epochs is 0.")

    if args.at_epochs > 0:
        train_action_tokenizer(args, device)
    else:
        print("Skipping Action Tokenizer training as --at_epochs is 0.")

    if args.dm_epochs > 0:
        train_dynamics_model(args, device)
    else:
        print("Skipping Dynamics Model training as --dm_epochs is 0.")

    print("\nAll training stages are complete.")
