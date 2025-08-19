#!/usr/bin/env python
"""
Gesture Flow-Matching Fine-tuning using OmniFlow Transformer.

This script fine-tunes the OmniFlowTransformerModel on gesture latents produced by the
ConcatRVQVAEAdapter (RVQVAE×4 concat). It uses the OmniFlowMatchEulerDiscreteScheduler
for noise scheduling and an epsilon-prediction target by default.
"""
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omniflow.utils.ema import EMAModel
import time
import math
import wandb

from omniflow.pipelines.omniflow_pipeline import OmniFlowPipeline
from omniges.gesture_vae_adapter import ConcatRVQVAEAdapter
from dataloaders.beat_sep_lower import CustomDataset as BeatDataset
from utils.config import parse_args as parse_dataset_args
import yaml


def load_yaml(fp: str) -> Dict:
    with open(fp, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict):
    ds_cfg_path = cfg["dataset"].get("dataset_cfg", "configs/beat2_rvqvae.yaml")
    ds_args, _ = parse_dataset_args(ds_cfg_path)

    def need_build(split: str) -> bool:
        preloaded_dir = ds_args.root_path + ds_args.cache_path + split + f"/{ds_args.pose_rep}_cache"
        mapping_path = os.path.join(preloaded_dir, "sample_db_mapping.pkl")
        return not os.path.exists(mapping_path)

    # Train always build (first time) to ensure cache exists
    train_set = BeatDataset(ds_args, "train", build_cache=True)

    # Val uses csv if present; build cache if missing
    has_split_csv = os.path.exists(os.path.join(ds_args.data_path, cfg["dataset"]["split_csv"]))
    val_split = "val" if has_split_csv else "test"
    val_build = need_build(val_split)
    val_set = BeatDataset(ds_args, val_split, build_cache=val_build)

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 4)
    return (
        DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nw),
    )


def build_adapter(cfg: Dict, device: torch.device) -> ConcatRVQVAEAdapter:
    gcfg = cfg["gesture"]
    ck = gcfg["rvq_ckpt"]
    return ConcatRVQVAEAdapter(
        upper_ckpt=ck["upper"],
        hands_ckpt=ck["hands"],
        lower_ckpt=ck["lower_trans"],
        face_ckpt=ck.get("face"),
        nb_code=cfg.get("nb_code", 1024),
        code_dim=cfg.get("code_dim", 128),
        num_quantizers=cfg.get("num_quantizers", 6),
        fuse_in_channels=cfg["backbone"]["dim"] // cfg["backbone"]["heads"],
        sample_size=cfg.get("sample_size", 128),
        device=device,
    )


class AudioProjector(nn.Module):
    def __init__(self, in_dim: int, token_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] or [B, F]
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.proj(x)


class AudioProjectorTemporal(nn.Module):
    def __init__(self, in_dim: int, token_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )

    def forward(self, x_seq: torch.Tensor, t_out: int) -> torch.Tensor:
        """x_seq: [B, T_aud, F] -> [B, t_out, token_dim]"""
        if x_seq.shape[1] != t_out:
            x_seq = torch.nn.functional.interpolate(x_seq.permute(0, 2, 1), size=t_out, mode="linear", align_corners=False).permute(0, 2, 1)
        return self.fc(x_seq)


def flow_train_step(
    transformer: nn.Module,
    scheduler,
    z0: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    loss_type: str = "epsilon",
) -> Tuple[torch.Tensor, Dict]:
    """One flow-matching step on latents z0 using scheduler timing.
    Returns loss and aux dict.
    """
    device = z0.device
    b = z0.shape[0]
    noise = torch.randn_like(z0)
    # Sample discrete timesteps
    t = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device).long()
    noisy = scheduler.add_noise(z0, noise, t)

    # Forward transformer; the exact signature may differ based on the checkpoint.
    # Common SD3-like signature: (hidden_states, timestep, encoder_hidden_states, pooled_projections, ...)
    model_out = transformer(
        hidden_states=noisy,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
    )
    if isinstance(model_out, dict) and "sample" in model_out:
        pred = model_out["sample"]
    else:
        pred = model_out

    if loss_type == "epsilon":
        target = noise
    elif loss_type == "sample":
        target = z0
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = torch.nn.functional.smooth_l1_loss(pred, target)
    return loss, {"t_mean": t.float().mean().item()}


def build_prompt_embeddings(pipe: OmniFlowPipeline, batch_size: int, device: torch.device):
    # Minimal empty prompts for conditioning; adapt if text/audio prompts are desired.
    prompt = [""] * batch_size
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        use_t5=False,
        add_token_embed=False,
        max_sequence_length=128,
    )
    return prompt_embeds, pooled_prompt_embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="omniges yaml config path")
    ap.add_argument("--max_steps", type=int, default=None, help="Stop after N optimizer steps (smoke test)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accelerator/Logging
    proj_conf = ProjectConfiguration(project_dir=cfg["logging"]["checkpoint_dir"], logging_dir=cfg["logging"]["log_dir"])
    accelerator = Accelerator(gradient_accumulation_steps=cfg["train"].get("grad_accum", 1), project_config=proj_conf, mixed_precision="bf16" if torch.cuda.is_available() else "no")
    if accelerator.is_main_process and cfg["logging"].get("wandb_project"):
        wandb.init(project=cfg["logging"]["wandb_project"], config=cfg)

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Adapter
    adapter = build_adapter(cfg, device)

    # OmniFlow pretrain bundle
    bundle_path = cfg.get("omniflow_pretrained", "./checkpoints/omniflow_pretrained")
    pipe = OmniFlowPipeline.load_pretrained(bundle_path, device=device, weight_dtype=torch.bfloat16, load_ema=False)
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    transformer.train()

    # Conditioning
    cond_drop_prob = cfg["train"].get("cond_drop_prob", 0.1)
    audio_temporal = AudioProjectorTemporal(in_dim=train_loader.dataset.args.audio_dims if hasattr(train_loader.dataset.args, 'audio_dims') else 1, token_dim=transformer.config.joint_attention_dim if hasattr(transformer, 'config') else 2048)

    # Optimizer
    opt = torch.optim.AdamW(transformer.parameters(), lr=cfg["train"]["lr"], betas=(0.9, 0.95), weight_decay=1e-4)

    # EMA
    ema = EMAModel(transformer.parameters(), decay=cfg.get("ema_momentum", 0.9999))

    # Prepare with accelerator
    transformer, opt, train_loader, val_loader, audio_temporal = accelerator.prepare(transformer, opt, train_loader, val_loader, audio_temporal)

    # Resume checkpoint
    resume_path = cfg["train"].get("resume")
    if resume_path and accelerator.is_main_process and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(transformer).load_state_dict(state["transformer"])  # type: ignore
        ema.load_state_dict(state.get("ema", {}))
        print(f"Resumed from {resume_path}")

    global_step = 0
    total_steps = cfg["train"]["epochs"] * math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    t0 = time.time()
    for epoch in range(cfg["train"]["epochs"]):
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                pose = batch["pose"].to(accelerator.device).float()
                z0 = adapter.encode(pose)
                # Build text embeddings
                pe, pp = build_prompt_embeddings(pipe, z0.shape[0], accelerator.device)
                # Multi-token audio (frame-aligned)
                if "audio" in batch:
                    aud = batch["audio"].to(accelerator.device).float()  # [B, T_aud, F]
                    T_out = cfg["train"].get("sequence_length", pose.shape[1])
                    a_seq = audio_temporal(aud, T_out=T_out)  # [B, T_out, token_dim]
                    pe = torch.cat([pe, a_seq], dim=1)
                # CFG-style dropout
                if cond_drop_prob > 0.0:
                    drop_mask = (torch.rand((pe.shape[0], 1, 1), device=accelerator.device) < cond_drop_prob)
                    pe = torch.where(drop_mask, torch.zeros_like(pe), pe)
                    pp = torch.where(drop_mask.squeeze(1), torch.zeros_like(pp), pp)

                opt.zero_grad(set_to_none=True)
                loss, aux = flow_train_step(transformer, scheduler, z0, pe, pp, loss_type="epsilon")
                accelerator.backward(loss)
                opt.step()
                ema.step(transformer.parameters())
            if accelerator.is_main_process and (global_step + 1) % cfg["train"]["log_interval"] == 0:
                elapsed = time.time() - t0
                msg = f"[flow] epoch={epoch+1} step={global_step+1}/{total_steps} loss={loss.item():.4f} t_mean={aux['t_mean']:.1f} time={elapsed:.1f}s"
                print(msg)
                if wandb.run:
                    wandb.log({"train/loss": loss.item(), "train/t_mean": aux['t_mean'], "train/step": global_step+1})
            global_step += 1
            # Smoke test early stop
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        # Validation
        if accelerator.is_main_process:
            with torch.no_grad():
                batch = next(iter(val_loader))
                pose = batch["pose"].to(accelerator.device).float()
                z0 = adapter.encode(pose)
                pe, pp = build_prompt_embeddings(pipe, z0.shape[0], accelerator.device)
                loss_val, _ = flow_train_step(transformer, scheduler, z0, pe, pp, loss_type="epsilon")
                print(f"[val] epoch={epoch+1} loss={loss_val.item():.4f}")
                if wandb.run:
                    wandb.log({"val/loss": loss_val.item(), "epoch": epoch+1})
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        # Save checkpoint
        if accelerator.is_main_process and ((epoch + 1) % cfg["train"].get("save_interval_epochs", 5) == 0):
            os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
            ck = {
                "transformer": accelerator.unwrap_model(transformer).state_dict(),  # type: ignore
                "ema": ema.state_dict(),
                "epoch": epoch+1,
                "global_step": global_step,
            }
            fp = os.path.join(cfg["logging"]["checkpoint_dir"], f"transformer_flow_e{epoch+1}.pt")
            torch.save(ck, fp)
            print(f"Saved checkpoint: {fp}")

    # Save finetuned transformer
    if accelerator.is_main_process:
        os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
        torch.save(accelerator.unwrap_model(transformer).state_dict(), os.path.join(cfg["logging"]["checkpoint_dir"], "transformer_gesture_flow.pt"))


if __name__ == "__main__":
    main()
