"""
A2G Training Script for Omniges
Audio-to-Gesture generation training using BEAT2 dataset

Based on omniflow/scripts/train.py but specialized for A2G task
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import numpy as np
from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import Omniges modules
from omniges.models.omniges_a2g import OmnigesA2GModel, A2GLoss, create_omniges_a2g_model
from omniges.dataloaders.beat_a2g_loader import create_a2g_dataloader
from utils.other_tools import AverageMeter, set_random_seed
# from utils.metric import calculate_fgd  # Will implement later


def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description='Omniges A2G Training')
    
    # Basic config
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/a2g', help='Output directory')
    parser.add_argument('--wandb_project', type=str, default='omniges-a2g', help='W&B project name')
    parser.add_argument('--dry_run', action='store_true', help='Dry run for testing')
    
    # Override config options
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, default=None, help='Override num epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_and_loss(config: dict, device: str) -> tuple:
    """Create model and loss function."""
    
    # Create model
    model = create_omniges_a2g_model(config_path=None, **{
        "dim_wavlm": config["audio"]["dim_wavlm"],
        "dim_mfcc": config["audio"]["dim_other"], 
        "dim_audio_fuse": config["audio"]["fusion"]["dim_fuse"],
        "audio_fusion_heads": config["audio"]["fusion"]["num_heads"],
        "audio_fusion_layers": config["audio"]["fusion"]["num_layers"],
        "mmdit_depth": config["backbone"]["layers"],
        "mmdit_heads": config["backbone"]["heads"],
        "rvqvae_ckpts": config["gesture"]["rvq_ckpt"],
        "device": device
    })
    
    # Create loss function
    loss_fn = A2GLoss(
        ce_weight=config["loss"]["gesture_ce_weight"],
        commitment_weight=config["loss"]["commitment"],
        velocity_weight=config["loss"]["vel_weight"],
        acceleration_weight=config["loss"]["acc_weight"]
    )
    
    return model, loss_fn


def create_optimizer_and_scheduler(model, config: dict):
    """Create optimizer and learning rate scheduler."""
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=1e-4,
        betas=(0.9, 0.95)
    )
    
    # Scheduler  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=config["train"]["lr"] * 0.01
    )
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader, 
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    config: dict,
    log_interval: int = 50
) -> dict:
    """Train for one epoch."""
    
    model.train()
    loss_meter = AverageMeter("loss")
    ce_loss_meter = AverageMeter("ce_loss")
    recon_loss_meter = AverageMeter("recon_loss")
    
    progress_bar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process
    )
    
    for step, batch in enumerate(progress_bar):
        # Prepare inputs
        audio_waveform = batch["audio_waveform"]  # (B, T_audio)
        gesture_parts = batch["gesture_parts"]    # Dict[part] -> (B, T, part_dim)
        
        # Generate random timesteps for diffusion training
        batch_size = audio_waveform.size(0)
        timesteps = torch.randint(0, 1000, (batch_size,), device=audio_waveform.device)
        
        # Encode target gestures to codes for supervision
        target_codes = {}
        with torch.no_grad():
            for part, gesture in gesture_parts.items():
                if gesture.numel() > 0:
                    # Use gesture processor to get target codes
                    # This would require the gesture processor to be accessible
                    # For now, simulate with dummy codes
                    target_codes[f"{part}_codes"] = torch.randint(
                        0, 512, gesture.shape[:2], device=gesture.device  # (B, T)
                    )
        
        # Forward pass
        with accelerator.autocast():
            outputs = model(
                audio_waveform=audio_waveform,
                timesteps=timesteps,
                target_gesture=gesture_parts,
                return_codes=True
            )
            
            # Compute loss
            loss_dict = loss_fn(
                predictions=outputs,
                targets={
                    "target_codes": target_codes,
                    "target_gesture": gesture_parts
                }
            )
            
            loss = loss_dict["total"]
        
        # Backward pass
        accelerator.backward(loss)
        
        # Update weights
        if (step + 1) % config["train"]["grad_accum"] == 0:
            # Gradient clipping
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics
        loss_meter.update(loss.item())
        if "gesture_ce" in loss_dict:
            ce_loss_meter.update(loss_dict["gesture_ce"].item())
        if "gesture_recon" in loss_dict:
            recon_loss_meter.update(loss_dict["gesture_recon"].item())
        
        # Log progress
        if step % log_interval == 0 and accelerator.is_local_main_process:
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'ce': f'{ce_loss_meter.avg:.4f}',
                'lr': f'{lr:.6f}'
            })
            
            # Log to wandb
            if not config["train"]["dry_run"]:
                wandb.log({
                    "train/loss": loss_meter.avg,
                    "train/ce_loss": ce_loss_meter.avg,
                    "train/recon_loss": recon_loss_meter.avg,
                    "train/learning_rate": lr,
                    "epoch": epoch,
                    "step": step
                })
    
    return {
        "loss": loss_meter.avg,
        "ce_loss": ce_loss_meter.avg,
        "recon_loss": recon_loss_meter.avg
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accelerator: Accelerator,
    epoch: int,
    config: dict
) -> dict:
    """Validate for one epoch."""
    
    model.eval()
    loss_meter = AverageMeter("val_loss")
    ce_loss_meter = AverageMeter("val_ce_loss")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
            audio_waveform = batch["audio_waveform"]
            gesture_parts = batch["gesture_parts"]
            
            batch_size = audio_waveform.size(0)
            timesteps = torch.randint(0, 1000, (batch_size,), device=audio_waveform.device)
            
            # Forward pass
            outputs = model(
                audio_waveform=audio_waveform,
                timesteps=timesteps,
                target_gesture=gesture_parts,
                return_codes=True
            )
            
            # Dummy target codes for validation
            target_codes = {}
            for part in gesture_parts.keys():
                target_codes[f"{part}_codes"] = torch.randint(
                    0, 512, gesture_parts[part].shape[:2], device=audio_waveform.device
                )
            
            # Compute loss
            loss_dict = loss_fn(
                predictions=outputs,
                targets={
                    "target_codes": target_codes,
                    "target_gesture": gesture_parts
                }
            )
            
            loss_meter.update(loss_dict["total"].item())
            if "gesture_ce" in loss_dict:
                ce_loss_meter.update(loss_dict["gesture_ce"].item())
    
    # Log validation metrics
    if accelerator.is_local_main_process and not config["train"]["dry_run"]:
        wandb.log({
            "val/loss": loss_meter.avg,
            "val/ce_loss": ce_loss_meter.avg,
            "epoch": epoch
        })
    
    return {
        "val_loss": loss_meter.avg,
        "val_ce_loss": ce_loss_meter.avg
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    output_dir: str,
    accelerator: Accelerator,
    is_best: bool = False
):
    """Save model checkpoint."""
    
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        
        # Unwrap model from DDP/accelerate
        unwrapped_model = accelerator.unwrap_model(model)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))
            
        logger.info(f"Saved checkpoint at epoch {epoch}")


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["train"]["lr"] = args.learning_rate
    if args.num_epochs:
        config["train"]["epochs"] = args.num_epochs
    if args.dry_run:
        config["train"]["dry_run"] = True
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if not config["train"]["dry_run"] else None,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            automatic_checkpoint_naming=True
        )
    )
    
    # Set random seed
    random_seed = config.get("seed", 42)
    set_seed(random_seed)
    
    # Initialize logging
    if accelerator.is_local_main_process and not config["train"]["dry_run"]:
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"a2g_epoch{config['train']['epochs']}_bs{config['train']['batch_size']}"
        )
    
    # Prepare dataset args from config
    class DatasetArgs:
        def __init__(self, config):
            dataset_config = config["dataset"]
            self.data_path = dataset_config["data_path"]
            self.data_path_1 = "./datasets/"
            self.cache_path = "/tmp/cache/"
            self.root_path = "./"
            self.pose_rep = dataset_config["pose_rep"]
            self.pose_fps = dataset_config["fps_pose"]
            self.audio_sr = dataset_config["sr_audio"]
            self.audio_fps = dataset_config["fps_pose"]  # Align with pose
            self.training_speakers = dataset_config["speakers"]
            self.additional_data = False
            self.disable_filtering = False
            self.clean_first_seconds = 2
            self.clean_final_seconds = 2
            self.multi_length_training = [1.0]
            self.test_length = 10
            self.stride = 10
            self.pose_length = 64  # Will be overridden by sequence_length
            self.word_rep = "textgrid"
            self.facial_rep = "expression"
            self.emo_rep = "emotion"
            self.sem_rep = "semantic"
            self.id_rep = "speaker"
            self.word_cache = False
            self.t_pre_encoder = "bert"
            self.beat_align = False
            self.new_cache = False
            self.ori_joints = "beat_smplx_full"
            self.tar_joints = "beat_smplx_full"
    
    dataset_args = DatasetArgs(config)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = create_a2g_dataloader(
        args=dataset_args,
        mode="train",
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        sequence_length=64,
        include_face=True
    )
    
    val_dataloader = create_a2g_dataloader(
        args=dataset_args,
        mode="val", 
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        sequence_length=64,
        include_face=True
    )
    
    # Create model and loss
    logger.info("Creating model...")
    model, loss_fn = create_model_and_loss(config, accelerator.device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('loss', float('inf'))
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Total epochs: {config['train']['epochs']}")
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    
    for epoch in range(start_epoch, config["train"]["epochs"]):
        
        # Training
        train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accelerator=accelerator,
            epoch=epoch,
            config=config,
            log_interval=config["train"]["log_interval"]
        )
        
        # Validation
        val_metrics = validate_epoch(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            accelerator=accelerator,
            epoch=epoch,
            config=config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch metrics
        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch {epoch+1}/{config['train']['epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
        
        # Save checkpoints
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        if (epoch + 1) % 5 == 0 or is_best:  # Save every 5 epochs or best
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_metrics['val_loss'],
                output_dir=args.output_dir,
                accelerator=accelerator,
                is_best=is_best
            )
        
        # Early exit for dry run
        if config["train"]["dry_run"] and epoch >= 2:
            logger.info("Dry run completed successfully!")
            break
    
    # Cleanup
    if accelerator.is_local_main_process and not config["train"]["dry_run"]:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
