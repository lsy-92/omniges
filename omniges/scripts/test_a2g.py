"""
A2G Testing and Evaluation Script
Generate gestures from audio and evaluate quality
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from omniges.models.omniges_a2g import create_omniges_a2g_model
from omniges.dataloaders.beat_a2g_loader import create_a2g_dataloader
from utils.metric import calculate_fgd  
from utils.fast_render import render_one_sequence


def parse_args():
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description='Omniges A2G Evaluation')
    
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./results/a2g_eval', help='Output directory')
    parser.add_argument('--render', action='store_true', help='Render gesture videos')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: str):
    """Load trained model from checkpoint."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_omniges_a2g_model(config_path=config_path)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model trained for {checkpoint['epoch']} epochs")
    
    return model, config


def generate_gestures(
    model,
    dataloader,
    device: str,
    num_samples: int = 100
) -> dict:
    """Generate gestures from audio inputs."""
    
    generated_gestures = []
    ground_truth_gestures = []
    audio_files = []
    
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating gestures"):
            if samples_processed >= num_samples:
                break
                
            audio_waveform = batch["audio_waveform"].to(device)
            gesture_parts = {k: v.to(device) for k, v in batch["gesture_parts"].items()}
            
            batch_size = audio_waveform.size(0)
            timesteps = torch.zeros(batch_size, device=device)  # No noise for inference
            
            # Generate gestures
            outputs = model(
                audio_waveform=audio_waveform,
                timesteps=timesteps,
                target_gesture=None,  # No target for inference
                return_codes=True
            )
            
            # Store results
            if "gesture_reconstructed" in outputs:
                generated_gestures.append(outputs["gesture_reconstructed"])
            ground_truth_gestures.append(gesture_parts)
            
            samples_processed += batch_size
    
    return {
        "generated": generated_gestures,
        "ground_truth": ground_truth_gestures,
        "num_samples": samples_processed
    }


def evaluate_gesture_quality(generated_data: dict, output_dir: str) -> dict:
    """Evaluate gesture generation quality."""
    
    metrics = {}
    
    generated_gestures = generated_data["generated"]
    ground_truth_gestures = generated_data["ground_truth"]
    
    logger.info("Computing gesture quality metrics...")
    
    # Compute metrics for each body part
    for part in ["upper", "hands", "lower_trans", "face"]:
        
        # Collect all generated and GT data for this part
        gen_part_data = []
        gt_part_data = []
        
        for gen_batch, gt_batch in zip(generated_gestures, ground_truth_gestures):
            if part in gen_batch and part in gt_batch:
                gen_part_data.append(gen_batch[part].cpu().numpy())
                gt_part_data.append(gt_batch[part].cpu().numpy())
        
        if gen_part_data and gt_part_data:
            # Concatenate all samples
            gen_part_all = np.concatenate(gen_part_data, axis=0)  # (N, T, D)
            gt_part_all = np.concatenate(gt_part_data, axis=0)
            
            # Flatten for metric computation
            gen_flat = gen_part_all.reshape(-1, gen_part_all.shape[-1])
            gt_flat = gt_part_all.reshape(-1, gt_part_all.shape[-1])
            
            # Compute FGD (Frechet Gesture Distance)
            try:
                fgd = calculate_fgd(gen_flat, gt_flat)
                metrics[f"{part}_fgd"] = fgd
            except Exception as e:
                logger.warning(f"Failed to compute FGD for {part}: {e}")
                metrics[f"{part}_fgd"] = float('inf')
            
            # Compute L1 diversity
            gen_std = np.std(gen_flat, axis=0).mean()
            gt_std = np.std(gt_flat, axis=0).mean()
            diversity_ratio = gen_std / (gt_std + 1e-8)
            metrics[f"{part}_diversity"] = diversity_ratio
            
            # Compute reconstruction error
            l1_error = np.mean(np.abs(gen_flat - gt_flat))
            metrics[f"{part}_l1_error"] = l1_error
    
    # Overall metrics
    all_fgd = [v for k, v in metrics.items() if "fgd" in k and v != float('inf')]
    if all_fgd:
        metrics["overall_fgd"] = np.mean(all_fgd)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return metrics


def render_sample_videos(
    generated_data: dict,
    config: dict,
    output_dir: str,
    num_videos: int = 5
):
    """Render sample gesture videos for qualitative evaluation."""
    
    if not config["eval"]["render"]:
        logger.info("Rendering disabled in config")
        return
    
    render_dir = os.path.join(output_dir, "rendered_videos")
    os.makedirs(render_dir, exist_ok=True)
    
    generated_gestures = generated_data["generated"]
    ground_truth_gestures = generated_data["ground_truth"]
    
    logger.info(f"Rendering {num_videos} sample videos...")
    
    videos_rendered = 0
    for batch_idx, (gen_batch, gt_batch) in enumerate(zip(generated_gestures, ground_truth_gestures)):
        if videos_rendered >= num_videos:
            break
            
        batch_size = list(gen_batch.values())[0].size(0)
        
        for sample_idx in range(min(batch_size, num_videos - videos_rendered)):
            # Extract single sample
            gen_sample = {k: v[sample_idx].cpu().numpy() for k, v in gen_batch.items()}
            gt_sample = {k: v[sample_idx].cpu().numpy() for k, v in gt_batch.items()}
            
            # Save as numpy files for rendering
            gen_path = os.path.join(render_dir, f"generated_{videos_rendered:03d}.npz")
            gt_path = os.path.join(render_dir, f"ground_truth_{videos_rendered:03d}.npz")
            
            # Convert back to SMPLX format for rendering
            gen_smplx = gesture_parts_to_smplx(gen_sample)
            gt_smplx = gesture_parts_to_smplx(gt_sample)
            
            np.savez(gen_path, **gen_smplx)
            np.savez(gt_path, **gt_smplx)
            
            # Render video if utility available
            try:
                render_one_sequence(
                    res_npz_path=gen_path,
                    gt_npz_path=gt_path,
                    output_dir=f"{render_dir}/video_{videos_rendered:03d}/",
                    audio_path=None,  # No audio file for now
                    use_matplotlib=False
                )
                logger.info(f"Rendered video {videos_rendered}")
            except Exception as e:
                logger.warning(f"Failed to render video {videos_rendered}: {e}")
            
            videos_rendered += 1
            
            if videos_rendered >= num_videos:
                break


def gesture_parts_to_smplx(gesture_parts: dict) -> dict:
    """Convert gesture parts back to SMPLX format for rendering."""
    
    # This is a simplified conversion - needs to be refined based on actual BEAT data structure
    smplx_data = {
        "poses": np.zeros((gesture_parts[list(gesture_parts.keys())[0]].shape[0], 165)),  # SMPLX pose params
        "trans": np.zeros((gesture_parts[list(gesture_parts.keys())[0]].shape[0], 3)),
        "betas": np.zeros(300),  # Default body shape
        "expressions": np.zeros((gesture_parts[list(gesture_parts.keys())[0]].shape[0], 100))
    }
    
    # Map body parts back to SMPLX format
    pose_idx = 0
    
    if "upper" in gesture_parts:
        upper_data = gesture_parts["upper"]  # (T, 78)
        smplx_data["poses"][:, pose_idx:pose_idx+78] = upper_data
        pose_idx += 78
    
    if "hands" in gesture_parts:
        hands_data = gesture_parts["hands"]  # (T, 180)
        # Map to SMPLX hand pose parameters
        smplx_data["poses"][:, 25*3:55*3] = hands_data  # Approximate mapping
    
    if "lower_trans" in gesture_parts:
        lower_trans_data = gesture_parts["lower_trans"]  # (T, 57)
        # Split into lower body pose and translation
        if lower_trans_data.shape[-1] >= 3:
            smplx_data["trans"] = lower_trans_data[:, -3:]  # Last 3 dims as translation
        if lower_trans_data.shape[-1] > 3:
            smplx_data["poses"][:, :24] = lower_trans_data[:, :24]  # Lower body poses
    
    if "face" in gesture_parts:
        face_data = gesture_parts["face"]  # (T, 103)
        if face_data.shape[-1] >= 100:
            smplx_data["expressions"] = face_data[:, -100:]  # Expression coefficients
        if face_data.shape[-1] >= 3:
            smplx_data["poses"][:, 66:69] = face_data[:, :3]  # Jaw pose
    
    return smplx_data


def main():
    """Main evaluation function."""
    
    args = parse_args()
    
    # Load model and config
    model, config = load_model(args.config, args.checkpoint, args.device)
    
    # Prepare dataset args  
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
            self.audio_fps = dataset_config["fps_pose"]
            self.training_speakers = dataset_config["speakers"]
            self.additional_data = False
            self.disable_filtering = False
            self.clean_first_seconds = 2
            self.clean_final_seconds = 2
            self.multi_length_training = [1.0]
            self.test_length = 10
            self.stride = 10
            self.pose_length = 64
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
    
    # Create test dataloader
    test_dataloader = create_a2g_dataloader(
        args=dataset_args,
        mode="test",
        batch_size=4,
        num_workers=2,
        sequence_length=64,
        include_face=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate gestures
    logger.info("Generating gestures from audio...")
    generated_data = generate_gestures(
        model=model,
        dataloader=test_dataloader, 
        device=args.device,
        num_samples=args.num_samples
    )
    
    # Evaluate quality
    logger.info("Evaluating gesture quality...")
    metrics = evaluate_gesture_quality(generated_data, args.output_dir)
    
    # Render videos
    if args.render:
        logger.info("Rendering sample videos...")
        render_sample_videos(generated_data, config, args.output_dir)
    
    # Save summary
    summary = {
        "config_path": args.config,
        "checkpoint_path": args.checkpoint, 
        "num_samples_evaluated": generated_data["num_samples"],
        "metrics": metrics
    }
    
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation completed! Results saved to {args.output_dir}")
    logger.info(f"Overall FGD: {metrics.get('overall_fgd', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
