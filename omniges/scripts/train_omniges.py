#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""
Omniges Training Script
Complete multi-task training for Text-Audio-Gesture generation
Based on OmniFlow + RVQVAE gesture processing
Supports: t2g, g2t, a2g, g2a, t2a, a2t
"""

import argparse
import copy
import gc
import time
from safetensors import safe_open
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Omniges components
from omniges.models import OmnigesFlowTransformerModel, GestureProcessor
from omniges.pipelines import OmnigesPipeline, OmnigesGestureVAE

# Import OmniFlow components
from omniflow.utils.ema import EMAModel
import torch.utils.data
from transformers.trainer_pt_utils import LabelSmoother
import itertools
import logging
import math
import random
import shutil
import warnings
from contextlib import nullcontext
import pandas as pd
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast, CLIPImageProcessor
import torch.nn.functional as F
import diffusers
from diffusers import AutoencoderKL
from omniflow.utils.scheduler import OmniFlowMatchEulerDiscreteScheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import torch.distributed as dist
import glob
from omniflow.models.audio_vae import load_audio_vae
from omniflow.utils.text_encode import encode_prompt_train, cat_and_pad, encode_prompt_for_decoder

if is_wandb_available():
    import wandb
from torch import nn
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

# Validation files for testing
VAL_FILES = ['./assets/girl.png']
VAL_FILES_AUDIO = ['./assets/car engine.mp3']

from omniflow.models.text_vae import LLamaForLatentConnector
from omniflow.models.encoders import LanguageBindAudioProcessor, LanguageBindAudio
import yaml
from transformers import AutoTokenizer, AutoConfig

# Import BEAT data processing
from dataloaders.beat_sep_lower import CustomDataset
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc


def load_yaml(fp: str):
    with open(fp, 'r') as file:
        data = yaml.safe_load(file)
    return data


def n_get_sigmas(noise_scheduler_copy, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def n_compute_text_embeddings(device, prompt, text_encoders, tokenizers, add_token_embed=True, train=False):
    print(f"DEBUG: [n_compute_text_embeddings] Input prompts count: {len(prompt)}")
    print(f"DEBUG: [n_compute_text_embeddings] Sample prompt: {prompt[0] if prompt else 'No prompts'}")
    print(f"DEBUG: [n_compute_text_embeddings] add_token_embed: {add_token_embed}, train: {train}")
    
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt_train(
            text_encoders, tokenizers, prompt, 256, add_token_embed=add_token_embed,
            normalize=True, drops=list(
                np.random.rand() > 0.5 for _ in range(4)
            ) if train else [False, False, False, False]
        )
        print(f"DEBUG: [n_compute_text_embeddings] Raw prompt_embeds shape: {prompt_embeds.shape}")
        print(f"DEBUG: [n_compute_text_embeddings] Raw pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
        
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        
        print(f"DEBUG: [n_compute_text_embeddings] Final prompt_embeds shape: {prompt_embeds.shape}")
        print(f"DEBUG: [n_compute_text_embeddings] Final pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
    return prompt_embeds, pooled_prompt_embeds


class OmnigesDataset(Dataset):
    """
    Omniges Dataset combining BEAT gesture data with text/audio
    Supports all task combinations: t2g, g2t, a2g, g2a, t2a, a2t
    """

    def __init__(
        self,
        beat_config_path="configs/shortcut_rvqvae_128.yaml",
        task_weights=[1/6] * 6,  # Equal weight for all 6 tasks
        size=512,
        is_train=True,
        image_processor=None,
        audio_processor=None,
        audio_processor_clip=None,
    ):
        self.size = size
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.audio_processor_clip = audio_processor_clip
        self.task_weights = task_weights
        self.is_train = is_train
        
        # Load BEAT config and create dataset
        with open(beat_config_path, 'r') as f:
            beat_config = yaml.safe_load(f)
            
        # Create args object from BEAT config
        class BeatArgs:
            def __init__(self, config):
                for key, value in config.items():
                    setattr(self, key, value)
                # Add ALL missing attributes for BEAT dataset
                self.multi_length_training = [1.0]
                self.beat_align = False
                self.word_cache = False          # Fix: word_cache 속성 추가
                self.facial_cache = False        # Fix: facial_cache 속성 추가  
                self.audio_cache = False         # Fix: audio_cache 속성 추가
                self.pose_cache = False          # Fix: pose_cache 속성 추가
                self.trans_cache = False         # Fix: trans_cache 속성 추가
                self.speaker_cache = False       # Fix: speaker_cache 속성 추가
                self.emotion_cache = False       # Fix: emotion_cache 속성 추가
                self.semantic_cache = False      # Fix: semantic_cache 속성 추가
                
        self.beat_args = BeatArgs(beat_config)
        
        # Create BEAT dataset
        self.beat_dataset = CustomDataset(
            self.beat_args, 
            loader_type="train" if is_train else "test",
            build_cache=True
        )
        
        # Task combinations
        self.tasks = ['t2g', 'g2t', 'a2g', 'g2a', 't2a', 'a2t']
        
        # Generate text prompts for gesture tasks
        self.gesture_prompts = [
            "A person waving hello",
            "Someone clapping their hands", 
            "A person pointing forward",
            "Dancing with arm movements",
            "Gesturing while speaking",
            "Hand gestures during conversation",
            "Expressive body language",
            "Animated talking with hands",
            "Conducting orchestra movements",
            "Sign language communication"
        ]

    def __len__(self):
        return len(self.beat_dataset)

    def __getitem__(self, index):
        # Get BEAT data
        beat_item = self.beat_dataset[index]
        
        # Random task selection
        task = np.random.choice(self.tasks, p=self.task_weights)
        
        # Process based on task
        if task in ['t2g', 'g2t']:
            # Text-Gesture tasks
            prompt = np.random.choice(self.gesture_prompts)
            prompt2 = prompt  # Same prompt for both encoders
            has_text = True
            has_gesture = True
            has_audio = False
            
        elif task in ['a2g', 'g2a']:
            # Audio-Gesture tasks
            prompt = ""  # No text for pure audio-gesture
            prompt2 = ""
            has_text = False
            has_gesture = True
            has_audio = True
            
        elif task == 't2a':
            # Text-Audio task (from OmniFlow)
            prompt = "Music playing"  # Audio description
            prompt2 = prompt
            has_text = True
            has_gesture = False
            has_audio = True
            
        elif task == 'a2t':
            # Audio-Text task (from OmniFlow)  
            prompt = ""  # Generated from audio
            prompt2 = ""
            has_text = True
            has_gesture = False
            has_audio = True
            
        # Process gesture data
        pose = beat_item['pose']       # (T, pose_dim)
        facial = beat_item['facial']   # (T, 100)
        trans = beat_item['trans']     # (T, 3)
        trans_v = beat_item['trans_v'] # (T, 3)
        audio = beat_item['audio']     # (T_audio,)
        
        # Convert to batch format
        gesture_sequence = self._process_gesture_data(pose, facial, trans, trans_v)
        
        # Process audio for audio VAE
        audio_vae_input = torch.zeros(1, 1, 1024, 64)  # Default
        audio_clip_input = torch.zeros(1, 3, 112, 1036)  # Default
        
        if has_audio and hasattr(beat_item, 'audio_name'):
            try:
                audio_path = beat_item.get('audio_name', '')
                if audio_path and os.path.exists(audio_path):
                    x = self.audio_processor.feature_extraction_vae(audio_path)
                    audio_vae_input = x['fbank'].unsqueeze(0)
                    audio_clip_input = self.audio_processor_clip([audio_path])['pixel_values']
                else:
                    # Use dummy audio
                    pass
            except Exception as e:
                logger.warning(f"Audio processing failed: {e}")
                
        # Create dummy image for compatibility (will be replaced by gesture)
        dummy_image = torch.zeros(3, self.size, self.size)
        dummy_image_clip = torch.zeros(1, 3, 224, 224)
        
        return {
            'gesture_sequence': gesture_sequence,    # Our new gesture data
            'image': dummy_image,                   # Dummy for compatibility
            'image_clip': dummy_image_clip,         # Dummy for compatibility  
            'caption': prompt,                      # Text prompt
            'caption2': prompt2,                    # Text prompt 2
            'audio': audio_vae_input,              # Audio for audio VAE
            'audio_clip': audio_clip_input,        # Audio for audio encoder
            'task': task,                          # Task type
            'has_gesture': has_gesture,            # Gesture availability
            'has_image': False,                    # Always False (replaced by gesture)
            'has_audio': has_audio,                # Audio availability
            'has_caption': has_text,               # Text availability
            'dataset': f'gesture_{task}',          # Dataset identifier
            'weight': [1.0, 1.0]                  # Task weight
        }
        
    def _process_gesture_data(self, pose, facial, trans, trans_v):
        """
        Process gesture data using proven adaptive method
        """
        # Add batch dimension
        pose = pose.unsqueeze(0)      # (1, T, pose_dim)
        facial = facial.unsqueeze(0)  # (1, T, 100)
        trans = trans.unsqueeze(0)    # (1, T, 3)
        trans_v = trans_v.unsqueeze(0) # (1, T, 3)
        
        B, T, pose_dim = pose.shape
        
        # Use proven adaptive split method
        upper_end = int(pose_dim * 0.4)
        hands_start = upper_end
        hands_end = int(pose_dim * 0.8)
        
        upper_pose = pose[:, :, :upper_end]
        hands_pose = pose[:, :, hands_start:hands_end]
        lower_pose = pose[:, :, hands_end:]
        
        # Combine lower with translation
        lower_trans = torch.cat([lower_pose, trans_v], dim=-1)
        
        # Pad to exact RVQVAE requirements
        upper_pose = F.pad(upper_pose, (0, max(0, 78 - upper_pose.shape[-1])))[:, :, :78]
        hands_pose = F.pad(hands_pose, (0, max(0, 180 - hands_pose.shape[-1])))[:, :, :180]
        lower_trans = F.pad(lower_trans, (0, max(0, 57 - lower_trans.shape[-1])))[:, :, :57]
        face_data = F.pad(facial, (0, max(0, 100 - facial.shape[-1])))[:, :, :100]
        
        # Combine all parts
        full_gesture = torch.cat([
            upper_pose,      # 78
            hands_pose,      # 180
            lower_trans,     # 57
            face_data        # 100
        ], dim=-1)  # (1, T, 415)
        
        return full_gesture.squeeze(0)  # (T, 415)


def omniges_collate_fn(examples):
    """
    Collate function adapted for Omniges
    """
    # Get gesture data instead of images
    gesture_sequences = [example["gesture_sequence"] for example in examples]
    gesture_sequences = torch.stack([torch.nn.functional.pad(seq, (0, 0, 0, 128 - seq.shape[0])) for seq in gesture_sequences])  # Pad to 128 frames
    
    # Keep dummy images for compatibility with OmniFlow logic
    pixel_values = [example["image"] for example in examples]
    clip_values = torch.cat([example["image_clip"] for example in examples])
    
    prompts = list([example["caption"] for example in examples])
    prompts2 = list([example["caption2"] for example in examples])

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    audio = torch.cat([example["audio"] for example in examples])
    audio_clip = torch.cat([example["audio_clip"] for example in examples])
    
    # Determine task from first example
    task_type = examples[0]['task']
    
    # Map Omniges tasks to OmniFlow-compatible names for processing
    task_mapping = {
        't2g': 'text2img',   # Text to Gesture -> Text to Image (internal)
        'g2t': 'img2text',   # Gesture to Text -> Image to Text (internal)  
        'a2g': 'aud2img',    # Audio to Gesture -> Audio to Image (internal)
        'g2a': 'img2aud',    # Gesture to Audio -> Image to Audio (internal)
        't2a': 'text2aud',   # Text to Audio (same)
        'a2t': 'aud2text'    # Audio to Text (same)
    }
    
    task = task_mapping[task_type]
    
    # Dropout logic adapted for gesture tasks
    drop_img = drop_text = drop_aud = None
    
    if task in ['text2img', 'text2aud']:  # t2g, t2a
        drop_text = (np.random.rand(len(prompts)) < 0.15).nonzero()[0]
    elif task in ['img2text', 'img2aud']:  # g2t, g2a  
        drop_img = (np.random.rand(len(prompts)) < 0.15).nonzero()[0]
    elif task in ['aud2text', 'aud2img']:  # a2t, a2g
        drop_aud = (np.random.rand(len(prompts)) < 0.15).nonzero()[0]
    
    batch = {
        "pixel_values": pixel_values,      # Dummy images (replaced by gesture_sequences)
        "gesture_sequences": gesture_sequences,  # NEW: Actual gesture data
        "prompts": prompts,
        "prompts2": prompts2,
        "task": task,
        "task_type": task_type,           # NEW: Original task type
        "clip_values": clip_values,
        "audio": audio,
        "audio_clip": audio_clip,
        'drop_img': drop_img,
        'drop_aud': drop_aud, 
        "drop_text": drop_text,
        "name": examples[0]['dataset'],
    }

    return batch


def prepare_omniges_inputs(
    transformer, args, text_encoder_one, text_encoder_two, text_encoder_three,
    device, batch, gesture_vae, tokenizer_three, text_encoders, tokenizers,
    tokenizer_one, tokenizer_two, weight_dtype, noise_scheduler_copy,
    noise_scheduler, audio_vae_factor, audiovae, text_vae_tokenizer,
    text_vae, audio_encoder, anchor=False, mm_encoder=None,
):
    """
    Prepare inputs for Omniges training
    Adapted from OmniFlow prepare_inputs with gesture processing
    """
    with torch.no_grad():
        models_to_accumulate = [transformer]

        task = batch['task']
        task_type = batch['task_type']  # Original omniges task
        
        # Process gesture data instead of image data
        gesture_sequences = batch["gesture_sequences"]  # (B, T, 415)
        prompts = np.array(batch["prompts"])
        prompts2 = np.array(batch["prompts2"])
        
        print(f"DEBUG: ========== DATALOADER OUTPUT ==========")
        print(f"DEBUG: task: {task}, task_type: {task_type}")
        print(f"DEBUG: gesture_sequences shape: {gesture_sequences.shape}")
        print(f"DEBUG: prompts count: {len(prompts)}")
        print(f"DEBUG: prompts2 count: {len(prompts2)}")
        if 'audio' in batch:
            print(f"DEBUG: audio shape: {batch['audio'].shape}")
        else:
            print(f"DEBUG: No audio in batch")
    
        bsz = len(prompts)
        
        # Encode gesture to latents using gesture VAE
        print(f"DEBUG: ========== GESTURE VAE ENCODING ==========")
        gesture_latents_dist = gesture_vae.encode(gesture_sequences.to(device))
        model_input = gesture_latents_dist.sample()  # (B, C, H, W) format
        print(f"DEBUG: gesture VAE output shape: {model_input.shape}")
        
        # The GestureVAE should now output (B, 512, T, 1) directly - no reshaping needed
        B, C, H, W = model_input.shape
        print(f"DEBUG: Gesture VAE output - B:{B}, C:{C}, H:{H}, W:{W}")
        
        # Verify expected format
        if C == 512 and W == 1:
            print(f"DEBUG: ✓ Correct format (B, 512, T, 1) - 4 parts concatenated")
        elif C == 128 and W == 4:
            print(f"DEBUG: ⚠ Legacy format (B, 128, T, 4) - converting to concat format")
            # Convert (B, 128, T, 4) -> (B, T, 128, 4) -> (B, T, 512) -> (B, 512, T, 1)
            model_input = model_input.permute(0, 2, 1, 3)  # (B, T, 128, 4)
            model_input = model_input.reshape(B, H, C * W)  # (B, T, 512) 
            model_input = model_input.permute(0, 2, 1).unsqueeze(-1)  # (B, 512, T, 1)
            print(f"DEBUG: After conversion - model_input shape: {model_input.shape}")
        else:
            print(f"DEBUG: ⚠ Unexpected format - C:{C}, W:{W}")
            print(f"DEBUG: Current model_input shape: {model_input.shape}")
            
        model_input = model_input * gesture_vae.config.scaling_factor
        model_input = model_input.to(dtype=weight_dtype)
        print(f"DEBUG: After scaling and dtype - model_input shape: {model_input.shape}")

        # Audio input (same as OmniFlow)
        print(f"DEBUG: ========== AUDIO VAE ENCODING ==========")
        raw_audio_embeds = batch['audio'].to(model_input.device)
        print(f"DEBUG: Raw audio input shape: {raw_audio_embeds.shape}")
        print(f"DEBUG: Raw audio input dtype: {raw_audio_embeds.dtype}")
        print(f"DEBUG: Raw audio input device: {raw_audio_embeds.device}")
        
        # Audio VAE encoding
        audio_latent_dist = audiovae.encode(raw_audio_embeds.float())
        print(f"DEBUG: Audio VAE latent_dist type: {type(audio_latent_dist)}")
        
        # Handle AutoencoderKLOutput correctly
        if hasattr(audio_latent_dist, 'latents'):
            # Most common case for AutoencoderKLOutput
            raw_audio_embeds = audio_latent_dist.latents
            print(f"DEBUG: Audio VAE latents shape (via .latents): {raw_audio_embeds.shape}")
        elif hasattr(audio_latent_dist, 'latent_dist'):
            raw_audio_embeds = audio_latent_dist.latent_dist.sample()
            print(f"DEBUG: Audio VAE sample shape (via latent_dist): {raw_audio_embeds.shape}")
        elif hasattr(audio_latent_dist, 'sample'):
            raw_audio_embeds = audio_latent_dist.sample()
            print(f"DEBUG: Audio VAE sample shape (direct): {raw_audio_embeds.shape}")
        else:
            # Fallback - assume it's already the latent tensor
            raw_audio_embeds = audio_latent_dist
            print(f"DEBUG: Audio VAE direct tensor shape: {raw_audio_embeds.shape}")
            
        print(f"DEBUG: Audio latent attributes: {[attr for attr in dir(audio_latent_dist) if not attr.startswith('_')]}")
        
        # Apply scaling factor
        raw_audio_embeds = raw_audio_embeds.mul_(audiovae.config.scaling_factor)
        print(f"DEBUG: Audio VAE scaling factor: {audiovae.config.scaling_factor}")
        print(f"DEBUG: After scaling - audio shape: {raw_audio_embeds.shape}")
        
        raw_audio_embeds = raw_audio_embeds.to(model_input)
        print(f"DEBUG: Final audio embeds shape: {raw_audio_embeds.shape}")
        print(f"DEBUG: Final audio embeds dtype: {raw_audio_embeds.dtype}")
        
        # Sample noise for different modalities
        bsz = model_input.shape[0]
        add_token_embed = True
        
        # Sample timesteps for 3 modalities
        print(f"DEBUG: ========== TIMESTEP SAMPLING ==========")
        print(f"DEBUG: batch size: {bsz}, total timesteps to sample: {bsz * 3}")
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz * 3,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        if args.uniform_flow:
            indices = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz*3,), device='cpu', dtype=torch.long
            )
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        print(f"DEBUG: Raw timesteps shape: {timesteps.shape}")
        timesteps, timesteps_text, timesteps_audio = timesteps.chunk(3)
        print(f"DEBUG: Split timesteps - gesture: {timesteps.shape}, text: {timesteps_text.shape}, audio: {timesteps_audio.shape}")
        
        # Get sigmas for each modality
        sigmas = n_get_sigmas(noise_scheduler_copy, device, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        sigma_text = n_get_sigmas(noise_scheduler_copy, device, timesteps_text, n_dim=model_input.ndim, dtype=model_input.dtype)
        sigmas_audio = n_get_sigmas(noise_scheduler_copy, device, timesteps_audio, n_dim=model_input.ndim, dtype=model_input.dtype)
        print(f"DEBUG: Sigmas shapes - gesture: {sigmas.shape}, text: {sigma_text.shape}, audio: {sigmas_audio.shape}")
        
        # Loss factors for different tasks
        loss_text_factor = 1
        loss_aud_factor = 1
        loss_gesture_factor = 1  # Renamed from loss_img_factor
        
        # Set up proper conditioning based on task
        can_generate_text = True
        if np.random.rand() < 0.1:
            can_generate_text = False
            
        # Task-specific conditioning (adapted for gesture)
        if task in ['text2img', 'text2aud']:  # t2g, t2a
            loss_text_factor = 0
            if np.random.rand() < 0.8:
                sigma_text = sigma_text * 0
                timesteps_text = timesteps_text * 0
        
        if task in ['img2aud', 'aud2img']:  # g2a, a2g
            loss_text_factor = 0
            sigma_text = sigma_text * 0 + 1
            timesteps_text = timesteps_text * 0 + 1000
            
        if batch['drop_text'] is not None:
            timesteps_text[batch['drop_text']] = 1000
            sigma_text[batch['drop_text']] = 1
        
        if batch['drop_aud'] is not None:
            timesteps_audio[batch['drop_aud']] = 1000
            sigmas_audio[batch['drop_aud']] = 1
            
        if batch['drop_img'] is not None:  # This now refers to gesture
            timesteps[batch['drop_img']] = 1000
            sigmas[batch['drop_img']] = 1
            
        if task in ['img2text', 'img2aud']:  # g2t, g2a
            loss_gesture_factor = 0
            if np.random.rand() < 0.8:
                sigmas = sigmas * 0
                timesteps = timesteps * 0
                
        if task in ['text2aud', 'aud2text']:  # t2a, a2t
            loss_gesture_factor = 0
            sigmas = sigmas * 0 + 1
            timesteps = timesteps * 0 + 1000
              
        if task in ['aud2text', 'aud2img']:  # a2t, a2g
            loss_aud_factor = 0
            if np.random.rand() < 0.8:
                sigmas_audio = sigmas_audio * 0
                timesteps_audio = timesteps_audio * 0
            
        if task in ['text2img', 'img2text']:  # t2g, g2t
            loss_aud_factor = 0
            sigmas_audio = sigmas_audio * 0 + 1
            timesteps_audio = timesteps_audio * 0 + 1000
        
        # Pooling mode determination
        if task in ['img2text', 'img2aud']:  # g2t, g2a
            pool_mode = 'gesture'  # Use gesture embeddings
        elif task in ['aud2img', 'aud2text']:  # a2g, a2t
            pool_mode = 'aud'
        else:
            pool_mode = 'text'
            
        if not can_generate_text:
            loss_text_factor = loss_text_factor * 0

        # Text encoding (same as OmniFlow)
        print(f"DEBUG: ========== TEXT ENCODING ==========")
        prompts = prompts.tolist()
        target_labels = tokenize_prompt(tokenizer_three, prompts)
        target_labels = target_labels.to(device)
        print(f"DEBUG: target_labels shape: {target_labels.shape}")
        
        prompt_embeds, pooled_prompt_embeds = n_compute_text_embeddings(
            device, prompts, text_encoders, tokenizers, add_token_embed=add_token_embed, train=False
        )
        print(f"DEBUG: prompt_embeds shape: {prompt_embeds.shape}")
        print(f"DEBUG: pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
        
        print(f"DEBUG: ========== TEXT VAE ENCODING ==========")
        prompt_embeds_vae = text_vae.encode(prompts, input_ids=None, tokenizer=tokenizer_three, sample=True)
        prompt_embeds_vae_uncond = text_vae.encode(prompts, input_ids=None, tokenizer=tokenizer_three, drop=True)
        print(f"DEBUG: prompt_embeds_vae shape: {prompt_embeds_vae.shape}")
        print(f"DEBUG: prompt_embeds_vae_uncond shape: {prompt_embeds_vae_uncond.shape}")

        if not can_generate_text:
            prompt_embeds_vae *= 0
            print(f"DEBUG: Text generation disabled - zeroed VAE embeddings")

        l_vae = prompt_embeds_vae.shape[1]
        print(f"DEBUG: l_vae (VAE sequence length): {l_vae}")
        
        # Prepare prompt embeddings 
        print(f"DEBUG: ========== TEXT EMBEDDINGS PREPARATION ==========")
        prompt_embeds = cat_and_pad([prompt_embeds_vae], max_dim=4096)
        prompt_embeds_uncond = cat_and_pad([prompt_embeds_vae_uncond], max_dim=4096)
        print(f"DEBUG: After cat_and_pad - prompt_embeds shape: {prompt_embeds.shape}")
        print(f"DEBUG: After cat_and_pad - prompt_embeds_uncond shape: {prompt_embeds_uncond.shape}")

        # Targets for text decoder
        targets = encode_prompt_for_decoder(prompts, text_vae_tokenizer, device=transformer.device)
        target_labels = targets['labels']
        print(f"DEBUG: Text decoder targets shape: {target_labels.shape}")
        print(f"DEBUG: Target labels sample: {targets.keys()}")

        # Pooled embeddings based on pool mode
        with torch.no_grad():
            if pool_mode == 'gesture':
                # Use gesture embeddings (dummy for now)
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
                if batch['drop_img'] is not None:
                    pooled_prompt_embeds[batch['drop_img']] = 0
            elif pool_mode == 'aud':
                audio_embeds = audio_encoder.get_image_features(
                    pixel_values=batch['audio_clip'].to(audio_encoder.dtype).to(audio_encoder.device)
                )
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
                pooled_prompt_embeds[..., :audio_embeds.shape[-1]] = audio_embeds
                if batch['drop_aud'] is not None:
                    pooled_prompt_embeds[batch['drop_aud']] = 0
            else:
                if batch['drop_text'] is not None:
                    pooled_prompt_embeds[batch['drop_text']] = 0
                    
        pooled_prompt_embeds = pooled_prompt_embeds.detach()
        
        # Apply dropout to pooled embeddings
        drop_pool = (torch.rand(pooled_prompt_embeds.shape[0]) < 0.85).view(-1, 1).to(pooled_prompt_embeds)
        pooled_prompt_embeds = pooled_prompt_embeds * drop_pool
        
        # Reshape sigma_text
        sigma_text = sigma_text.view(-1, 1, 1)
        
        # Generate noise
        print(f"DEBUG: ========== NOISE GENERATION ==========")
        noise = torch.randn_like(model_input)
        noise_text = torch.randn_like(prompt_embeds)
        print(f"DEBUG: noise shape: {noise.shape}")
        print(f"DEBUG: noise_text shape: {noise_text.shape}")
        
        # Add noise to inputs
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
        noisy_prompt_embeds = sigma_text * noise_text + (1.0 - sigma_text) * prompt_embeds 
        print(f"DEBUG: noisy_model_input shape: {noisy_model_input.shape}")
        print(f"DEBUG: noisy_prompt_embeds shape: {noisy_prompt_embeds.shape}")

        noise_audio = torch.randn_like(raw_audio_embeds)
        sigmas_audio = sigmas_audio.view(-1, 1, 1, 1)
        noisy_audio_embeds = sigmas_audio * noise_audio + (1.0 - sigmas_audio) * raw_audio_embeds
        print(f"DEBUG: noise_audio shape: {noise_audio.shape}")
        print(f"DEBUG: noisy_audio_embeds shape: {noisy_audio_embeds.shape}") 

        # Clean up text embeddings
        noisy_prompt_embeds[:, -l_vae:, prompt_embeds_vae.shape[-1]:] = 0
        noisy_prompt_embeds = noisy_prompt_embeds.detach()
        
        return (
            noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,
            noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,
            sigmas, sigmas_audio, model_input,
            loss_gesture_factor,  # Renamed from loss_img_factor
            loss_text_factor,
            loss_aud_factor,
            noise_scheduler_copy,
            raw_audio_embeds,
            task, task_type,  # Include both task names
            prompts,
            noise,
            noise_text,
            noise_audio,
            target_labels,
            prompt_embeds_vae_uncond,
            gesture_sequences  # NEW: Add gesture sequences
        )


def compute_omniges_loss(
    transformer, noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,
    noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,
    sigmas, sigmas_audio, model_input, loss_gesture_factor, loss_text_factor, loss_aud_factor,
    noise_scheduler_copy, last_lr, raw_audio_embeds, task, task_type, prompts,
    noise, noise_text, noise_audio, text_vae, target_labels, do_decode,
    prompt_embeds_vae_uncond, precondition_text_outputs=False, anchor=False, batch=None,
    gesture_sequences=None
):
    """
    Compute loss for Omniges training
    Adapted from OmniFlow with gesture processing
    """
    
    # Forward pass through OmnigesFlow
    print(f"DEBUG: ========== MODEL FORWARD PASS ==========")
    print(f"DEBUG: Forward pass inputs:")
    print(f"DEBUG:   noisy_model_input shape: {noisy_model_input.shape}")
    print(f"DEBUG:   timesteps shape: {timesteps.shape}")
    print(f"DEBUG:   timesteps_text shape: {timesteps_text.shape}")
    print(f"DEBUG:   timesteps_audio shape: {timesteps_audio.shape}")
    print(f"DEBUG:   noisy_prompt_embeds shape: {noisy_prompt_embeds.shape}")
    print(f"DEBUG:   noisy_audio_embeds shape: {noisy_audio_embeds.shape}")
    print(f"DEBUG:   pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
    
    output_dict = transformer(
        hidden_states=noisy_model_input,              # Gesture latents
        timestep=timesteps,                           # Gesture timestep
        timestep_text=timesteps_text,                 # Text timestep
        timestep_audio=timesteps_audio,               # Audio timestep
        encoder_hidden_states=noisy_prompt_embeds,    # Text embeddings
        audio_hidden_states=noisy_audio_embeds,       # Audio embeddings
        sigma_text=sigma_text,
        target_prompt_embeds=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        targets=targets,
        return_dict=False,
        use_text_output=True,
        prompt_embeds_uncond=None if np.random.rand() < 0.5 else prompt_embeds_uncond,
        detach_logits=not anchor,
        split_cond=False,
        text_vae=text_vae,
        text_x0=precondition_text_outputs,
        decode_text=True,
        # Task-specific dropout logic
        # For input modalities: never drop
        # For output modalities: use batch dropout settings
        drop_gesture=(task in ['text2img', 'aud2img'] and batch['drop_img'] is not None),  # Only drop gesture for T2G, A2G tasks
        drop_text=(task in ['img2text', 'aud2text'] and batch['drop_text'] is not None),   # Only drop text for G2T, A2T tasks
        drop_audio=(task in ['text2aud', 'img2aud'] and batch['drop_aud'] is not None)     # Only drop audio for T2A, G2A tasks
    )
    
    # Extract predictions
    print(f"DEBUG: ========== MODEL OUTPUT ==========")
    model_pred = output_dict['output']              # Gesture output
    model_pred_audio = output_dict['audio_hidden_states']  # Audio output
    model_pred_text = output_dict['model_pred_text']       # Text output
    logits = output_dict['logits']
    logits_labels = output_dict['logits_labels']
    
    print(f"DEBUG: Model outputs:")
    print(f"DEBUG:   model_pred shape: {model_pred.shape if model_pred is not None else None}")
    print(f"DEBUG:   model_pred_audio shape: {model_pred_audio.shape if model_pred_audio is not None else None}")
    print(f"DEBUG:   model_pred_text shape: {model_pred_text.shape if model_pred_text is not None else None}")
    print(f"DEBUG:   logits shape: {logits.shape if logits is not None else None}")
    print(f"DEBUG:   logits_labels shape: {logits_labels.shape if logits_labels is not None else None}")
    
    # Compute velocity targets
    print(f"DEBUG: ========== VELOCITY TARGETS ==========")
    v_theta = noise - model_input                    # Gesture velocity
    v_theta_audio = noise_audio - raw_audio_embeds   # Audio velocity
    print(f"DEBUG: v_theta shape: {v_theta.shape}")
    print(f"DEBUG: v_theta_audio shape: {v_theta_audio.shape}")
    
    print(f"DEBUG: Loss input comparison:")
    print(f"DEBUG:   model_pred shape: {model_pred.shape if model_pred is not None else None}")
    print(f"DEBUG:   v_theta shape: {v_theta.shape}")
    print(f"DEBUG:   Are shapes compatible? {model_pred.shape == v_theta.shape if model_pred is not None else 'model_pred is None'}")
    
    # Handle text embeddings (model_pred_text can be None for some tasks)
    if model_pred_text is not None:
        raw_text_embeds = prompt_embeds[..., :model_pred_text.shape[-1]]
        noise_text = noise_text[..., :model_pred_text.shape[-1]]
    else:
        raw_text_embeds = prompt_embeds
        # noise_text는 이미 올바른 크기

    # Loss weighting
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    weighting_text = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigma_text)
    weighting_audio = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas_audio)
    
    # Apply dropout to weighting
    if batch['drop_img'] is not None:
        weighting[batch['drop_img']] = 0

    # Gesture loss (adapted from image loss)
    print(f"DEBUG: ========== LOSS CALCULATION ==========")
    print(f"DEBUG: Gesture loss inputs:")
    print(f"DEBUG:   weighting shape: {weighting.shape}")
    print(f"DEBUG:   model_pred shape: {model_pred.shape if model_pred is not None else None}")
    print(f"DEBUG:   v_theta shape: {v_theta.shape}")
    
    if model_pred is not None and v_theta is not None:
        loss_gesture = (weighting.float() * (model_pred - v_theta.float()) ** 2).mean()
        print(f"DEBUG:   gesture loss value: {loss_gesture.item()}")
    else:
        loss_gesture = torch.tensor(0.0, device=weighting.device)
        print(f"DEBUG:   gesture loss set to 0 (None inputs)")

    # Text loss (same as OmniFlow) - with None handling
    print(f"DEBUG: Text loss inputs:")
    print(f"DEBUG:   weighting_text shape: {weighting_text.shape}")
    with torch.no_grad():
        weighting_text = weighting_text.view(-1, 1, 1)
        if batch['drop_text'] is not None:
            weighting_text[batch['drop_text']] = 0
            print(f"DEBUG:   Applied text dropout")
    
    # Handle text loss only if text output exists        
    if model_pred_text is not None and loss_text_factor > 0:
        if precondition_text_outputs:
            loss_text = (weighting_text.float() * (model_pred_text.float() - raw_text_embeds.float().detach()) ** 2).mean()
            norm_1 = F.normalize(model_pred_text, dim=-1, eps=1e-4).float()
            norm_2 = F.normalize(raw_text_embeds, dim=-1, eps=1e-4).float().detach()
            loss_text_norm = (weighting_text.float() * (norm_1 - norm_2) ** 2).mean()
            loss_text_norm = loss_text_norm * 0.1
        else:
            v_theta_text = noise_text - raw_text_embeds
            loss_text = (weighting_text.float() * (model_pred_text.float() - v_theta_text.float()) ** 2).mean()
            loss_text_norm = 0
    else:
        # No text output or text factor is 0
        loss_text = torch.tensor(0.0, device=model_input.device)
        loss_text_norm = 0
        
    # Audio loss (same as OmniFlow)
    print(f"DEBUG: Audio loss inputs:")
    print(f"DEBUG:   weighting_audio shape before view: {weighting_audio.shape}")
    weighting_audio = weighting_audio.view(-1, 1, 1, 1)
    print(f"DEBUG:   weighting_audio shape after view: {weighting_audio.shape}")
    print(f"DEBUG:   model_pred_audio shape: {model_pred_audio.shape if model_pred_audio is not None else None}")
    print(f"DEBUG:   v_theta_audio shape: {v_theta_audio.shape}")
    
    if model_pred_audio is not None:
        loss_audio = (weighting_audio.float() * (model_pred_audio - v_theta_audio.float()) ** 2).mean()
        print(f"DEBUG:   audio loss value: {loss_audio.item()}")
    else:
        loss_audio = torch.tensor(0.0, device=weighting_audio.device)
        print(f"DEBUG:   audio loss set to 0 (None model_pred_audio)")

    # Decode loss for text generation (same as OmniFlow)
    if anchor:
        from train import WeightedLabelSmoother, compute_decode_loss_weight
        label_smoother = WeightedLabelSmoother(epsilon=0.0, ignore_index=-100)
        decode_loss_tgt_weight = torch.ones(len(timesteps_text)).to(logits)
        if anchor:
            decode_loss_weight = torch.ones(len(timesteps_text)).to(logits)
        else:
            decode_loss_weight = compute_decode_loss_weight(timesteps_text, noise_scheduler_copy.config.num_train_timesteps)
        if batch['drop_text'] is not None:
            decode_loss_weight[batch['drop_text']] = 0
            decode_loss_tgt_weight[batch['drop_text']] = 0
        decode_loss_pred = label_smoother([logits], target_labels, shift_labels=True, sample_weight=decode_loss_weight)
        decode_loss_tgt = label_smoother([logits_labels], target_labels, shift_labels=True, sample_weight=decode_loss_tgt_weight)
        decode_loss = None
    else:
        decode_loss_pred = 0
        decode_loss_tgt = 0
        decode_loss = None

    # Total loss
    loss = (loss_gesture * loss_gesture_factor + 
            (loss_text + loss_text_norm) * loss_text_factor + 
            loss_audio * loss_aud_factor + 
            (decode_loss_tgt + decode_loss_pred) * loss_text_factor * 0.1)

    # Logging
    logs = {
        "loss": loss.detach().item(), 
        "lr": last_lr,
        "loss_aud_factor": loss_aud_factor,
        "loss_gesture_factor": loss_gesture_factor,  # Renamed
        "loss_text_factor": loss_text_factor,
        "task_type": task_type  # Log original task type
    }
    
    if loss_text_factor > 0 and model_pred_text is not None:
        logs.update({
            "loss_text": loss_text.detach().item(),
            "loss_text_norm": loss_text_norm.detach().item() if isinstance(loss_text_norm, torch.Tensor) else loss_text_norm,
        })
        with torch.no_grad():
            if raw_text_embeds is not None:
                logs.update({
                    "text_embed_mean": raw_text_embeds.mean().item(),
                    "text_embed_std": raw_text_embeds.std().item(),
                })
        if anchor:
            logs.update({
                "decode_loss_tgt": decode_loss_tgt.detach().item(),
                "decode_loss": decode_loss_pred.detach().item(),
            })
            
    if loss_gesture_factor > 0:
        logs.update({
            "loss_gesture": loss_gesture.detach().item(),
        })
        
    if loss_aud_factor > 0:
        logs.update({
            "loss_audio": loss_audio.detach().item(),
        })
        
    # Compute predictions
    with torch.no_grad():
        model_pred = model_pred * (-sigmas) + noisy_model_input
        model_pred_audio = model_pred_audio * (-sigmas_audio) + noisy_audio_embeds
        target = model_input
        
    return (
        loss, decode_loss, logs, task_type, model_pred, logits, target, prompts,
        model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text, raw_text_embeds
    )


def omniges_forward_pass(
    transformer, args, text_encoder_one, text_encoder_two, text_encoder_three,
    accelerator, batch, gesture_vae, tokenizer_three, text_encoders, tokenizers,
    tokenizer_one, tokenizer_two, weight_dtype, noise_scheduler_copy,
    noise_scheduler, audio_vae_factor, audiovae, text_vae_tokenizer,
    last_lr, text_vae, audio_encoder, do_decode=False,
    precondition_text_outputs=False, anchor=False, mm_encoder=None
):
    """
    Complete forward pass for Omniges training
    """
    
    # Prepare inputs
    (noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,
     noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,
     sigmas, sigmas_audio, model_input, loss_gesture_factor, loss_text_factor, loss_aud_factor,
     noise_scheduler_copy, raw_audio_embeds, task, task_type, prompts, noise, noise_text, noise_audio,
     target_labels, prompt_embeds_vae_uncond, gesture_sequences) = prepare_omniges_inputs(
        transformer, args, text_encoder_one, text_encoder_two, text_encoder_three,
        accelerator, batch, gesture_vae, tokenizer_three, text_encoders, tokenizers,
        tokenizer_one, tokenizer_two, weight_dtype, noise_scheduler_copy,
        noise_scheduler, audio_vae_factor, audiovae, text_vae_tokenizer,
        text_vae, audio_encoder, anchor, mm_encoder=mm_encoder
    )
    
    # Compute loss
    loss, decode_loss, logs, task_type, model_pred, logits, target, prompts, model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text, raw_text_embeds = compute_omniges_loss(
        transformer, noisy_model_input, timesteps, timesteps_text, timesteps_audio, noisy_prompt_embeds,
        noisy_audio_embeds, sigma_text, prompt_embeds, pooled_prompt_embeds, targets, prompt_embeds_uncond,
        sigmas, sigmas_audio, model_input, loss_gesture_factor, loss_text_factor, loss_aud_factor,
        noise_scheduler_copy, last_lr, raw_audio_embeds, task, task_type, prompts,
        noise, noise_text, noise_audio, text_vae, target_labels, do_decode,
        prompt_embeds_vae_uncond, precondition_text_outputs=precondition_text_outputs,
        anchor=anchor, batch=batch, gesture_sequences=gesture_sequences
    )
    
    return loss, decode_loss, logs, task_type, model_pred, logits, target, prompts, model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text.detach() if model_pred_text is not None else None, raw_text_embeds.detach() if raw_text_embeds is not None else None


@torch.no_grad()
def log_omniges_validation(
    pipeline, args, accelerator, pipeline_args, global_step,
    is_final_validation=False, prefix='', do_gesture=True, do_audio=True, do_text=True,
):
    """
    Validation logging for Omniges
    Tests all supported tasks: t2g, g2t, a2g, g2a, t2a, a2t
    """
    logger.info(f"Running Omniges validation... Generating samples for all tasks")
    pipeline = pipeline.to(accelerator.device)
    
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    autocast_ctx = nullcontext()
    
    with autocast_ctx:
        phase_name = f"test_{prefix}" if is_final_validation else f"validation_{prefix}"
        
        # Test Text to Gesture (t2g)
        if do_gesture:
            try:
                gesture_results = []
                test_prompts = ["A person waving hello", "Someone clapping hands", "Dancing movements"]
                for prompt in test_prompts:
                    result = pipeline(
                        prompt=prompt,
                        task='t2g',
                        seq_length=128,
                        guidance_scale=7.0,
                        generator=generator
                    )
                    gesture_results.append(result)
                    
                # Log to wandb
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        # Log gesture sequences as numpy arrays
                        gesture_data = []
                        for i, result in enumerate(gesture_results):
                            if hasattr(result, 'gestures'):
                                gesture_np = result.gestures.cpu().numpy()
                                gesture_data.append({
                                    'prompt': test_prompts[i],
                                    'gesture_shape': str(gesture_np.shape),
                                    'gesture_mean': float(gesture_np.mean()),
                                    'gesture_std': float(gesture_np.std())
                                })
                        
                        df = pd.DataFrame(gesture_data)
                        html = wandb.Html(df.to_html(), inject=True)
                        tracker.log({f"t2g_{phase_name}": html})
                        
            except Exception as e:
                logger.warning(f"T2G validation failed: {e}")
        
        # Test Audio to Gesture (a2g)
        if do_gesture and do_audio:
            try:
                for ref_audio in ['assets/car engine.mp3']:
                    if os.path.exists(ref_audio):
                        result = pipeline(
                            input_aud=ref_audio,
                            task='a2g',
                            seq_length=128,
                            guidance_scale=7.0
                        )
                        
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                if hasattr(result, 'gestures'):
                                    gesture_np = result.gestures.cpu().numpy()
                                    gesture_info = {
                                        'audio_file': ref_audio,
                                        'gesture_shape': str(gesture_np.shape),
                                        'gesture_mean': float(gesture_np.mean()),
                                        'gesture_std': float(gesture_np.std())
                                    }
                                    tracker.log({f"a2g_{phase_name}": gesture_info})
                                    
            except Exception as e:
                logger.warning(f"A2G validation failed: {e}")
        
        # Test Gesture to Text (g2t)
        if do_text:
            try:
                # Create dummy gesture for testing
                dummy_gesture = torch.randn(1, 128, 415).to(accelerator.device)
                
                result = pipeline(
                    input_gesture=dummy_gesture,
                    task='g2t',
                    guidance_scale=2.0
                )
                
                if isinstance(result, tuple) and len(result) >= 2:
                    generated_text = result[0][0] if result[0] else "No text generated"
                    
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            tracker.log({
                                f"g2t_{phase_name}": {
                                    'generated_text': generated_text,
                                    'gesture_input_shape': str(dummy_gesture.shape)
                                }
                            })
                            
            except Exception as e:
                logger.warning(f"G2T validation failed: {e}")
        
        # Test Text to Audio (t2a) - from OmniFlow
        if do_audio:
            try:
                spec, _ = pipeline(
                    prompt="Music playing softly",
                    task='t2a',
                    guidance_scale=4.0,
                    num_inference_steps=28
                )
                
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        # Log audio spectrogram info
                        tracker.log({
                            f"t2a_{phase_name}": {
                                'spec_shape': str(spec.shape) if hasattr(spec, 'shape') else 'No shape',
                                'spec_mean': float(np.mean(spec)) if spec is not None else 0,
                                'spec_std': float(np.std(spec)) if spec is not None else 0
                            }
                        })
                        
            except Exception as e:
                logger.warning(f"T2A validation failed: {e}")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


def parse_omniges_args(input_args=None):
    """Parse arguments for Omniges training"""
    parser = argparse.ArgumentParser(description="Omniges multi-modal training script.")
    
    # Basic model args
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained OmniFlow model",
    )
    
    parser.add_argument(
        "--beat_config_path",
        type=str,
        default="configs/shortcut_rvqvae_128.yaml",
        help="Path to BEAT dataset configuration",
    )
    
    parser.add_argument(
        "--rvqvae_checkpoints",
        type=str,
        default="./ckpt/",
        help="Directory containing RVQVAE checkpoints",
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default='/localhome/jacklishufan/TinyLlama_v1.1',
        required=True,
        help="Path to tokenizer for text VAE",
    )
    
    # Training args
    parser.add_argument("--output_dir", type=str, default="omniges-training", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Training seed")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for compatibility")
    parser.add_argument("--seq_length", type=int, default=128, help="Gesture sequence length")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_ema", action="store_true", help="Use EMA")
    parser.add_argument("--ema_momentum", type=float, default=0.9999, help="EMA momentum")
    
    # Validation args
    parser.add_argument("--validation_prompt", type=str, default="A person waving", help="Validation prompt")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of validation samples")
    parser.add_argument("--val_every", type=int, default=500, help="Validation frequency")
    
    # Scheduler args
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps")
    
    # Loss weighting args
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Logit normal mean")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Logit normal std")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Mode scale")
    parser.add_argument("--uniform_flow", action="store_true", help="Use uniform flow matching")
    
    # Checkpoint args
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Checkpoint frequency")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5, help="Max checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    # Optimizer args  
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Logging args
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting backend")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    
    # Advanced args
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Text VAE args
    parser.add_argument("--text_vae", type=str, required=True, help="Path to text VAE model")
    parser.add_argument("--precondition_text_outputs", action="store_true", help="Precondition text outputs")
    parser.add_argument("--anchor", action="store_true", help="Use anchor loss")
    
    # Parse args
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # Validate args
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
        
    return args


def tokenize_prompt(tokenizer, prompt):
    """Tokenize prompt for text generation"""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def load_safe_tensors(fp, model):
    """Safe tensor loading with shape checking"""
    tensors = torch.load(fp, map_location='cpu')
    
    model_dict = model.state_dict()
    keys_to_pop = []
    for k, v in tensors.items():
        if k in model_dict and model_dict[k].shape != v.shape:
            print(f"SIZE MISMATCH {k}: {model_dict[k].shape} vs {v.shape}")
            keys_to_pop.append(k)
    for k in keys_to_pop:
        tensors.pop(k)
        
    res = model.load_state_dict(tensors, strict=False)
    print(f"Loaded {fp}: {res}")
    del tensors
    torch.cuda.empty_cache()


def load_safe_tensors_ema(fp, model):
    """Load EMA model weights"""
    tensors = torch.load(fp, map_location='cpu')
    res = model.load_state_dict(tensors)
    print(f"Loaded EMA {fp}: {res}")
    del tensors
    torch.cuda.empty_cache()


def main(args):
    """Main training function for Omniges"""
    
    # Setup accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizers (same as OmniFlow)
    tokenizer_one = CLIPTokenizer.from_pretrained('laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    tokenizer_three = T5TokenizerFast.from_pretrained('google/flan-t5-large')

    # Load schedulers (same as OmniFlow)
    noise_scheduler = OmniFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", shift=1
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    noise_scheduler_pipeline = copy.deepcopy(noise_scheduler)
    
    # Load text encoders (same as OmniFlow)
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', projection_dim=768
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    text_encoder_three = T5EncoderModel.from_pretrained('google/flan-t5-large')

    # Load encoders
    audio_encoder = LanguageBindAudio.from_pretrained('LanguageBind/LanguageBind_Audio_FT')
    audio_encoder.text_model = nn.Identity()
    
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    audio_processor_clip = LanguageBindAudioProcessor(audio_encoder.config)
    
    # Set encoders to eval
    text_encoder_one.eval()
    text_encoder_two.eval()
    text_encoder_three.eval()
    
    # Load VAEs
    audiovae, audio_processor = load_audio_vae()
    
    # Load text VAE
    text_vae_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    text_vae_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    config = AutoConfig.from_pretrained(args.text_vae)
    text_vae = LLamaForLatentConnector._from_config(config, torch_dtype=torch.bfloat16)
    text_vae.prepare_tokenizer(text_vae_tokenizer)
    text_vae.set_encoder(text_encoder_three)
    
    # Create gesture VAE
    rvqvae_checkpoints = {
        'upper': os.path.join(args.rvqvae_checkpoints, 'net_300000_upper.pth'),
        'hands': os.path.join(args.rvqvae_checkpoints, 'net_300000_hands.pth'),
        'lower_trans': os.path.join(args.rvqvae_checkpoints, 'net_300000_lower.pth'),
        'face': os.path.join(args.rvqvae_checkpoints, 'net_300000_face.pth')
    }
    gesture_vae = OmnigesGestureVAE(rvqvae_checkpoints)
    
        # Create NEW Omniges transformer - OmniFlow 차원에 맞춤
    transformer = OmnigesFlowTransformerModel(
        seq_length=args.seq_length,
        gesture_latent_dim=512,      # 128 * 4 parts
        num_layers=24,               # OmniFlow 실제 레이어 수 (더 큰 모델)
        num_attention_heads=24,      # OmniFlow 실제 head 수
        attention_head_dim=64,
        joint_attention_dim=4096,
        caption_projection_dim=1536, # OmniFlow 실제 차원 1536
        pooled_projection_dim=2048,
        audio_input_dim=8,
        gesture_output_dim=512,
        add_audio=True,
        use_audio_mae=False,
        drop_gesture=False,
        drop_text=False,
        drop_audio=False
    )
    
    # Set text decoder
    transformer.set_text_decoder(text_vae)
    
    # Load OmniFlow weights if available
    if args.pretrained_model_name_or_path:
        fp = os.path.join(args.pretrained_model_name_or_path, 'transformer/diffusion_pytorch_model.bin')
        if os.path.exists(fp):
            try:
                load_safe_tensors(fp, transformer)
                logger.info("Loaded OmniFlow weights successfully")
            except Exception as e:
                logger.warning(f"Could not load OmniFlow weights: {e}")
    
    # Set requires_grad
    transformer.requires_grad_(True)
    text_vae.requires_grad_(False)
    audio_encoder.requires_grad_(False)
    audiovae.requires_grad_(False)
    gesture_vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    
    # Weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # EMA setup
    if args.use_ema and accelerator.is_main_process:
        ema_transformer = EMAModel(transformer.parameters(), decay=args.ema_momentum)
        
    # Move models to device
    gesture_vae.to(accelerator.device, dtype=weight_dtype)
    audiovae.to(accelerator.device, dtype=torch.float32)
    text_vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    audio_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    # Create optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Create dataset
    train_dataset = OmnigesDataset(
        beat_config_path=args.beat_config_path,
        task_weights=[1/6] * 6,  # Equal weight for all tasks
        size=args.resolution,
        is_train=True,
        image_processor=image_processor,
        audio_processor=audio_processor,
        audio_processor_clip=audio_processor_clip,
    )
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=omniges_collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    # Create scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare with accelerator
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
    
    # Create text encoders list
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    
    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running Omniges Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Supported tasks: t2g, g2t, a2g, g2a, t2a, a2t")
    
    global_step = 0
    first_epoch = 0
    
    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps * args.gradient_accumulation_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # Initialize tracking
    if accelerator.is_main_process:
        accelerator.init_trackers("omniges-training", config=vars(args))
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([transformer]):
                
                # Forward pass
                loss, decode_loss, logs, task_type, model_pred, logits, target, prompts, model_pred_audio, model_pred_audio, raw_audio_embeds, model_pred_text, raw_text_embeds = transformer(
                    kkwargs={
                        'args': args,
                        'text_encoder_one': text_encoder_one,
                        'text_encoder_two': text_encoder_two,
                        'text_encoder_three': text_encoder_three,
                        'accelerator': accelerator.device,
                        'batch': batch,
                        'gesture_vae': gesture_vae,  # Use gesture VAE instead of image VAE
                        'tokenizer_three': tokenizer_three,
                        'text_encoders': text_encoders,
                        'tokenizers': tokenizers,
                        'tokenizer_one': tokenizer_one,
                        'tokenizer_two': tokenizer_two,
                        'weight_dtype': weight_dtype,
                        'noise_scheduler_copy': noise_scheduler_copy,
                        'noise_scheduler': noise_scheduler,
                        'audio_vae_factor': 1,
                        'audiovae': audiovae,
                        'text_vae_tokenizer': text_vae_tokenizer,
                        'last_lr': lr_scheduler.get_last_lr()[0],
                        'text_vae': text_vae,
                        'audio_encoder': audio_encoder,
                        'do_decode': False,
                        'precondition_text_outputs': args.precondition_text_outputs,
                        'anchor': args.anchor,
                        'mm_encoder': None,
                    },
                    forward_function=omniges_forward_pass
                )

                # Backward pass
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # EMA update
                if accelerator.sync_gradients:
                    if args.use_ema and accelerator.is_main_process:
                        if global_step % 100 == 0:  # Update EMA every 100 steps
                            ema_transformer.step(transformer.parameters())

            # Progress tracking
            progress_bar.update(1)
            if accelerator.sync_gradients:
                global_step += 1
                
                # Log metrics
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Checkpointing
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
                    # Save EMA separately
                    if args.use_ema:
                        ema_path = os.path.join(save_path, "ema_transformer.pt")
                        torch.save(ema_transformer.state_dict(), ema_path)
                    
                    logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation - 모든 태스크 테스트
                if global_step % args.val_every == 0 and global_step > 0:
                    transformer.eval()
                    
                    # Create validation pipeline
                    pipeline = OmnigesPipeline(
                        transformer=accelerator.unwrap_model(transformer),
                        scheduler=noise_scheduler_pipeline,
                        gesture_vae=gesture_vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        text_encoder_3=accelerator.unwrap_model(text_encoder_three),
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        tokenizer_3=tokenizer_three,
                        audio_vae=audiovae,
                        audio_processor=audio_processor,
                        audio_processor_clip=audio_processor_clip,
                        audio_encoder=accelerator.unwrap_model(audio_encoder),
                        text_vae=text_vae,
                        text_vae_tokenizer=text_vae_tokenizer,
                        text_x0=args.precondition_text_outputs,
                    )
                    
                    # 🎯 모든 태스크별 검증 (OmniFlow 방식 확장)
                    validation_results = {}
                    
                    # 1. Text to Gesture (t2g) 
                    try:
                        t2g_result = pipeline(
                            prompt=args.validation_prompt,
                            task='t2g',
                            seq_length=128,
                            guidance_scale=7.0
                        )
                        if hasattr(t2g_result, 'gestures'):
                            gesture_np = t2g_result.gestures.cpu().numpy()
                            validation_results['t2g'] = {
                                'shape': gesture_np.shape,
                                'mean': float(gesture_np.mean()),
                                'std': float(gesture_np.std())
                            }
                            logger.info(f"  ✅ T2G validation: {gesture_np.shape}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ T2G validation failed: {e}")
                    
                    # 2. Gesture to Text (g2t)
                    try:
                        dummy_gesture = torch.randn(1, 128, 415).to(accelerator.device)
                        g2t_result = pipeline(
                            input_gesture=dummy_gesture,
                            task='g2t',
                            guidance_scale=2.0
                        )
                        if isinstance(g2t_result, tuple) and len(g2t_result) >= 2:
                            generated_text = g2t_result[0][0] if g2t_result[0] else "No text"
                            validation_results['g2t'] = {
                                'text': generated_text,
                                'length': len(generated_text.split())
                            }
                            logger.info(f"  ✅ G2T validation: '{generated_text[:30]}...'")
                    except Exception as e:
                        logger.warning(f"  ⚠️ G2T validation failed: {e}")
                    
                    # 3. Audio to Gesture (a2g) - if audio available
                    if os.path.exists('./assets/car engine.mp3'):
                        try:
                            a2g_result = pipeline(
                                input_aud='./assets/car engine.mp3',
                                task='a2g',
                                seq_length=128,
                                guidance_scale=7.0
                            )
                            if hasattr(a2g_result, 'gestures'):
                                gesture_np = a2g_result.gestures.cpu().numpy()
                                validation_results['a2g'] = {
                                    'shape': gesture_np.shape,
                                    'mean': float(gesture_np.mean())
                                }
                                logger.info(f"  ✅ A2G validation: {gesture_np.shape}")
                        except Exception as e:
                            logger.warning(f"  ⚠️ A2G validation failed: {e}")
                    
                    # 4. Gesture to Audio (g2a)
                    try:
                        g2a_result = pipeline(
                            input_gesture=dummy_gesture,
                            task='g2a',
                            guidance_scale=4.0
                        )
                        if isinstance(g2a_result, tuple) and len(g2a_result) >= 1:
                            audio_spec = g2a_result[0]
                            validation_results['g2a'] = {
                                'audio_shape': str(audio_spec.shape) if hasattr(audio_spec, 'shape') else 'No shape',
                                'audio_mean': float(np.mean(audio_spec)) if audio_spec is not None else 0
                            }
                            logger.info(f"  ✅ G2A validation: audio generated")
                    except Exception as e:
                        logger.warning(f"  ⚠️ G2A validation failed: {e}")
                    
                    # 5. Text to Audio (t2a) - OmniFlow 방식
                    try:
                        t2a_result = pipeline(
                            prompt="Music playing",
                            task='t2a',
                            guidance_scale=4.0
                        )
                        if isinstance(t2a_result, tuple) and len(t2a_result) >= 1:
                            audio_spec = t2a_result[0]
                            validation_results['t2a'] = {
                                'audio_shape': str(audio_spec.shape) if hasattr(audio_spec, 'shape') else 'No shape'
                            }
                            logger.info(f"  ✅ T2A validation: audio generated")
                    except Exception as e:
                        logger.warning(f"  ⚠️ T2A validation failed: {e}")
                    
                    # 6. Audio to Text (a2t) - OmniFlow 방식
                    if os.path.exists('./assets/car engine.mp3'):
                        try:
                            a2t_result = pipeline(
                                input_aud='./assets/car engine.mp3',
                                task='a2t',
                                guidance_scale=2.0
                            )
                            if isinstance(a2t_result, tuple) and len(a2t_result) >= 2:
                                generated_text = a2t_result[0][0] if a2t_result[0] else "No text"
                                validation_results['a2t'] = {
                                    'text': generated_text,
                                    'length': len(generated_text.split())
                                }
                                logger.info(f"  ✅ A2T validation: '{generated_text[:30]}...'")
                        except Exception as e:
                            logger.warning(f"  ⚠️ A2T validation failed: {e}")
                    
                    # Log all validation results
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            # Create validation summary table
                            val_data = []
                            for task, result in validation_results.items():
                                val_data.append({
                                    'task': task.upper(),
                                    'status': '✅ Success',
                                    'details': str(result)
                                })
                            
                            if val_data:
                                df = pd.DataFrame(val_data)
                                html = wandb.Html(df.to_html(), inject=True)
                                tracker.log({f"validation_all_tasks_step_{global_step}": html})
                            
                            # Log individual task results
                            for task, result in validation_results.items():
                                tracker.log({f"val_{task}": result}, step=global_step)
                    
                    transformer.train()
                    del pipeline
                    torch.cuda.empty_cache()
                
                if global_step >= args.max_train_steps:
                    break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-final")
        accelerator.save_state(save_path)
        
        if args.use_ema:
            ema_path = os.path.join(save_path, "ema_transformer.pt")
            torch.save(ema_transformer.state_dict(), ema_path)
            
        logger.info(f"Training complete! Final checkpoint saved to {save_path}")
        
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_omniges_args()
    main(args)
