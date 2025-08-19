"""
Omniges Pipeline: Text-Audio-Gesture Multimodal Pipeline
Based on OmniFlow with Image stream replaced by Gesture stream
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import OmniFlow components
from omniflow.pipelines.omniflow_pipeline import OmniFlowPipeline
from omniflow.models.omni_flow import OmniFlowTransformerModel
from omniflow.models.text_vae import LLamaForLatentConnector
from omniflow.models.audio_vae import load_audio_vae
from omniflow.models.encoders import LanguageBindAudio, LanguageBindAudioProcessor

# Import our gesture components
from omniges.models.omniges_a2g import GestureProcessor, create_rvqvae_manager

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer, 
    T5EncoderModel,
    T5TokenizerFast,
    AutoTokenizer,
    AutoConfig
)
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput

logger = logging.get_logger(__name__)


class OmnigesGestureVAE(nn.Module):
    """
    Gesture VAE Adapter for Omniges Pipeline
    Adapts 4x RVQVAE to work like Image VAE in OmniFlow
    """
    
    def __init__(self, rvqvae_checkpoints: Dict[str, str]):
        super().__init__()
        self.gesture_processor = GestureProcessor(
            ckpt_paths=rvqvae_checkpoints,
            device="cuda"
        )
        
        # Mimic VAE config for compatibility
        self.config = type('Config', (), {
            'scaling_factor': 1.0,
            'shift_factor': 0.0,
            'block_out_channels': [128, 256, 512, 1024]  # For compatibility
        })()
        
    def encode(self, gesture_sequence):
        """
        Encode gesture sequence to latents
        Args:
            gesture_sequence: (B, T, 415) - combined gesture features
        Returns:
            latents: (B, C, H, W) - 2D latent representation for transformer
        """
        # Split gesture into parts (using the same logic as our adapter)
        B, T, total_dim = gesture_sequence.shape
        print(f"DEBUG: [GestureVAE.encode] Input gesture_sequence shape: {gesture_sequence.shape}")
        
        # Simple proportional split (can be improved with proper joint mapping)
        gesture_parts = {
            'upper': gesture_sequence[:, :, :78],           # 78 dims
            'hands': gesture_sequence[:, :, 78:258],        # 180 dims  
            'lower_trans': gesture_sequence[:, :, 258:315], # 57 dims
            'face': gesture_sequence[:, :, 315:415]         # 100 dims
        }
        
        print(f"DEBUG: [GestureVAE.encode] Split parts shapes:")
        for part, data in gesture_parts.items():
            print(f"DEBUG:   {part}: {data.shape}")
        
        # Encode each part to latents using RVQVAE.map2latent() 
        # Following shortcut_rvqvae_trainer.py approach
        latents_dict = self.gesture_processor.encode_gesture(gesture_parts)
        
        # DEBUG: Check what's actually encoded
        print(f"DEBUG: [GestureVAE.encode] latents_dict keys: {latents_dict.keys()}")
        for key, latent in latents_dict.items():
            if latent is not None:
                print(f"DEBUG: [GestureVAE.encode] {key} shape: {latent.shape}")
            else:
                print(f"DEBUG: [GestureVAE.encode] {key} is None!")
        
        # Extract individual part latents following shortcut_rvqvae_trainer.py pattern
        # Each should be (B, T//downsample, 128) after map2latent()
        # From debug: each is (B, 32, 128), so we need T=32
        actual_T = list(latents_dict.values())[0].shape[1] if latents_dict else T  # Use actual downsampled T
        latent_upper = latents_dict.get('upper_latents', torch.zeros((B, actual_T, 128), device=gesture_sequence.device))
        latent_hands = latents_dict.get('hands_latents', torch.zeros((B, actual_T, 128), device=gesture_sequence.device)) 
        latent_lower = latents_dict.get('lower_trans_latents', torch.zeros((B, actual_T, 128), device=gesture_sequence.device))
        latent_face = latents_dict.get('face_latents', torch.zeros((B, actual_T, 128), device=gesture_sequence.device))
        
        print(f"DEBUG: [GestureVAE.encode] Using actual_T: {actual_T}")
        print(f"DEBUG: [GestureVAE.encode] Individual latent shapes:")
        print(f"DEBUG:   upper: {latent_upper.shape}, hands: {latent_hands.shape}")
        print(f"DEBUG:   lower: {latent_lower.shape}, face: {latent_face.shape}")
        
        # Concatenate along feature dimension: (B, T, 128) * 4 -> (B, T, 512)
        # Following: latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top, latent_face_top], dim=2)
        combined_latents = torch.cat([latent_upper, latent_hands, latent_lower, latent_face], dim=2)  # (B, T, 512)
        print(f"DEBUG: [GestureVAE.encode] combined_latents shape after concat: {combined_latents.shape}")
        
        # Reshape to match transformer expectations: (B, T, 512) -> (B, 512, T, 1)
        # This matches the format expected by OmniFlow where image latents are (B, C, H, W)
        # Use actual_T instead of original T
        latents_2d = combined_latents.permute(0, 2, 1).unsqueeze(-1)  # (B, 512, actual_T, 1)
        print(f"DEBUG: [GestureVAE.encode] Final latents_2d shape: {latents_2d.shape}")
        
        # IMPORTANT: This should now be (B, 512, actual_T, 1) - 4 parts concatenated in channel dimension
        
        # Return DistributionMock for compatibility with VAE interface
        return DistributionMock(latents_2d)
        
    def decode(self, latents_2d, return_dict=True):
        """
        Decode 2D latents back to gesture sequence
        Args:
            latents_2d: (B, 512, T, 1) - 2D latent representation
        Returns:
            gesture_sequence: (B, T, 415) - decoded gesture features
        """
        B, total_latent_dim, T, _ = latents_2d.shape  # (B, 512, T, 1)
        print(f"DEBUG: [GestureVAE.decode] Input latents_2d shape: {latents_2d.shape}")
        
        # Reshape back to concatenated format: (B, 512, T, 1) -> (B, T, 512)
        combined_latents = latents_2d.squeeze(-1).permute(0, 2, 1)  # (B, T, 512)
        print(f"DEBUG: [GestureVAE.decode] combined_latents shape: {combined_latents.shape}")
        
        # Split back to individual part latents following shortcut_rvqvae_trainer.py pattern
        # rec_latent_upper = sample[...,:code_dim]
        # rec_latent_hands = sample[...,code_dim:code_dim*2] 
        # etc.
        code_dim = 128  # Each part has 128 latent dimensions
        latent_upper = combined_latents[..., :code_dim]                    # (B, T, 128)
        latent_hands = combined_latents[..., code_dim:code_dim*2]          # (B, T, 128)
        latent_lower = combined_latents[..., code_dim*2:code_dim*3]        # (B, T, 128)
        latent_face = combined_latents[..., code_dim*3:code_dim*4]         # (B, T, 128)
        
        print(f"DEBUG: [GestureVAE.decode] Split latents shapes:")
        print(f"DEBUG:   upper: {latent_upper.shape}, hands: {latent_hands.shape}")
        print(f"DEBUG:   lower: {latent_lower.shape}, face: {latent_face.shape}")
        
        # Prepare latents dict for RVQVAE decoding
        latents_dict = {
            'upper_latents': latent_upper,
            'hands_latents': latent_hands, 
            'lower_trans_latents': latent_lower,
            'face_latents': latent_face
        }
            
        # Decode through RVQVAE
        print(f"DEBUG: [GestureVAE.decode] Calling gesture_processor.decode_gesture...")
        decoded_parts = self.gesture_processor.decode_gesture(latents_dict)
        
        print(f"DEBUG: [GestureVAE.decode] Decoded parts shapes:")
        for part, data in decoded_parts.items():
            print(f"DEBUG:   {part}: {data.shape}")
        
        # Combine parts back to full gesture
        gesture_sequence = torch.cat([
            decoded_parts['upper'],      # (B, T, 78)
            decoded_parts['hands'],      # (B, T, 180)
            decoded_parts['lower_trans'], # (B, T, 57)
            decoded_parts['face']        # (B, T, 100)
        ], dim=-1)  # (B, T, 415)
        
        print(f"DEBUG: [GestureVAE.decode] Final gesture_sequence shape: {gesture_sequence.shape}")
        
        if return_dict:
            return type('DecodeOutput', (), {'sample': gesture_sequence})()
        return gesture_sequence


class DistributionMock:
    """Mock VAE distribution for compatibility"""
    def __init__(self, latents):
        self.latents = latents
        
    def sample(self):
        return self.latents
        
    @property 
    def mean(self):
        return self.latents


class OmnigesPipeline(OmniFlowPipeline):
    """
    Omniges Pipeline: Text-Audio-Gesture Multimodal Generation
    Extends OmniFlow by replacing Image stream with Gesture stream
    
    Supported Tasks:
    - t2g: Text to Gesture
    - a2g: Audio to Gesture  
    - g2t: Gesture to Text
    - g2a: Gesture to Audio
    - t2a: Text to Audio (unchanged)
    - a2t: Audio to Text (unchanged)
    """
    
    def __init__(
        self,
        transformer: OmniFlowTransformerModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        gesture_vae: OmnigesGestureVAE,  # Replace image VAE with gesture VAE
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        audio_vae: AutoencoderKL,
        audio_processor,
        audio_processor_clip,
        audio_encoder: LanguageBindAudio,
        text_vae: LLamaForLatentConnector,
        text_vae_tokenizer,
        text_x0: bool = True,
        **kwargs
    ):
        # Initialize parent without vae (we'll override)
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=None,  # We'll use gesture_vae instead
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
            text_vae_tokenizer=text_vae_tokenizer,
            audio_vae=audio_vae,
            audio_processor=audio_processor,
            audio_processor_clip=audio_processor_clip,
            audio_encoder=audio_encoder,
            text_vae=text_vae,
            text_x0=text_x0,
            mode='gesture',  # Set mode to gesture
            **kwargs
        )
        
        # Replace image VAE with gesture VAE
        self.gesture_vae = gesture_vae
        self.vae = gesture_vae  # For compatibility with parent methods
        
        # Update scale factor for gesture
        self.vae_scale_factor = 8  # Can be adjusted based on gesture dimensions
        
    @torch.no_grad()
    def encode_gesture(self, pose_sequence: torch.Tensor):
        """Encode pose sequence to latents"""
        return self.gesture_vae.encode(pose_sequence)
        
    @torch.no_grad()
    def decode_gesture(self, latents: torch.Tensor):
        """Decode latents to pose sequence"""
        return self.gesture_vae.decode(latents)
        
    def prepare_gesture_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        seq_length: int,
        num_parts: int,
        dtype: torch.dtype,
        device: torch.device,
        generator,
        latents=None,
    ):
        """Prepare gesture latents for generation"""
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
            
        # Shape: (batch_size, channels, seq_length, num_parts)
        shape = (batch_size, num_channels_latents, seq_length, num_parts)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
            
        from diffusers.utils.torch_utils import randn_tensor
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        
        return latents
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        seq_length: Optional[int] = 128,  # Gesture sequence length instead of height/width
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_gestures_per_prompt: Optional[int] = 1,  # Instead of num_images_per_prompt
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        task: str = 't2g',  # Default to text-to-gesture
        input_gesture: Optional[torch.FloatTensor] = None,  # Instead of input_img
        input_aud: Optional[str] = None,
        return_dict: bool = True,
        use_text_output: bool = True,
        drop_text: bool = False,
        drop_gesture: bool = False,  # Instead of drop_image
        drop_audio: bool = False,
        **kwargs
    ):
        """
        Omniges Pipeline Call
        
        Supported Tasks:
        - t2g: Text to Gesture
        - a2g: Audio to Gesture
        - g2t: Gesture to Text  
        - g2a: Gesture to Audio
        - t2a: Text to Audio
        - a2t: Audio to Text
        
        Args:
            prompt: Text prompt for generation
            seq_length: Gesture sequence length (instead of height/width)
            task: Task type ('t2g', 'a2g', 'g2t', 'g2a', 't2a', 'a2t')
            input_gesture: Input gesture sequence for g2t, g2a tasks
            input_aud: Input audio path for a2g, a2t tasks
        """
        
        # Validate task
        assert task in ['t2g', 'a2g', 'g2t', 'g2a', 't2a', 'a2t'], f"Unsupported task: {task}"
        
        # Set default sequence length
        seq_length = seq_length or 128
        
        # Map gesture tasks to original image tasks for processing
        task_mapping = {
            't2g': 't2i',  # Text to Gesture -> Text to Image
            'a2g': 'a2i',  # Audio to Gesture -> Audio to Image  
            'g2t': 'i2t',  # Gesture to Text -> Image to Text
            'g2a': 'i2a',  # Gesture to Audio -> Image to Audio
            't2a': 't2a',  # Text to Audio (unchanged)
            'a2t': 'a2t'   # Audio to Text (unchanged)
        }
        
        # Process input gesture for g2t, g2a tasks
        if task in ['g2t', 'g2a'] and input_gesture is not None:
            # Encode gesture to latents using gesture VAE
            gesture_latents = self.encode_gesture(input_gesture)
            if hasattr(gesture_latents, 'sample'):
                gesture_latents = gesture_latents.sample()
            input_img = gesture_latents  # Use as "image" input for parent pipeline
        else:
            input_img = None
            
        # Map task and call parent pipeline
        mapped_task = task_mapping[task]
        
        # Override parent's VAE temporarily for gesture processing
        original_vae = getattr(self, 'vae', None)
        self.vae = self.gesture_vae
        
        try:
            # Call parent pipeline with mapped task
            result = super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                height=seq_length * 8,  # Convert seq_length to "height" for compatibility
                width=32,  # Fixed "width" for gesture (4 parts * 8)
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                num_images_per_prompt=num_gestures_per_prompt,
                generator=generator,
                latents=latents,
                task=mapped_task,
                input_img=input_img,
                input_aud=input_aud,
                return_dict=return_dict,
                use_text_output=use_text_output,
                drop_image=drop_gesture,  # Map gesture dropout to image dropout
                drop_text=drop_text,
                drop_audio=drop_audio,
                **kwargs
            )
            
            # Post-process results for gesture tasks
            if task in ['t2g', 'a2g']:
                # Decode latents to gesture sequences
                if hasattr(result, 'images'):
                    gesture_latents = result.images  # Actually latents from transformer
                else:
                    gesture_latents = result[0]
                    
                gesture_sequences = self.decode_gesture(gesture_latents)
                if hasattr(gesture_sequences, 'sample'):
                    gesture_sequences = gesture_sequences.sample
                    
                if return_dict:
                    return type('OmnigesOutput', (), {
                        'gestures': gesture_sequences,
                        'gesture_latents': gesture_latents
                    })()
                else:
                    return (gesture_sequences,)
                    
            else:
                # For g2t, g2a, t2a, a2t - return as is
                return result
                
        finally:
            # Restore original VAE
            if original_vae is not None:
                self.vae = original_vae
                
    @classmethod 
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        rvqvae_checkpoints: Dict[str, str],
        device: str = 'cuda',
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        """
        Load pretrained Omniges pipeline
        
        Args:
            pretrained_model_path: Path to OmniFlow checkpoint
            rvqvae_checkpoints: Dict of RVQVAE checkpoint paths
        """
        
        # Load OmniFlow components
        omniflow_pipeline = OmniFlowPipeline.load_pretrained(
            pretrained_model_path, 
            device=device,
            weight_dtype=weight_dtype
        )
        
        # Create gesture VAE
        gesture_vae = OmnigesGestureVAE(rvqvae_checkpoints)
        gesture_vae.to(device, dtype=weight_dtype)
        
        # Create Omniges pipeline 
        omniges_pipeline = cls(
            transformer=omniflow_pipeline.transformer,
            scheduler=omniflow_pipeline.scheduler,
            gesture_vae=gesture_vae,
            text_encoder=omniflow_pipeline.text_encoder,
            tokenizer=omniflow_pipeline.tokenizer,
            text_encoder_2=omniflow_pipeline.text_encoder_2,
            tokenizer_2=omniflow_pipeline.tokenizer_2,
            text_encoder_3=omniflow_pipeline.text_encoder_3,
            tokenizer_3=omniflow_pipeline.tokenizer_3,
            audio_vae=omniflow_pipeline.audio_vae,
            audio_processor=omniflow_pipeline.audio_processor,
            audio_processor_clip=omniflow_pipeline.audio_processor_clip,
            audio_encoder=omniflow_pipeline.audio_encoder,
            text_vae=omniflow_pipeline.text_vae,
            text_vae_tokenizer=omniflow_pipeline.text_vae_tokenizer,
            text_x0=omniflow_pipeline.text_x0,
            **kwargs
        )
        
        return omniges_pipeline


def create_omniges_pipeline(
    omniflow_checkpoint_path: str,
    rvqvae_checkpoints: Dict[str, str],
    device: str = 'cuda',
    weight_dtype: torch.dtype = torch.bfloat16
):
    """
    Create Omniges pipeline with OmniFlow checkpoint + RVQVAE checkpoints
    
    Args:
        omniflow_checkpoint_path: Path to OmniFlow model checkpoint
        rvqvae_checkpoints: Dict mapping part names to checkpoint paths
        
    Returns:
        OmnigesPipeline ready for text-audio-gesture generation
    """
    
    return OmnigesPipeline.from_pretrained(
        pretrained_model_path=omniflow_checkpoint_path,
        rvqvae_checkpoints=rvqvae_checkpoints,
        device=device,
        weight_dtype=weight_dtype
    )


# Example usage and testing
if __name__ == "__main__":
    # Example RVQVAE checkpoints
    rvqvae_checkpoints = {
        'upper': './ckpt/net_300000_upper.pth',
        'hands': './ckpt/net_300000_hands.pth', 
        'lower_trans': './ckpt/net_300000_lower.pth',
        'face': './ckpt/net_300000_face.pth'
    }
    
    # Test gesture VAE creation
    gesture_vae = OmnigesGestureVAE(rvqvae_checkpoints)
    
    # Test encode/decode
    dummy_gesture = torch.randn(2, 128, 415)  # (B, T, full_gesture_dim)
    encoded = gesture_vae.encode(dummy_gesture)
    print(f"Encoded shape: {encoded.sample().shape}")
    
    decoded = gesture_vae.decode(encoded.sample())
    print(f"Decoded shape: {decoded.sample.shape}")
    
    print("✅ Omniges Pipeline ready!")
