"""
Gesture Processor for Omniges
4x RVQVAE Gesture Encoder/Decoder for different body parts

Extracted from omniges_a2g.py and optimized for multi-task use
Following shortcut_rvqvae_trainer.py pattern for proper concatenation
"""

import torch
import torch.nn as nn
from typing import Dict
from models.vq.model import RVQVAE


class GestureProcessor(nn.Module):
    """
    4x RVQVAE Gesture Encoder/Decoder for different body parts
    Follows shortcut_rvqvae_trainer.py pattern for proper latent concatenation
    """
    
    def __init__(
        self,
        ckpt_paths: Dict[str, str],
        device: str = "cuda"
    ):
        super().__init__()
        
        self.body_parts = ["upper", "hands", "lower_trans", "face"]
        self.part_dims = {
            "upper": 78,
            "hands": 180, 
            "lower_trans": 57,
            "face": 100
        }
        
        # Load pre-trained RVQVAE models
        self.rvqvae_models = nn.ModuleDict()
        self.load_rvqvae_models(ckpt_paths, device)
        
        # Freeze RVQVAE parameters (use as fixed encoders/decoders)
        for model in self.rvqvae_models.values():
            for param in model.parameters():
                param.requires_grad = False
        
    def load_rvqvae_models(self, ckpt_paths: Dict[str, str], device: str):
        """Load pre-trained RVQVAE models for each body part."""
        # Default args for RVQVAE (matching actual checkpoint settings)
        class DefaultArgs:
            def __init__(self):
                self.nb_code = 1024  # From checkpoint: codebook size is 1024
                self.code_dim = 128  # From checkpoint: embedding dim is 128
                self.down_t = 2
                self.stride_t = 2  
                self.width = 512
                self.depth = 3
                self.dilation_growth_rate = 3
                self.vq_act = 'relu'
                self.vq_norm = None
                self.num_quantizers = 6
                self.shared_codebook = False
                self.quantize_dropout_prob = 0.2
                # Additional required args
                self.mu = 0.99  # EMA decay rate
                self.quantizer = 'ema_reset'
                self.beta = 1.0
                self.vae_length = 64
                self.vae_codebook_size = 1024  # Match nb_code
                self.vae_quantizer_lambda = 1.0
        
        args = DefaultArgs()
        
        for part in self.body_parts:
            if part in ckpt_paths and ckpt_paths[part] is not None:
                # Create RVQVAE model
                model = RVQVAE(
                    args,
                    input_width=self.part_dims[part],
                    nb_code=args.nb_code,
                    code_dim=args.code_dim,
                    output_emb_width=args.code_dim,
                    down_t=args.down_t,
                    stride_t=args.stride_t,
                    width=args.width,
                    depth=args.depth,
                    dilation_growth_rate=args.dilation_growth_rate,
                    activation=args.vq_act,
                    norm=args.vq_norm
                )
                
                # Load checkpoint
                ckpt = torch.load(ckpt_paths[part], map_location='cpu')
                model.load_state_dict(ckpt['net'], strict=True)
                model.to(device).eval()
                
                self.rvqvae_models[part] = model
                print(f"Loaded RVQVAE model for {part} from {ckpt_paths[part]}")
    
    def encode_gesture(self, gesture_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode gesture data to continuous latents (following shortcut_rvqvae_trainer.py)
        
        Args:
            gesture_dict: Dict with body part data
                - upper: (B, T, 78) in 6D rotation format
                - hands: (B, T, 180) in 6D rotation format
                - lower_trans: (B, T, 57) in 6D rotation format  
                - face: (B, T, 100)
                
        Returns:
            Dict with continuous latents for each part
        """
        encoded = {}
        
        for part, data in gesture_dict.items():
            if part in self.rvqvae_models:
                with torch.no_grad():
                    # Use map2latent for continuous latent extraction
                    latents = self.rvqvae_models[part].map2latent(data)  # (B, T//downsample, code_dim)
                    encoded[f"{part}_latents"] = latents
        
        return encoded
    
    def decode_gesture(
        self, 
        latents_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode gesture data from continuous latents (following shortcut_rvqvae_trainer.py)
        
        Args:
            latents_dict: Continuous latents for each body part
            
        Returns:
            Reconstructed gesture data for each body part
        """
        decoded = {}
        
        for part in self.body_parts:
            part_key = f"{part}_latents"
            if part in self.rvqvae_models and part_key in latents_dict:
                input_latents = latents_dict[part_key]
                with torch.no_grad():
                    # Use latent2origin for continuous latent decoding
                    recon_output = self.rvqvae_models[part].latent2origin(input_latents)
                    if isinstance(recon_output, tuple):
                        decoded[part] = recon_output[0]  # Take first element if tuple
                    else:
                        decoded[part] = recon_output
        
        return decoded
