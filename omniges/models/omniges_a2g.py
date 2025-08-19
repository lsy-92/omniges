"""
Omniges A2G Model - Audio to Gesture Generation
Built on top of OmniFlow pipeline, replacing image stream with gesture stream

Key Components:
1. Audio Fusion: WavLM + MFCC bidirectional cross-attention
2. MMDiT Backbone: 3-modality transformer (text + audio + gesture)  
3. Gesture Encoder/Decoder: 4x RVQVAE (upper, hands, lower_trans, face)
4. A2G Task Head: Audio → Gesture generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
import numpy as np
import os

# Import existing components
from mmdit.mmdit_generalized_pytorch import MMDiT
from models.vq.model import RVQVAE
from omniflow.models.text_vae import LLamaForLatentConnector
from fsq.components.audio_feature import MFCCEncoder
from models.wavlm.WavLM import WavLM, WavLMConfig


class AudioFusionModule(nn.Module):
    """
    Enhanced WavLM + MFCC bidirectional cross-attention fusion
    """
    def __init__(
        self, 
        dim_wavlm: int = 768,
        dim_mfcc: int = 128, 
        dim_fuse: int = 512,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Input projections
        self.wavlm_proj = nn.Linear(dim_wavlm, dim_fuse)
        self.mfcc_proj = nn.Linear(dim_mfcc, dim_fuse)
        
        # Layer normalization
        self.wavlm_norm = nn.LayerNorm(dim_fuse)
        self.mfcc_norm = nn.LayerNorm(dim_fuse)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim_fuse, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Gating mechanism for fusion
        self.gate_proj = nn.Linear(dim_fuse * 2, dim_fuse)
        self.gate_activation = nn.Sigmoid()
        
        # Output projection with residual
        self.output_proj = nn.Linear(dim_fuse, dim_fuse)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        wavlm_feats: torch.Tensor,  # (B, T, 768)
        mfcc_feats: torch.Tensor,   # (B, T, 128)
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            wavlm_feats: WavLM features (B, T, 768)
            mfcc_feats: MFCC features (B, T, 128)
            attn_mask: Optional attention mask (B, T)
            
        Returns:
            Fused audio features (B, T, 512)
        """
        # Project to common dimension
        h_wavlm = self.wavlm_norm(self.wavlm_proj(wavlm_feats))  # (B, T, 512)
        h_mfcc = self.mfcc_norm(self.mfcc_proj(mfcc_feats))      # (B, T, 512)
        
        # Bidirectional cross-attention
        for layer in self.cross_attn_layers:
            # WavLM attends to MFCC
            h_w2m, _ = layer(h_wavlm, h_mfcc, h_mfcc, key_padding_mask=attn_mask)
            h_wavlm = h_wavlm + self.dropout(h_w2m)
            
            # MFCC attends to WavLM  
            h_m2w, _ = layer(h_mfcc, h_wavlm, h_wavlm, key_padding_mask=attn_mask)
            h_mfcc = h_mfcc + self.dropout(h_m2w)
        
        # Gated fusion
        concat_feats = torch.cat([h_wavlm, h_mfcc], dim=-1)  # (B, T, 1024)
        gate = self.gate_activation(self.gate_proj(concat_feats))  # (B, T, 512)
        
        fused = gate * h_wavlm + (1 - gate) * h_mfcc  # (B, T, 512)
        
        # Output projection with residual
        output = fused + self.dropout(self.output_proj(fused))
        
        return output


class GestureProcessor(nn.Module):
    """
    4x RVQVAE Gesture Encoder/Decoder for different body parts
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
        
        print(f"DEBUG: encode_gesture input keys: {list(gesture_dict.keys())}")
        print(f"DEBUG: loaded RVQVAE models: {list(self.rvqvae_models.keys())}")
        
        for part, data in gesture_dict.items():
            print(f"DEBUG: Processing part {part}, data shape: {data.shape}")
            if part in self.rvqvae_models:
                with torch.no_grad():
                    # Use map2latent for continuous latent extraction
                    latents = self.rvqvae_models[part].map2latent(data)  # (B, T//downsample, code_dim)
                    print(f"DEBUG: {part} encoded to latents shape: {latents.shape}")
                    encoded[f"{part}_latents"] = latents
            else:
                print(f"WARNING: {part} not found in RVQVAE models!")
        
        print(f"DEBUG: Final encoded keys: {list(encoded.keys())}")
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
        
        print(f"DEBUG: [GestureProcessor.decode_gesture] Input latents:")
        for key, latents in latents_dict.items():
            if latents is not None:
                print(f"DEBUG:   {key}: {latents.shape}")
            else:
                print(f"DEBUG:   {key}: None")
        
        for part in self.body_parts:
            part_key = f"{part}_latents"
            print(f"DEBUG: [GestureProcessor.decode_gesture] Processing part: {part}")
            if part in self.rvqvae_models and part_key in latents_dict:
                input_latents = latents_dict[part_key]
                print(f"DEBUG:   Input {part_key} shape: {input_latents.shape}")
                with torch.no_grad():
                    # Use latent2origin for continuous latent decoding
                    recon_output = self.rvqvae_models[part].latent2origin(input_latents)
                    if isinstance(recon_output, tuple):
                        decoded[part] = recon_output[0]  # Take first element if tuple
                        print(f"DEBUG:   Decoded {part} shape (from tuple): {decoded[part].shape}")
                    else:
                        decoded[part] = recon_output
                        print(f"DEBUG:   Decoded {part} shape: {decoded[part].shape}")
            else:
                if part not in self.rvqvae_models:
                    print(f"DEBUG:   WARNING: {part} not in RVQVAE models")
                if part_key not in latents_dict:
                    print(f"DEBUG:   WARNING: {part_key} not in input latents_dict")
        
        print(f"DEBUG: [GestureProcessor.decode_gesture] Final decoded parts: {list(decoded.keys())}")
        return decoded


class OmnigesA2GModel(nn.Module):
    """
    Omniges A2G Model: Audio to Gesture Generation
    
    Architecture:
    1. Audio Fusion: WavLM + MFCC → 512dim  
    2. Text Encoder: T5/LLaMA → 1024dim
    3. MMDiT Backbone: (text=1024, audio=512, gesture=415)
    4. Gesture Decoder: 4x RVQVAE decoders
    """
    
    def __init__(
        self,
        # Model dimensions
        dim_text: int = 1024,
        dim_audio_fuse: int = 512,
        dim_gesture_total: int = 415,  # 78+180+57+100
        
        # Audio fusion config
        dim_wavlm: int = 768,
        dim_mfcc: int = 128,
        audio_fusion_heads: int = 8,
        audio_fusion_layers: int = 2,
        
        # MMDiT config
        mmdit_depth: int = 12,
        mmdit_heads: int = 8,
        mmdit_residual_streams: int = 4,
        
        # RVQVAE checkpoints
        rvqvae_ckpts: Dict[str, str] = None,
        
        # Text encoder config
        text_vae_path: Optional[str] = None,
        
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.body_parts = ["upper", "hands", "lower_trans", "face"]
        self.part_dims = {"upper": 78, "hands": 180, "lower_trans": 57, "face": 100}
        
        # 1. Audio Fusion Module
        self.audio_fusion = AudioFusionModule(
            dim_wavlm=dim_wavlm,
            dim_mfcc=dim_mfcc,
            dim_fuse=dim_audio_fuse,
            num_heads=audio_fusion_heads,
            num_layers=audio_fusion_layers
        )
        
        # 2. WavLM Encoder (pre-trained)
        self.wavlm = self._load_wavlm()
        
        # 3. MFCC Encoder  
        self.mfcc_encoder = MFCCEncoder(n_mfcc=dim_mfcc, hop_length=520)
        
        # 4. Text Encoder (optional for A2G, but keep for future T2G)
        self.text_encoder = None
        if text_vae_path:
            self.text_encoder = self._load_text_vae(text_vae_path)
            
        # 5. MMDiT Backbone
        self.backbone = MMDiT(
            depth=mmdit_depth,
            dim_modalities=(dim_text, dim_audio_fuse, dim_gesture_total),
            num_residual_streams=mmdit_residual_streams,
            dim_head=64,
            heads=mmdit_heads,
            dim_cond=dim_audio_fuse  # Enable time conditioning
        )
        
        # 6. Gesture Processor
        if rvqvae_ckpts is None:
            rvqvae_ckpts = {
                "upper": "ckpt/net_300000_upper.pth",
                "hands": "ckpt/net_300000_hands.pth", 
                "lower_trans": "ckpt/net_300000_lower.pth",
                "face": "ckpt/net_300000_face.pth"
            }
        
        self.gesture_processor = GestureProcessor(rvqvae_ckpts, device)
        
        # 7. Gesture Generation Heads (continuous latent prediction)
        # Each head predicts continuous latent features for RVQVAE decoding
        code_dim = 128  # From checkpoint analysis
        self.gesture_heads = nn.ModuleDict({
            part: nn.Linear(dim_gesture_total, code_dim)  # Continuous latent dim
            for part, dim in self.part_dims.items()
        })
        
        # 8. Time conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim_audio_fuse),
            nn.SiLU(),
            nn.Linear(dim_audio_fuse, dim_audio_fuse)
        )
        
    def _load_wavlm(self) -> WavLM:
        """Load pre-trained WavLM model."""
        # Default to WavLM-Base+ if path not specified
        wavlm_path = "models/wavlm/WavLM-Base+.pt"
        if os.path.exists(wavlm_path):
            checkpoint = torch.load(wavlm_path, map_location='cpu')
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        else:
            # Fallback: create default WavLM config
            cfg = WavLMConfig()
            model = WavLM(cfg)
            print("Warning: WavLM checkpoint not found, using random initialization")
            
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def _load_text_vae(self, path: str) -> LLamaForLatentConnector:
        """Load pre-trained text VAE model (optional for A2G)."""
        # This is for future T2G implementation
        return None
    
    def encode_audio(
        self, 
        audio_waveform: torch.Tensor,  # (B, T_audio)
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode audio waveform to fused features
        
        Args:
            audio_waveform: Raw audio waveform (B, T_audio) at 16kHz
            audio_mask: Optional mask for audio
            
        Returns:
            Fused audio features (B, T_frames, 512)
        """
        # 1. Extract WavLM features
        with torch.no_grad():
            wavlm_output = self.wavlm.extract_features(audio_waveform)
            if isinstance(wavlm_output, tuple):
                wavlm_feats = wavlm_output[0]  # (B, T_wavlm, 768)
            else:
                wavlm_feats = wavlm_output  # (B, T_wavlm, 768)
        
        # 2. Extract MFCC features  
        mfcc_feats = self.mfcc_encoder(audio_waveform)  # (B, T_mfcc, 128)
        
        # 3. Align temporal dimensions (interpolate to common frame rate)
        target_frames = min(wavlm_feats.size(1), mfcc_feats.size(1))
        
        if wavlm_feats.size(1) != target_frames:
            wavlm_feats = F.interpolate(
                wavlm_feats.transpose(1, 2), 
                size=target_frames, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
        if mfcc_feats.size(1) != target_frames:
            mfcc_feats = F.interpolate(
                mfcc_feats.transpose(1, 2),
                size=target_frames,
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # 4. Bidirectional cross-attention fusion
        fused_audio = self.audio_fusion(wavlm_feats, mfcc_feats, audio_mask)
        
        return fused_audio  # (B, T_frames, 512)
    
    def forward(
        self,
        audio_waveform: torch.Tensor,           # (B, T_audio) 
        timesteps: torch.Tensor,                 # (B,)
        target_gesture: Optional[Dict[str, torch.Tensor]] = None,  # Training targets
        text_input: Optional[torch.Tensor] = None,  # Future T2G support
        audio_mask: Optional[torch.Tensor] = None,
        return_codes: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for A2G generation
        
        Args:
            audio_waveform: Raw audio input (B, T_audio)
            timesteps: Diffusion timesteps (B,)
            target_gesture: Optional target gestures for training
            text_input: Optional text input for future T2G
            audio_mask: Optional audio attention mask
            return_codes: Whether to return quantized codes
            
        Returns:
            Dictionary containing generated gestures and intermediate representations
        """
        batch_size = audio_waveform.size(0)
        
        # 1. Audio Encoding
        audio_features = self.encode_audio(audio_waveform, audio_mask)  # (B, T, 512)
        seq_len = audio_features.size(1)
        
        # 2. Text Encoding (placeholder for A2G, filled with zeros)
        if text_input is not None and self.text_encoder is not None:
            text_features = self.text_encoder.encode(text_input)  # (B, T, 1024)
        else:
            # Zero text features for pure A2G task
            text_features = torch.zeros(
                batch_size, seq_len, 1024, 
                device=audio_features.device,
                dtype=audio_features.dtype
            )
        
        # 3. Gesture Features (noise or target for training)
        if target_gesture is not None:
            # Training mode: use target gestures + noise
            gesture_features = self._prepare_gesture_targets(target_gesture, timesteps)
        else:
            # Inference mode: start with noise
            gesture_features = torch.randn(
                batch_size, seq_len, sum(self.part_dims.values()),
                device=audio_features.device
            )
        
        # 4. Time embedding
        time_emb = self.time_embed(
            self._timestep_embedding(timesteps, 256)
        )  # (B, 512)
        
        # 5. MMDiT Backbone
        modality_tokens = (text_features, audio_features, gesture_features)
        
        output_tokens = self.backbone(
            modality_tokens=modality_tokens,
            time_cond=time_emb
        )
        
        # Extract gesture output
        _, _, gesture_output = output_tokens  # (B, T, 415)
        
        # 6. Generate gesture latents for each body part (continuous approach)
        gesture_latents = {}
        gesture_reconstructed = {}
        
        for part in self.body_parts:
            if part in self.gesture_processor.rvqvae_models:
                # Predict continuous latents from full gesture features
                part_latents = self.gesture_heads[part](gesture_output)  # (B, T, code_dim=128)
                gesture_latents[f"{part}_latents"] = part_latents
                
                # Decode to gesture space using continuous latents
                if return_codes:
                    recon_dict = self.gesture_processor.decode_gesture(
                        latents_dict={f"{part}_latents": part_latents}
                    )
                    if part in recon_dict:
                        gesture_reconstructed[part] = recon_dict[part]
        
        return {
            "audio_features": audio_features,
            "gesture_features": gesture_output, 
            "gesture_latents": gesture_latents,  # Continuous latents instead of discrete codes
            "gesture_reconstructed": gesture_reconstructed,
            "text_features": text_features,
            "time_embedding": time_emb
        }
    
    def _prepare_gesture_targets(
        self, 
        target_gesture: Dict[str, torch.Tensor], 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Prepare target gesture features for training (add noise for diffusion)."""
        # Concatenate all body parts
        parts_list = []
        for part in self.body_parts:
            if part in target_gesture:
                parts_list.append(target_gesture[part])
        
        if parts_list:
            gesture_concat = torch.cat(parts_list, dim=-1)  # (B, T, 415)
            
            # Add noise based on timesteps (diffusion training)
            noise_scale = (timesteps.float() / 1000.0).view(-1, 1, 1)
            noise = torch.randn_like(gesture_concat) * noise_scale
            
            return gesture_concat + noise
        else:
            # Return zero tensor if no targets
            batch_size = timesteps.size(0)
            return torch.zeros(
                batch_size, 64, sum(self.part_dims.values()),  # Default 64 frames
                device=timesteps.device
            )
    
    def _timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Create timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class A2GLoss(nn.Module):
    """Loss function for A2G training (following shortcut_rvqvae_trainer.py approach)."""
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        commitment_weight: float = 0.02,
        velocity_weight: float = 0.1,
        acceleration_weight: float = 0.1,
        latent_weight: float = 1.0
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.commitment_weight = commitment_weight  
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.latent_weight = latent_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute A2G training loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss dictionary
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Latent Space Loss (primary loss, following shortcut_rvqvae_trainer.py)
        if "gesture_latents" in predictions and "target_latents" in targets:
            latent_loss = 0.0
            for part in ["upper", "hands", "lower_trans", "face"]:
                if f"{part}_latents" in predictions["gesture_latents"] and f"{part}_latents" in targets["target_latents"]:
                    pred_latents = predictions["gesture_latents"][f"{part}_latents"]  # (B, T, 128)
                    target_latents = targets["target_latents"][f"{part}_latents"]    # (B, T, 128)
                    
                    # Align temporal dimensions if needed
                    if pred_latents.shape[1] != target_latents.shape[1]:
                        target_aligned = F.interpolate(
                            target_latents.transpose(1, 2), 
                            size=pred_latents.shape[1], 
                            mode='linear'
                        ).transpose(1, 2)
                    else:
                        target_aligned = target_latents
                    
                    latent_loss += self.mse_loss(pred_latents, target_aligned)
            
            losses["latent_loss"] = latent_loss * self.latent_weight
            total_loss += losses["latent_loss"]
        
        # 2. Optional Gesture Reconstruction Loss (secondary)
        if "gesture_reconstructed" in predictions and "target_gesture" in targets:
            recon_loss = 0.0
            for part in ["upper", "hands", "lower_trans", "face"]:
                if part in predictions["gesture_reconstructed"] and part in targets["target_gesture"]:
                    pred_recon = predictions["gesture_reconstructed"][part]  # (B, T_recon, D)
                    target_pose = targets["target_gesture"][part]           # (B, T_target, D)
                    
                    # Align temporal dimensions if they don't match (due to RVQVAE up/downsampling)
                    if pred_recon.shape[1] != target_pose.shape[1]:
                        # Interpolate target to match prediction length
                        target_aligned = F.interpolate(
                            target_pose.transpose(1, 2), 
                            size=pred_recon.shape[1], 
                            mode='linear'
                        ).transpose(1, 2)
                    else:
                        target_aligned = target_pose
                    
                    recon_loss += self.mse_loss(pred_recon, target_aligned)
            
            losses["gesture_recon"] = recon_loss * 0.1  # Lower weight
            total_loss += losses["gesture_recon"]
        
        # 3. Velocity Consistency Loss (optional)
        if self.velocity_weight > 0 and "gesture_reconstructed" in predictions:
            vel_loss = self._compute_velocity_loss(predictions["gesture_reconstructed"])
            losses["velocity"] = vel_loss * self.velocity_weight
            total_loss += losses["velocity"]
        
        losses["total"] = total_loss
        return losses
    
    def _compute_velocity_loss(self, gesture_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute velocity consistency loss."""
        vel_loss = 0.0
        
        for part, gesture in gesture_dict.items():
            # Compute velocity  
            vel = gesture[:, 1:] - gesture[:, :-1]  # (B, T-1, D)
            vel_smooth = vel[:, 1:] - vel[:, :-1]   # (B, T-2, D)
            
            # L1 loss on velocity smoothness
            vel_loss += F.l1_loss(vel_smooth, torch.zeros_like(vel_smooth))
        
        return vel_loss


def create_omniges_a2g_model(config_path: str = None, **kwargs) -> OmnigesA2GModel:
    """Factory function to create Omniges A2G model from config."""
    
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model parameters from config
        model_config = {
            "dim_wavlm": config.get("audio", {}).get("dim_wavlm", 768),
            "dim_mfcc": config.get("audio", {}).get("dim_other", 128),
            "dim_audio_fuse": config.get("audio", {}).get("fusion", {}).get("dim_fuse", 512),
            "audio_fusion_heads": config.get("audio", {}).get("fusion", {}).get("num_heads", 8),
            "audio_fusion_layers": config.get("audio", {}).get("fusion", {}).get("num_layers", 2),
            "mmdit_depth": config.get("backbone", {}).get("layers", 12),
            "mmdit_heads": config.get("backbone", {}).get("heads", 8),
            "rvqvae_ckpts": config.get("gesture", {}).get("rvq_ckpt", {}),
        }
        
        model_config.update(kwargs)
        return OmnigesA2GModel(**model_config)
    
    else:
        return OmnigesA2GModel(**kwargs)


if __name__ == "__main__":
    # Test model creation
    model = create_omniges_a2g_model("configs/omniges/a2g_mmdit.yaml")
    print(f"Created Omniges A2G model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    audio_input = torch.randn(batch_size, audio_length)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(audio_input, timesteps)
        print("Model forward pass successful!")
        print(f"Audio features shape: {output['audio_features'].shape}")
        print(f"Gesture features shape: {output['gesture_features'].shape}")
        for part in ["upper", "hands", "lower_trans"]:
            if part in output["gesture_codes"]:
                print(f"{part} codes shape: {output['gesture_codes'][part].shape}")
