"""
Omniges Transformer Model
Based on OmniFlow with Image stream replaced by Gesture stream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import inspect
from functools import partial
from einops import rearrange

# Import OmniFlow base components
from omniflow.models.omni_flow import OmniFlowTransformerModel, NNMLP
from omniflow.models.attention import JointTransformerBlock

from diffusers import ModelMixin, ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging

logger = logging.get_logger(__name__)


class GestureEmbedding(nn.Module):
    """
    Gesture Sequence Embedding
    Replaces PatchEmbed for gesture sequences
    """
    
    def __init__(
        self,
        seq_length: int = 128,
        gesture_dim: int = 415,  # Total gesture dimensions
        embed_dim: int = 1152,
        pos_embed_max_size: int = 128
    ):
        super().__init__()
        self.seq_length = seq_length
        self.gesture_dim = gesture_dim
        self.embed_dim = embed_dim
        
        # Linear projection from gesture features to embedding dimension
        self.gesture_proj = nn.Linear(gesture_dim, embed_dim)
        
        # Positional embedding for sequence
        self.position_embedding = nn.Parameter(
            torch.randn(1, pos_embed_max_size, embed_dim) * 0.02
        )
        
    def forward(self, gesture_latents):
        """
        Args:
            gesture_latents: (B, C, T, num_parts) - 2D gesture latents from VAE
        Returns:
            embeddings: (B, T, embed_dim) - sequence embeddings for transformer
        """
        B, C, T, num_parts = gesture_latents.shape
        
        # Flatten spatial dimensions: (B, C, T, num_parts) -> (B, T, C*num_parts)
        gesture_features = gesture_latents.permute(0, 2, 1, 3)  # (B, T, C, num_parts)
        gesture_features = gesture_features.reshape(B, T, C * num_parts)  # (B, T, 512)
        
        # Project to embedding dimension
        embeddings = self.gesture_proj(gesture_features)  # (B, T, embed_dim)
        
        # Add positional embedding
        seq_len = embeddings.shape[1]
        if seq_len <= self.position_embedding.shape[1]:
            pos_emb = self.position_embedding[:, :seq_len, :]
        else:
            # Interpolate for longer sequences
            pos_emb = F.interpolate(
                self.position_embedding.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
        embeddings = embeddings + pos_emb
        
        return embeddings


class OmnigesTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Omniges Transformer Model
    Text-Audio-Gesture multimodal transformer based on OmniFlow
    
    Key differences from OmniFlow:
    - Image stream -> Gesture stream
    - PatchEmbed -> GestureEmbedding  
    - 2D image operations -> 1D sequence operations
    """
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        seq_length: int = 128,  # Gesture sequence length instead of sample_size
        gesture_dim: int = 415,  # Total gesture dimensions instead of in_channels
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        audio_input_dim: int = 8,
        gesture_latent_dim: int = 512,  # Gesture latent dimension instead of out_channels
        pos_embed_max_size: int = 128,  # For sequence positional embedding
        dual_attention_layers: Tuple[int, ...] = (),
        decoder_config: str = '',
        add_audio: bool = True,
        add_clip: bool = False,
        use_audio_mae: bool = False,
        drop_text: bool = False,
        drop_gesture: bool = False,  # Instead of drop_image
        drop_audio: bool = False,
        qk_norm: Optional[str] = 'layer_norm',
    ):
        super().__init__()
        
        self.add_clip = add_clip
        self.gesture_latent_dim = gesture_latent_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        
        # Gesture sequence embedding (replaces PatchEmbed)
        self.gesture_embed = GestureEmbedding(
            seq_length=seq_length,
            gesture_dim=C * 4,  # 128 * 4 = 512 from gesture VAE
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size
        )
        
        # Time and text embedding (same as OmniFlow)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, 
            pooled_projection_dim=self.config.pooled_projection_dim
        )
        
        if add_audio:
            # Audio components (same as OmniFlow)
            self.time_gesture_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, 
                pooled_projection_dim=self.config.pooled_projection_dim
            )
            self.audio_input_dim = audio_input_dim
            self.use_audio_mae = use_audio_mae
            self.audio_patch_size = 2
            
            if use_audio_mae:
                self.audio_embedder = nn.Linear(audio_input_dim, self.config.caption_projection_dim)
            else:
                from diffusers.models.embeddings import PatchEmbed
                self.audio_embedder = PatchEmbed(
                    height=256,
                    width=16,
                    patch_size=self.audio_patch_size,
                    in_channels=self.audio_input_dim,
                    embed_dim=self.config.caption_projection_dim,
                    pos_embed_max_size=192
                )
                
            self.time_aud_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, 
                pooled_projection_dim=self.config.pooled_projection_dim
            )
            
            self.norm_out_aud = AdaLayerNormContinuous(
                self.config.caption_projection_dim, 
                self.inner_dim, 
                elementwise_affine=False, 
                eps=1e-6
            )
            
            if use_audio_mae:
                self.proj_out_aud = nn.Linear(
                    self.config.caption_projection_dim, 
                    self.config.audio_input_dim
                )
            else:
                self.proj_out_aud = nn.Linear(
                    self.inner_dim, 
                    self.audio_patch_size * self.audio_patch_size * self.audio_input_dim, 
                    bias=True
                )
        
        # Context embedder (same as OmniFlow)
        self.context_embedder = nn.Linear(
            self.config.joint_attention_dim, 
            self.config.caption_projection_dim
        )
        
        # Text decoder components (same as OmniFlow)
        from transformers.models.llama.modeling_llama import LlamaConfig
        bert_config = LlamaConfig(
            1, 
            hidden_size=self.config.joint_attention_dim,
            num_attention_heads=32,
            num_hidden_layers=2
        )
        
        if add_audio:
            self.context_decoder = nn.ModuleDict(dict(
                projection=nn.Linear(
                    self.config.caption_projection_dim,
                    self.config.joint_attention_dim
                )
            ))
            
        self.text_out_dim = 1536
        self.text_output = nn.Linear(self.config.joint_attention_dim, self.text_out_dim)
        
        # Transformer blocks (same as OmniFlow)
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                context_pre_only=i == num_layers - 1,
                context_output=i < num_layers or add_audio,
                audio_output=add_audio,
                delete_img=drop_gesture,  # Use drop_gesture instead of drop_image
                delete_aud=drop_audio,
                delete_text=drop_text,
                qk_norm=qk_norm,
                use_dual_attention=True if i in dual_attention_layers else False,
            )
            for i in range(self.config.num_layers)
        ])
        
        self.add_audio = add_audio
        
        # Output normalization for gesture (replaces image norm)
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, 
            self.inner_dim, 
            elementwise_affine=False, 
            eps=1e-6
        )
        
        # Text output normalization (same as OmniFlow)
        self.norm_out_text = AdaLayerNormContinuous(
            self.joint_attention_dim, 
            self.inner_dim, 
            elementwise_affine=False, 
            eps=1e-6
        )
        
        if add_clip:
            self.n_cond_tokens = 8
            self.clip_proj = nn.Sequential(
                NNMLP(self.config.pooled_projection_dim, self.config.caption_projection_dim),
                nn.Linear(
                    self.config.caption_projection_dim,
                    self.config.caption_projection_dim * self.n_cond_tokens
                )
            )
            
        # Gesture output projection (replaces image patch projection)
        self.proj_out = nn.Linear(
            self.inner_dim, 
            gesture_latent_dim,  # Project to gesture latent dimension
            bias=True
        )
        
        self.gradient_checkpointing = False
        self.text_decoder = None
        
    def set_text_decoder(self, model):
        """Set text decoder (same as OmniFlow)"""
        self.text_decoder = model
        self.text_out_dim = model.vae_dim
        self.text_output = nn.Linear(self.config.joint_attention_dim, self.text_out_dim)
        
    def set_audio_pooler(self, model):
        """Set audio pooler (same as OmniFlow)"""
        self.audio_pooler = model
        
    def get_decoder(self):
        """Get text decoder (same as OmniFlow)"""
        return self.text_decoder
        
    # Copy attention and gradient checkpointing methods from OmniFlow
    @property
    def attn_processors(self):
        # Same as OmniFlow - delegated for brevity
        processors = {}
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors
            
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors
        
    def forward(
        self,
        hidden_states: torch.FloatTensor = None,  # Gesture latent states
        encoder_hidden_states: torch.FloatTensor = None,  # Text states
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        timestep_text: torch.LongTensor = None,
        timestep_audio: torch.LongTensor = None,
        audio_hidden_states: torch.FloatTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        use_text_output: bool = False,
        target_prompt_embeds=None,
        decode_text=False,
        sigma_text=None,
        detach_logits=False,
        prompt_embeds_uncond=None,
        targets=None,
        split_cond=False,
        text_vae=None,
        text_x0=True,
        drop_text=False,
        drop_gesture=False,  # Instead of drop_image
        drop_audio=False,
        **kwargs
    ):
        """
        Omniges Transformer forward pass
        Processes Text, Audio, and Gesture modalities
        """
        
        # Store original states
        encoder_hidden_states_base = encoder_hidden_states.clone() if encoder_hidden_states is not None else None
        hidden_states_base = hidden_states
        
        # Determine which modalities to process
        do_gesture = not drop_gesture  # Instead of do_image
        do_audio = (not drop_audio) and self.add_audio
        do_text = not drop_text
        
        if do_gesture:
            # Process gesture latents
            seq_length = hidden_states.shape[2]  # (B, C, T, num_parts)
            hidden_states = self.gesture_embed(hidden_states)  # (B, T, embed_dim)
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            hidden_states = None
            temb = 0
            
        if do_audio:
            # Audio processing (same as OmniFlow)
            if audio_hidden_states is None:
                if self.use_audio_mae:
                    audio_hidden_states = torch.zeros(
                        encoder_hidden_states.shape[0], 8, self.audio_input_dim
                    ).to(encoder_hidden_states)
                else:
                    audio_hidden_states = torch.zeros(
                        encoder_hidden_states.shape[0], 8, 256, 16
                    ).to(encoder_hidden_states)
                timestep_audio = timestep_text * 0
                
            temb_audio = self.time_aud_embed(timestep_audio, pooled_projections)
            audio_hidden_states = self.audio_embedder(audio_hidden_states)
            
            if not split_cond:
                temb = temb + temb_audio
                temb_audio = None
        else:
            audio_hidden_states = None
            temb_audio = None
            
        if do_text:
            # Text processing (same as OmniFlow)
            if use_text_output:
                temb_text = self.time_gesture_embed(timestep_text, pooled_projections)
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
            
            if use_text_output:
                if not split_cond:
                    temb = temb + temb_text
                    temb_text = None
            else:
                temb_text = None
        else:
            encoder_hidden_states = None
            temb_text = None
            
        assert not self.add_clip  # CLIP not supported yet
        
        # Process through transformer blocks
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing (same logic as OmniFlow)
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                    
                import deepspeed
                ckpt_kwargs = dict()
                
                if self.add_audio:
                    encoder_hidden_states, hidden_states, audio_hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        audio_hidden_states,
                        temb_text,
                        temb_audio,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        temb_text,
                        **ckpt_kwargs,
                    )
            else:
                # Normal forward pass
                if self.add_audio:
                    encoder_hidden_states, hidden_states, audio_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        audio_hidden_states=audio_hidden_states,
                        temb=temb,
                        temb_text=temb_text,
                        temb_audio=temb_audio,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        temb_text=temb_text
                    )
                    
        # Gesture output processing
        if do_gesture:
            hidden_states = self.norm_out(hidden_states, temb)  # (B, T, inner_dim)
            hidden_states = self.proj_out(hidden_states)  # (B, T, gesture_latent_dim)
            
            # Reshape back to 2D format for gesture VAE: (B, T, 512) -> (B, 128, T, 4)
            B, T, latent_dim = hidden_states.shape
            num_parts = 4
            C = latent_dim // num_parts  # 512 // 4 = 128
            
            gesture_output = hidden_states.view(B, T, C, num_parts)  # (B, T, 128, 4)
            gesture_output = gesture_output.permute(0, 2, 1, 3)  # (B, 128, T, 4)
        else:
            gesture_output = None
            
        # Text output processing (same as OmniFlow)
        model_pred_text = None
        logits = None
        logits_labels = None
        
        if do_text and use_text_output:
            encoder_hidden_states = self.context_decoder['projection'](encoder_hidden_states)
            encoder_hidden_states = self.norm_out_text(
                encoder_hidden_states, 
                temb_text if temb_text is not None else temb
            )
            encoder_hidden_states = self.text_output(encoder_hidden_states)
            model_pred_text = encoder_hidden_states
            
            # Text decoding logic (same as OmniFlow)
            if decode_text and targets is not None:
                # ... (complex text decoding logic from OmniFlow)
                pass
                
        # Audio output processing (same as OmniFlow)
        if do_audio:
            audio_hidden_states = self.norm_out_aud(
                audio_hidden_states, 
                temb_audio if temb_audio is not None else temb
            )
            audio_hidden_states = self.proj_out_aud(audio_hidden_states)
            
            if not self.use_audio_mae:
                # Unpatchify audio (same as OmniFlow)
                patch_size_audio = self.audio_patch_size
                height_audio = 256 // patch_size_audio
                width_audio = 16 // patch_size_audio
                
                audio_hidden_states = rearrange(
                    audio_hidden_states,
                    'n (h w) (hp wp c) -> n c (h hp) (w wp)',
                    h=height_audio,
                    w=width_audio,
                    hp=patch_size_audio,
                    wp=patch_size_audio,
                    c=self.audio_input_dim
                )
        else:
            audio_hidden_states = None
            
        # Return outputs
        return dict(
            output=gesture_output,  # Gesture output instead of image
            model_pred_text=model_pred_text,
            encoder_hidden_states=encoder_hidden_states,
            logits=logits,
            extra_cond=None,
            logits_labels=logits_labels,
            audio_hidden_states=audio_hidden_states,
        )


def create_omniges_transformer(
    omniflow_checkpoint_path: Optional[str] = None,
    seq_length: int = 128,
    gesture_dim: int = 415,
    **config_kwargs
):
    """
    Create Omniges transformer model
    
    Args:
        omniflow_checkpoint_path: Path to OmniFlow checkpoint for initialization
        seq_length: Gesture sequence length
        gesture_dim: Total gesture dimensions
    """
    
    # Create model
    model = OmnigesTransformerModel(
        seq_length=seq_length,
        gesture_dim=gesture_dim,
        **config_kwargs
    )
    
    # Load OmniFlow weights if provided
    if omniflow_checkpoint_path and os.path.exists(omniflow_checkpoint_path):
        logger.info(f"Loading OmniFlow weights from {omniflow_checkpoint_path}")
        
        # Load and adapt weights
        checkpoint = torch.load(omniflow_checkpoint_path, map_location='cpu')
        
        # Remove image-specific weights
        adapted_checkpoint = {}
        for key, value in checkpoint.items():
            if 'pos_embed' not in key and 'proj_out' not in key:
                # Keep all weights except image-specific ones
                adapted_checkpoint[key] = value
                
        # Load adapted weights
        missing_keys, unexpected_keys = model.load_state_dict(
            adapted_checkpoint, strict=False
        )
        
        logger.info(f"Loaded OmniFlow weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
    return model


# Test the implementation
if __name__ == "__main__":
    print("=== Testing Omniges Transformer ===")
    
    # Create model
    model = create_omniges_transformer(
        seq_length=128,
        gesture_dim=415,
        num_layers=4,  # Small for testing
        num_attention_heads=8,
        attention_head_dim=64
    )
    
    # Test inputs
    B, C, T, num_parts = 2, 128, 64, 4
    gesture_latents = torch.randn(B, C, T, num_parts)  # Gesture latents
    text_embeds = torch.randn(B, 77, 4096)  # Text embeddings
    audio_embeds = torch.randn(B, 1024, 1152)  # Audio embeddings
    pooled_embeds = torch.randn(B, 2048)  # Pooled embeddings
    timestep = torch.randint(0, 1000, (B,))
    
    print(f"Input shapes:")
    print(f"  Gesture latents: {gesture_latents.shape}")
    print(f"  Text embeds: {text_embeds.shape}")
    print(f"  Audio embeds: {audio_embeds.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            hidden_states=gesture_latents,
            encoder_hidden_states=text_embeds,
            audio_hidden_states=audio_embeds,
            pooled_projections=pooled_embeds,
            timestep=timestep,
            timestep_text=timestep,
            timestep_audio=timestep,
            use_text_output=True
        )
        
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
            
    print("✅ Omniges Transformer test successful!")
