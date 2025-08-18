import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Transformer1D(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: int,
        nhead: int,
        num_layers: int,
        cond_channels: int,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.cond_drop_prob = cond_drop_prob

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # Input projection
        self.init_conv = nn.Conv1d(channels, dim, 1)

        # Conditioning projection
        self.cond_proj = nn.Conv1d(cond_channels, dim, 1)
        self.null_cond_emb = nn.Parameter(torch.randn(1, dim, 1))


        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=nhead, 
            dim_feedforward=dim*4, 
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.final_conv = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.Mish(),
            nn.Conv1d(dim, channels, 1)
        )

    def forward(self, x, time, cond=None, cond_drop_prob=None):
        # x: (batch, channels, seq_len)
        # time: (batch,)
        # cond: (batch, cond_channels, seq_len)
        
        drop_prob = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob

        x = self.init_conv(x)

        # Time embedding
        t_emb = self.time_mlp(time)
        x = x + t_emb.unsqueeze(-1)

        # Conditioning
        if cond is not None:
            if drop_prob > 0.0:
                mask = (torch.rand(x.shape[0], device=x.device) < drop_prob).view(-1, 1, 1)
                cond_emb = self.cond_proj(cond)
                null_emb = self.null_cond_emb.expand_as(cond_emb)
                cond_emb = torch.where(mask, null_emb, cond_emb)
            else:
                cond_emb = self.cond_proj(cond)
            
            if cond_emb.shape[-1] != x.shape[-1]:
                cond_emb = F.interpolate(cond_emb, size=x.shape[-1], mode='linear', align_corners=False)
            x = x + cond_emb

        # Transformer expects (batch, seq_len, dim)
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)

        # Final projection
        x = self.final_conv(x)

        return x
