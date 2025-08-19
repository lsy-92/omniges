import torch
import torch.nn as nn
from typing import Optional, Dict


class AudioAttentionFusion(nn.Module):
    """WavLM과 MFCC/WavEncoder 임베딩을 교차어텐션과 게이팅으로 융합."""

    def __init__(self, dim_wavlm: int, dim_other: int, dim_fuse: int = 512, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.dim_fuse = dim_fuse
        self.proj_wavlm = nn.Linear(dim_wavlm, dim_fuse)
        self.proj_other = nn.Linear(dim_other, dim_fuse)
        self.ln_w = nn.LayerNorm(dim_fuse)
        self.ln_o = nn.LayerNorm(dim_fuse)

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.ModuleDict(
                    dict(
                        attn_wq_o_kv=nn.MultiheadAttention(dim_fuse, num_heads, batch_first=True),
                        attn_oq_w_kv=nn.MultiheadAttention(dim_fuse, num_heads, batch_first=True),
                        ln1=nn.LayerNorm(dim_fuse),
                        ln2=nn.LayerNorm(dim_fuse),
                        ffn=nn.Sequential(
                            nn.Linear(dim_fuse, dim_fuse * 4), nn.GELU(), nn.Linear(dim_fuse * 4, dim_fuse)
                        ),
                    )
                )
            )
        self.layers = nn.ModuleList(layers)
        self.gate = nn.Sequential(nn.Linear(dim_fuse * 2, dim_fuse), nn.GELU(), nn.Linear(dim_fuse, dim_fuse), nn.Sigmoid())

    def forward(self, h_wavlm: torch.Tensor, h_other: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """입력: [B, T, D1], [B, T, D2]; 출력: [B, T, dim_fuse]"""
        xw = self.ln_w(self.proj_wavlm(h_wavlm))
        xo = self.ln_o(self.proj_other(h_other))
        for blk in self.layers:
            # Q: wavlm, KV: other
            q = blk.ln1(xw)
            k = xo
            v = xo
            a1, _ = blk.attn_wq_o_kv(q, k, v, key_padding_mask=attn_mask)
            xw = xw + a1

            # Q: other, KV: wavlm
            q2 = blk.ln2(xo)
            k2 = xw
            v2 = xw
            a2, _ = blk.attn_oq_w_kv(q2, k2, v2, key_padding_mask=attn_mask)
            xo = xo + a2

            # gating fusion + FFN on fused representation
            g = self.gate(torch.cat([xw, xo], dim=-1))
            xf = g * xw + (1.0 - g) * xo
            xf = xf + blk.ffn(xf)
            xw, xo = xf, xf

        return xw  # fused


class OmniGesModel(nn.Module):
    """Omniges 스켈레톤: mmdit 백본과 모달 인코더/헤드 연결을 위한 인터페이스.

    이 스켈레톤은 추후 omniflow/models/omni_flow.py 및 mmdit 백본과 연동되도록 설계됨.
    현재는 오디오 융합과 제스처/텍스트/오디오 인코더 자리표시자만 포함.
    """

    def __init__(
        self,
        dim_text: int = 1024,
        dim_wavlm: int = 768,
        dim_other_audio: int = 80,
        dim_fuse: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        gesture_code_dims: Dict[str, int] = None,
    ):
        super().__init__()
        gesture_code_dims = gesture_code_dims or {"upper": 78, "hands": 180, "lower_trans": 57, "face": 100}

        # 모달 투영
        self.text_proj = nn.Linear(dim_text, dim_fuse)
        self.audio_fusion = AudioAttentionFusion(dim_wavlm, dim_other_audio, dim_fuse, num_heads, num_layers)

        # 제스처 헤드(코드 예측형) 자리표시자: 부위별 로짓 차원은 추후 RVQVAE 코드북 크기에 맞춤
        self.gesture_heads = nn.ModuleDict({k: nn.Linear(dim_fuse, v) for k, v in gesture_code_dims.items()})

        # 백본 자리표시자: mmdit와 연결 예정
        self.backbone = None

    def encode_text(self, text_latent: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text_latent)

    def encode_audio(self, wavlm_feats: torch.Tensor, other_feats: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.audio_fusion(wavlm_feats, other_feats, attn_mask)

    def encode_gesture(self, gesture_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 자리표시자: 필요 시 RVQVAE 인코더 연동
        return gesture_inputs

    def forward(
        self,
        text_latent: Optional[torch.Tensor] = None,
        wavlm_feats: Optional[torch.Tensor] = None,
        other_audio_feats: Optional[torch.Tensor] = None,
        gesture_inputs: Optional[Dict[str, torch.Tensor]] = None,
        timesteps: Optional[torch.Tensor] = None,
        task: str = "a2g",
    ) -> Dict[str, torch.Tensor]:
        """간단한 포워드: 오디오 융합 → 제스처 헤드 예측(스켈레톤)."""
        out: Dict[str, torch.Tensor] = {}
        h_text = self.encode_text(text_latent) if text_latent is not None else None
        h_audio = None
        if wavlm_feats is not None and other_audio_feats is not None:
            h_audio = self.encode_audio(wavlm_feats, other_audio_feats)

        h = h_audio if h_audio is not None else h_text
        if h is None:
            raise ValueError("at least one of text or audio inputs must be provided")

        if task in ("a2g", "t2g"):
            for part, head in self.gesture_heads.items():
                out[f"logits_{part}"] = head(h)
            # 연속 포즈(6D) 예측용 간단 헤드(스켈레톤): 파트 로짓 평균을 6D 포즈로 투영
            # 기본값: 55*6=330 차원으로 가정
            pose_dim = 330
            if not hasattr(self, "pose6d_head"):
                self.pose6d_head = nn.Linear(h.shape[-1], pose_dim)
            out["pose6d"] = self.pose6d_head(h)
        else:
            out["features"] = h
        return out
