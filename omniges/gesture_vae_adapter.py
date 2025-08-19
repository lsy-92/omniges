import os
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq.model import RVQVAE


def _build_lower_masks() -> Dict[str, torch.Tensor]:
    """Return boolean index masks for upper, hands, lower(+trans), face over 330/333-dim pose.

    Assumes 55 joints with 6D rotation -> 330 dims. Some configs append 3-dim translation to reach 333.
    """
    # upper body joints by index
    upper_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    upper = []
    for j in upper_joints:
        upper.extend([j * 6 + k for k in range(6)])

    # hands joints (25..54)
    hands_joints = list(range(25, 55))
    hands = []
    for j in hands_joints:
        hands.extend([j * 6 + k for k in range(6)])

    # lower body joints plus 3-dim translation (indices 330,331,332)
    lower_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower = []
    for j in lower_joints:
        lower.extend([j * 6 + k for k in range(6)])
    lower_trans = lower + [330, 331, 332]

    # face block (100 dims) conventionally after 333; here kept empty unless present in input
    face = list(range(333, 433))

    return {
        "upper": torch.tensor(upper, dtype=torch.long),
        "hands": torch.tensor(hands, dtype=torch.long),
        "lower": torch.tensor(lower, dtype=torch.long),
        "lower_trans": torch.tensor(lower_trans, dtype=torch.long),
        "face": torch.tensor(face, dtype=torch.long),
    }


class ConcatRVQVAEAdapter(nn.Module):
    """Wrap four RVQVAEs (upper/hands/lower(+trans)/face) and concat their latents for OmniFlow.

    - encode: pose [B, T, D] -> concat latent [B, Cin, H, W] for transformer
    - decode: transformer latents [B, Cin, H, W] -> pose [B, T, D]
    """

    def __init__(
        self,
        upper_ckpt: str,
        hands_ckpt: str,
        lower_ckpt: str,
        face_ckpt: Optional[str] = None,
        *,
        nb_code: int = 512,
        code_dim: int = 256,
        num_quantizers: int = 6,
        shared_codebook: bool = False,
        quantize_dropout_prob: float = 0.2,
        fuse_in_channels: int = 16,
        sample_size: int = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.masks = _build_lower_masks()
        self.sample_size = sample_size
        self.fuse_in_channels = fuse_in_channels

        # Build lightweight args for RVQVAE
        def mk_args():
            # Provide required attributes for quantizer
            return SimpleNamespace(
                num_quantizers=num_quantizers,
                shared_codebook=shared_codebook,
                quantize_dropout_prob=quantize_dropout_prob,
                mu=0.99,
            )

        # Instantiate RVQVAE for each part
        self.vq_upper = RVQVAE(mk_args(), input_width=78, nb_code=nb_code, code_dim=code_dim,
                               output_emb_width=code_dim)
        self.vq_hands = RVQVAE(mk_args(), input_width=180, nb_code=nb_code, code_dim=code_dim,
                               output_emb_width=code_dim)
        self.vq_lower = RVQVAE(mk_args(), input_width=57, nb_code=nb_code, code_dim=code_dim,
                               output_emb_width=code_dim)
        self.vq_face = None
        if face_ckpt is not None and os.path.exists(face_ckpt):
            self.vq_face = RVQVAE(mk_args(), input_width=100, nb_code=nb_code, code_dim=code_dim,
                                  output_emb_width=code_dim)

        # Load checkpoints (expects {'net': state_dict})
        def load(model: nn.Module, path: str):
            sd = torch.load(path, map_location="cpu")
            key = "net" if isinstance(sd, dict) and "net" in sd else None
            missing, unexpected = model.load_state_dict(sd[key] if key else sd, strict=False)
            if missing or unexpected:
                print(f"[RVQVAE load] missing={len(missing)} unexpected={len(unexpected)} from {path}")
            model.to(self.device).eval()

        load(self.vq_upper, upper_ckpt)
        load(self.vq_hands, hands_ckpt)
        load(self.vq_lower, lower_ckpt)
        if self.vq_face is not None and face_ckpt is not None and os.path.exists(face_ckpt):
            load(self.vq_face, face_ckpt)

        # Projection heads
        # concat channels -> in_channels for transformer (default 16)
        concat_dim = code_dim * (3 + (1 if self.vq_face is not None else 0))
        self.proj_to_in = nn.Conv1d(concat_dim, fuse_in_channels, kernel_size=1)
        self.proj_from_in = nn.Conv1d(fuse_in_channels, concat_dim, kernel_size=1)
        self.to(self.device)

    @torch.no_grad()
    def encode(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """pose_seq: [B, T, D] -> latents_image: [B, Cin, H, W] with H=W=sample_size.
        """
        b, t, d = pose_seq.shape
        pose_seq = pose_seq.to(self.device)

        # Slice parts
        def safe_slice(idx: torch.Tensor) -> torch.Tensor:
            valid = idx[idx < d]
            return pose_seq.index_select(-1, valid)

        upper = safe_slice(self.masks["upper"])  # [B,T,78]
        hands = safe_slice(self.masks["hands"])  # [B,T,180]
        lower_t = safe_slice(self.masks["lower_trans"]) if d >= 333 else safe_slice(self.masks["lower"])  # [B,T,57|54]
        # Face optional
        face = None
        if self.vq_face is not None and d >= 433:
            face = safe_slice(self.masks["face"])  # [B,T,100]

        # Map to latents (time-downsampling inside encoders). Shapes: [B, Tq, Cq]
        z_upper = self.vq_upper.map2latent(upper)
        z_hands = self.vq_hands.map2latent(hands)
        # lower encoder expects 57 dims; if we sliced only 54, pad zeros for trans
        if lower_t.shape[-1] == 54:
            pad = torch.zeros((b, t, 3), device=lower_t.device, dtype=lower_t.dtype)
            lower_in = torch.cat([lower_t, pad], dim=-1)
        else:
            lower_in = lower_t
        z_lower = self.vq_lower.map2latent(lower_in)
        zs = [z_upper, z_hands, z_lower]
        if self.vq_face is not None and face is not None:
            zs.append(self.vq_face.map2latent(face))

        # Align time by linear interpolation to a common length
        lens = [z.shape[1] for z in zs]
        target_len = max(lens)
        zs_aligned = []
        for z in zs:
            if z.shape[1] != target_len:
                zt = F.interpolate(z.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False)
                z = zt.permute(0, 2, 1)
            zs_aligned.append(z)

        z_cat = torch.cat(zs_aligned, dim=-1)  # [B, Tq, Ccat]
        # Project to transformer in-channels and fold to square
        z_cat_c = z_cat.permute(0, 2, 1)  # [B, Ccat, Tq]
        z_in = self.proj_to_in(z_cat_c)   # [B, Cin, Tq]

        t_sq = self.sample_size * self.sample_size
        if z_in.shape[-1] != t_sq:
            z_in = F.interpolate(z_in, size=t_sq, mode="linear", align_corners=False)
        latents_2d = z_in.view(z_in.shape[0], z_in.shape[1], self.sample_size, self.sample_size)
        return latents_2d

    @torch.no_grad()
    def decode(self, latents_2d: torch.Tensor, T_out: int, D_out: int = 333) -> torch.Tensor:
        """latents_2d: [B, Cin, H, W] -> pose_seq: [B, T_out, D_out].
        """
        b, cin, h, w = latents_2d.shape
        assert h == self.sample_size and w == self.sample_size, "Unexpected latent spatial size"
        z_in = latents_2d.view(b, cin, h * w)
        z_cat_c = self.proj_from_in(z_in)  # [B, Ccat, Tq]
        z_cat = z_cat_c.permute(0, 2, 1)   # [B, Tq, Ccat]

        # Split channels
        parts = 3 + (1 if self.vq_face is not None else 0)
        per = z_cat.shape[-1] // parts
        z_up, z_hd, z_lo = z_cat[..., :per], z_cat[..., per:2 * per], z_cat[..., 2 * per:3 * per]
        z_fc = z_cat[..., 3 * per:4 * per] if parts == 4 else None

        # Interpolate to decoder's temporal resolution if needed
        def to_t(z, t_out):
            if z.shape[1] != t_out:
                zt = F.interpolate(z.permute(0, 2, 1), size=t_out, mode="linear", align_corners=False)
                z = zt.permute(0, 2, 1)
            return z

        z_up = to_t(z_up, T_out)
        z_hd = to_t(z_hd, T_out)
        z_lo = to_t(z_lo, T_out)
        if z_fc is not None:
            z_fc = to_t(z_fc, T_out)

        # Decode each part back to pose segments
        rec_up, _, _ = self.vq_upper.latent2origin(z_up)
        rec_hd, _, _ = self.vq_hands.latent2origin(z_hd)
        rec_lo, _, _ = self.vq_lower.latent2origin(z_lo)
        rec_fc = None
        if self.vq_face is not None and z_fc is not None:
            rec_fc, _, _ = self.vq_face.latent2origin(z_fc)

        # Assemble full pose [B, T, D_out]
        out = torch.zeros((b, T_out, D_out), device=latents_2d.device, dtype=rec_up.dtype)
        def place(seg, idx):
            idx = idx[idx < D_out]
            out.index_copy_(dim=2, index=idx, source=seg)

        place(rec_up, self.masks["upper"])
        place(rec_hd, self.masks["hands"])
        if rec_lo.shape[-1] == 57:
            place(rec_lo[..., :54], self.masks["lower"])  # rotations
            if D_out >= 333:
                # last 3 are translation
                out[..., 330:333] = rec_lo[..., 54:57]
        else:
            place(rec_lo, self.masks["lower"])  # 54-d only
        if rec_fc is not None and D_out >= 433:
            place(rec_fc, self.masks["face"])
        return out
