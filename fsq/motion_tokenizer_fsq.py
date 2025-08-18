from typing import Any, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np
import os
import scipy
import wandb

from src.models.components.fsq_tokenizer import MotionTokenizerFSQ
from src.utils.viz_util import generate_bvh, render_bvh, unnormalize, make_side_by_side_video, render_motion_to_video, parse_output_vec


class MotionTokenizerFSQLitModule(LightningModule):
    """Lightning module to train FSQ motion tokenizer with reconstruction loss.

    This trains the frame-wise encoder/decoder end-to-end with straight-through quantization.
    """

    def __init__(
        self,
        optimizer,
        chunk_size,
        pose_dim,
        motion_fps,
        n_joints,
        data_norm_stat_path,
        normalization_method,
        hidden_dim: int = 256,
        l1_weight: float = 1.0,
        vel_weight: float = 0.5,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.pose_dim = pose_dim
        self.chunk_size = chunk_size
        self.motion_fps = motion_fps

        self.tokenizer = MotionTokenizerFSQ(pose_dim, code_dim=hidden_dim)

        self.optimizer_cfg = optimizer
        self.l1_weight = l1_weight
        self.vel_weight = vel_weight
        # normalization stats for visualization
        self.data_stat = None
        if data_norm_stat_path:
            self.data_stat = np.load(data_norm_stat_path)
            self.norm_method = normalization_method

        self._val_vis_cache = None

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        # Expect a N-frame chunk; return reconstructed window
        pose_hat, _, _, _, _ = self.tokenizer.forward(pose_seq)
        return pose_hat

    def configure_optimizers(self):
        lr = self.hparams.optimizer.lr
        optimizer = torch.optim.AdamW(self.tokenizer.parameters(), lr=lr)

        warmup_e = int(getattr(self.hparams, 'warmup_epochs', 0) or 0)

        def lr_lambda(epoch: int):
            # piecewise: warmup then linear decay to 0
            max_epochs = getattr(self.trainer, 'max_epochs', None) or 1
            if warmup_e > 0 and epoch < warmup_e:
                return float(epoch + 1) / float(max(1, warmup_e))
            remain = max(1, max_epochs - warmup_e)
            progress = min(max(0, epoch - warmup_e), remain) / float(remain)
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: Any, batch_idx: int):
        pose_seq, audio, aux_info = batch
        B, T, D = pose_seq.shape
        n = self.chunk_size
        assert T >= n, f"sequence length {T} must be >= chunk_size {n}"

        max_start = T - n + 1
        num_chunks = max(1, T // n)

        if max_start <= num_chunks:
            starts = torch.arange(max_start, device=pose_seq.device)
        else:
            starts = torch.randperm(max_start, device=pose_seq.device)[:num_chunks]
            starts, _ = torch.sort(starts)

        # Same start indices for every sequence in the batch
        chunk_list = [pose_seq[:, int(s.item()):int(s.item()) + n, :] for s in starts]
        chunks = torch.stack(chunk_list, dim=1)        # (B, num_chunks, n, D)
        chunks = chunks.view(B * len(starts), n, D)    # (B*num_chunks, n, D)

        pose_hat, q_loss, perplexity, activate, _ = self.tokenizer.forward(chunks)
        target_win = chunks

        # Align lengths by center-cropping to the same length
        Lp = pose_hat.size(1)
        Lt = target_win.size(1)
        L = min(Lp, Lt)
        sp = (Lp - L) // 2
        st = (Lt - L) // 2
        pose_hat_c = pose_hat[:, sp:sp+L]
        target_c = target_win[:, st:st+L]

        loss_l1 = F.l1_loss(pose_hat_c, target_c)
        # Velocity loss within each window (aligned)
        vel_hat = pose_hat_c[:, 1:] - pose_hat_c[:, :-1]
        vel_tgt = target_c[:, 1:] - target_c[:, :-1]
        loss_vel = F.l1_loss(vel_hat, vel_tgt)
        loss = self.l1_weight * loss_l1 + self.vel_weight * loss_vel

        self.log_dict({
            'train/loss': loss,
            'train/l1': loss_l1,
            'train/vel': loss_vel,
            'train/perplexity': perplexity.detach(),
            'train/activate': activate.detach(),
        }, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        pose_seq, audio, aux_info = batch
        B, T, D = pose_seq.shape
        n = self.chunk_size
        assert T >= n, f"sequence length {T} must be >= chunk_size {n}"
        chunks = torch.stack([pose_seq[:, i:i+n, :] for i in range(T - n + 1)], dim=1).view(B * (T - n + 1), n, D)
        pose_hat, q_loss, perplexity, activate, indices = self.tokenizer.forward(chunks)

        # Align lengths by center-cropping to the same length
        Lp = pose_hat.size(1)
        Lt = chunks.size(1)
        L = min(Lp, Lt)
        sp = (Lp - L) // 2
        st = (Lt - L) // 2
        pose_hat_c = pose_hat[:, sp:sp+L]
        chunks_c = chunks[:, st:st+L]

        loss_l1 = F.l1_loss(pose_hat_c, chunks_c)
        vel_hat = pose_hat_c[:, 1:] - pose_hat_c[:, :-1]
        vel_tgt = chunks_c[:, 1:] - chunks_c[:, :-1]
        loss_vel = F.l1_loss(vel_hat, vel_tgt)
        loss = self.l1_weight * loss_l1 + self.vel_weight * loss_vel

        self.log_dict({
            'val/loss': loss,
            'val/l1': loss_l1,
            'val/vel': loss_vel,
            'val/perplexity': perplexity.detach(),
            'val/activate': activate.detach(),
        }, prog_bar=True)
        # cache first batch for visualization
        if batch_idx == 0:
            with torch.no_grad():
                # Frame-level overlap-add
                K = 32  # windows size
                H = 24  # hop
                x = pose_seq[0]  # (T, D)
                Tlen = x.size(0)
                device = x.device
                acc = torch.zeros(Tlen, D, device=device)
                wsum = torch.zeros(Tlen, 1, device=device)

                if Tlen <= K:
                    starts = [0]
                else:
                    starts = list(range(0, Tlen - K + 1, H))
                    if starts[-1] != Tlen - K:
                        starts.append(Tlen - K)

                # precompute triangular weights of length K
                idx = torch.arange(K, device=device, dtype=torch.float32)
                base_w = 1.0 - (idx - (K - 1) / 2).abs() / ((K + 1) / 2)
                base_w = base_w.clamp_min(0.0).unsqueeze(1)  # (K,1)

                for s in starts:
                    e = s + K
                    chunk = x[s:e].unsqueeze(0)  # (1, K, D)
                    yhat, *_ = self.tokenizer.forward(chunk)  # (1, K, D)
                    yhat = yhat.squeeze(0)
                    w = base_w
                    acc[s:e] += w * yhat
                    wsum[s:e] += w

                recon_full = acc / (wsum + 1e-8)

            self._val_vis_cache = {
                'pose_seq': pose_seq.detach().cpu(),
                'recon_full': recon_full.detach().cpu(),
                'audio': audio.detach().cpu(),
                'aux_info': aux_info,
            }
        return loss

    def on_validation_epoch_end(self):
        if self._val_vis_cache is None:
            return
        cache = self._val_vis_cache
        self._val_vis_cache = None

        output_dir = self.trainer.default_root_dir
        os.makedirs(output_dir, exist_ok=True)

        pose_seq = cache['pose_seq'][0]
        recon_full = cache['recon_full']
        audio = cache['audio'][0].numpy()
        aux_info = cache['aux_info']

        # use full-length GT and reconstruction
        gt_center = pose_seq

        # unnormalize
        if self.data_stat is not None:
            gt_npy = unnormalize(gt_center.numpy(), self.norm_method, self.data_stat)
            recon_npy = unnormalize(recon_full.numpy(), self.norm_method, self.data_stat)
        else:
            gt_npy = gt_center.numpy()
            recon_npy = recon_full.numpy()

        wav_path = os.path.join(output_dir, f'sample_epoch_{self.current_epoch}.wav')
        scipy.io.wavfile.write(wav_path, 16000, audio)

        title = str(aux_info['vid'][0]) if isinstance(aux_info, dict) and 'vid' in aux_info else f'epoch {self.current_epoch}'

        gt_bvh = os.path.join(output_dir, f'gt_epoch_{self.current_epoch}.bvh')
        rc_bvh = os.path.join(output_dir, f'rec_epoch_{self.current_epoch}.bvh')
        gt_root, _, gt_rot = parse_output_vec(gt_npy, self.hparams.n_joints)
        rc_root, _, rc_rot = parse_output_vec(recon_npy, self.hparams.n_joints)
        generate_bvh(gt_root, gt_rot, gt_bvh, 'data_beat2/pymo_pipe.sav')
        generate_bvh(rc_root, rc_rot, rc_bvh, 'data_beat2/pymo_pipe.sav')
        gt_mp4 = render_bvh(gt_bvh, wav_path, f'GT {title}', out_path=output_dir, fps=self.motion_fps)
        rc_mp4 = render_bvh(rc_bvh, wav_path, f'Recon {title}', out_path=output_dir, fps=self.motion_fps)

        merged = make_side_by_side_video(gt_mp4, rc_mp4, output_dir, f'tokenizer_epoch_{self.current_epoch}.mp4')
        if self.trainer.logger is not None:
            wandb.log({"val/tokenizer_video": wandb.Video(merged, fps=15, format="mp4")})
