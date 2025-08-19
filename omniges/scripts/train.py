#!/usr/bin/env python
import os
import math
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omniges.models import OmniGesModel
from omniges.gesture_vae_adapter import ConcatRVQVAEAdapter
from dataloaders.beat_sep_lower import CustomDataset as BeatDataset
from utils.config import parse_args as parse_dataset_args
from utils.metric import L1div
from utils import other_tools_hf
import numpy as np
import os.path as osp
from types import SimpleNamespace
from utils import rotation_conversions as rc


def load_yaml(fp: str) -> Dict:
    with open(fp, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict):
    ds_cfg_path = cfg["dataset"].get("dataset_cfg", "configs/beat2_rvqvae.yaml")
    ds_args, _ = parse_dataset_args(ds_cfg_path)

    train_set = BeatDataset(ds_args, "train", build_cache=True)
    # val/test 분리는 CSV의 type 컬럼에 의존. 없으면 test 재사용
    try:
        val_set = BeatDataset(ds_args, "val", build_cache=False)
    except Exception:
        val_set = BeatDataset(ds_args, "test", build_cache=False)
    test_set = BeatDataset(ds_args, "test", build_cache=False)

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader


def build_model(cfg: Dict) -> OmniGesModel:
    audio_cfg = cfg["audio"]
    gesture_dims = cfg["gesture"]["code_dims"]
    model = OmniGesModel(
        dim_text=cfg["text"]["dim"],
        dim_wavlm=audio_cfg["dim_wavlm"],
        dim_other_audio=audio_cfg["dim_other"],
        dim_fuse=audio_cfg["fusion"]["dim_fuse"],
        num_heads=audio_cfg["fusion"]["num_heads"],
        num_layers=audio_cfg["fusion"]["num_layers"],
        gesture_code_dims=gesture_dims,
    )
    return model


def build_adapter(cfg: Dict, device: torch.device) -> ConcatRVQVAEAdapter:
    gcfg = cfg["gesture"]
    ck = gcfg["rvq_ckpt"]
    adapter = ConcatRVQVAEAdapter(
        upper_ckpt=ck["upper"],
        hands_ckpt=ck["hands"],
        lower_ckpt=ck["lower_trans"],
        face_ckpt=ck.get("face"),
        nb_code=cfg.get("nb_code", 512),
        code_dim=cfg.get("code_dim", 256),
        num_quantizers=cfg.get("num_quantizers", 6),
        fuse_in_channels=cfg["backbone"]["dim"] // cfg["backbone"]["heads"],
        sample_size=cfg.get("sample_size", 128),
        device=device,
    )
    return adapter


def train_one_epoch(model: OmniGesModel, adapter: ConcatRVQVAEAdapter, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, step0: int, log_interval: int) -> int:
    model.train()
    loss_rec = nn.SmoothL1Loss()
    step = step0
    for batch in loader:
        pose = batch["pose"].to(device).float()  # [B, T, D]
        # 어댑터 인코드→디코드 재구성 손실(Transformer는 동결, 어댑터 1x1 Conv 학습)
        z2d = adapter.encode(pose)
        rec = adapter.decode(z2d, T_out=pose.shape[1], D_out=pose.shape[2])
        loss = loss_rec(rec, pose)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step + 1) % log_interval == 0:
            print(f"[train] step={step+1} rec_loss={loss.item():.4f} rec_shape={rec.shape}")
        step += 1
    return step


@torch.no_grad()
def validate(model: OmniGesModel, adapter: ConcatRVQVAEAdapter, loader: DataLoader, device: torch.device):
    model.eval()
    div = L1div()
    for batch in loader:
        pose = batch["pose"].to(device).float()
        z2d = adapter.encode(pose)
        rec = adapter.decode(z2d, T_out=pose.shape[1], D_out=pose.shape[2])
        div.run(rec.detach().cpu().numpy().reshape(rec.shape[0], -1))
    print(f"[val] L1div={div.avg():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="omniges yaml config path")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    adapter = build_adapter(cfg, device)
    # 어댑터 1x1 Conv만 학습
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(adapter.proj_to_in.parameters()) + list(adapter.proj_from_in.parameters())
    optim = torch.optim.AdamW(params, lr=cfg["train"]["lr"])

    dry_run = cfg["train"].get("dry_run", False)
    steps_per_epoch = math.ceil(len(train_loader))
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        global_step = train_one_epoch(model, adapter, train_loader, optim, device, global_step, cfg["train"]["log_interval"])
        validate(model, adapter, val_loader, device)
        if dry_run:
            print("dry_run: stop early after first epoch")
            break

    # 테스트 및 렌더(초기 통합: GT 포즈로 파이프라인 확인)
    if cfg.get("eval", {}).get("render", False):
        print("Render pipeline check: using GT npz (placeholder)")
        batch = next(iter(test_loader))
        audio_name = batch.get("audio_name")
        if isinstance(audio_name, list):
            audio_name = audio_name[0]
        elif hasattr(audio_name, "__iter__"):
            audio_name = audio_name[0]
        audio_path = audio_name
        base_id = osp.splitext(osp.basename(audio_name))[0]
        data_path = cfg["dataset"]["data_path"]
        pose_rep = cfg["dataset"]["pose_rep"]
        gt_npz_path = osp.join(data_path, pose_rep, f"{base_id}.npz")
        out_dir = osp.join("outputs", "omniges_test")
        os.makedirs(out_dir, exist_ok=True)

        # other_tools_hf는 args에서 렌더 세팅을 읽음 → 간단한 네임스페이스로 전달
        eval_cfg = cfg.get("eval", {})
        args_ns = SimpleNamespace(
            render_video_fps=eval_cfg.get("render_video_fps", 30),
            render_video_width=eval_cfg.get("render_video_width", 1920),
            render_video_height=eval_cfg.get("render_video_height", 720),
            render_concurrent_num=eval_cfg.get("render_concurrent_num", 2),
            render_tmp_img_filetype=eval_cfg.get("render_tmp_img_filetype", "bmp"),
            debug=False,
        )
        # SMPL-X 모델 경로는 데이터셋 args 기준
        ds_cfg_path = cfg["dataset"].get("dataset_cfg", "configs/beat2_rvqvae.yaml")
        ds_args, _ = parse_dataset_args(ds_cfg_path)
        model_folder = osp.join(ds_args.data_path_1, "smplx_models/")

        try:
            clip_path = other_tools_hf.render_one_sequence_no_gt(
                res_npz_path=gt_npz_path,
                output_dir=out_dir,
                audio_path=audio_path,
                model_folder=model_folder,
                args=args_ns,
            )
            print(f"Rendered video: {clip_path}")
        except Exception as e:
            print(f"Render failed: {e}")


if __name__ == "__main__":
    main()
