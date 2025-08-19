#!/usr/bin/env python
"""
Minimal smoke test for gesture flow training.
- Builds adapter and pipeline
- Creates synthetic batch or takes 1 real batch if available
- Runs one optimizer step and prints loss
"""
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import yaml
from typing import Dict

from omniflow.pipelines.omniflow_pipeline import OmniFlowPipeline
from omniges.gesture_vae_adapter import ConcatRVQVAEAdapter
from utils.config import parse_args as parse_dataset_args
from dataloaders.beat_sep_lower import CustomDataset as BeatDataset
from torch.utils.data import DataLoader


def load_yaml(fp: str) -> Dict:
    with open(fp, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adapter and pipeline
    gcfg = cfg["gesture"]["rvq_ckpt"]
    adapter = ConcatRVQVAEAdapter(
        upper_ckpt=gcfg["upper"], hands_ckpt=gcfg["hands"], lower_ckpt=gcfg["lower_trans"], face_ckpt=gcfg.get("face"),
        fuse_in_channels=cfg["backbone"]["dim"] // cfg["backbone"]["heads"], sample_size=cfg.get("sample_size", 128), device=device,
    )
    bundle = cfg.get("omniflow_pretrained", "./checkpoints/omniflow_pretrained")
    pipe = OmniFlowPipeline.load_pretrained(bundle, device=device, weight_dtype=torch.bfloat16, load_ema=False)
    transformer = pipe.transformer.train()
    scheduler = pipe.scheduler

    # Try real batch else synthetic
    try:
        ds_args, _ = parse_dataset_args(cfg["dataset"]["dataset_cfg"])
        train_set = BeatDataset(ds_args, "train", build_cache=False)
        batch = next(iter(DataLoader(train_set, batch_size=1, shuffle=True)))
        pose = batch["pose"].to(device).float()
    except Exception:
        T_out, D_out = cfg["train"].get("sequence_length", 64), 333
        pose = torch.zeros((1, T_out, D_out), device=device)

    z0 = adapter.encode(pose)
    pe, pp = pipe.encode_prompt([""], device=device, num_images_per_prompt=1, do_classifier_free_guidance=False, use_t5=False)
    opt = torch.optim.AdamW(transformer.parameters(), lr=1e-5)
    # One step
    noise = torch.randn_like(z0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (z0.shape[0],), device=device).long()
    noisy = scheduler.add_noise(z0, noise, t)
    out = transformer(hidden_states=noisy, timestep=t, encoder_hidden_states=pe, pooled_projections=pp)
    pred = out["sample"] if isinstance(out, dict) and "sample" in out else out
    loss = torch.nn.functional.mse_loss(pred, noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print(f"Smoke OK. loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
