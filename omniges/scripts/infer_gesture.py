#!/usr/bin/env python
"""
Gesture inference using OmniFlow + ConcatRVQVAEAdapter.
Generates gesture latents via flow denoising and decodes to poses; optional rendering.
"""
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import os.path as osp
import yaml
from typing import Dict

import torch

from omniflow.pipelines.omniflow_pipeline import OmniFlowPipeline
from omniges.gesture_vae_adapter import ConcatRVQVAEAdapter
from utils.config import parse_args as parse_dataset_args
from utils import other_tools_hf


def load_yaml(fp: str) -> Dict:
    with open(fp, "r") as f:
        return yaml.safe_load(f)


def build_adapter(cfg: Dict, device: torch.device) -> ConcatRVQVAEAdapter:
    gcfg = cfg["gesture"]
    ck = gcfg["rvq_ckpt"]
    return ConcatRVQVAEAdapter(
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


@torch.no_grad()
def run_denoise(pipe: OmniFlowPipeline, shape, guidance_scale: float = 1.0):
    device = pipe.transformer.device
    scheduler = pipe.scheduler
    latents = torch.randn(shape, device=device)
    scheduler.set_timesteps(scheduler.config.num_inference_steps, device=device)
    timesteps = scheduler.timesteps.to(device)

    # Minimal prompts
    # Use audio+text combined embeddings if possible; here we pass empty prompt and optional audio
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt_with_audio(
        prompt=[""],
        audio_paths=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=(guidance_scale > 1.0),
        use_t5=False,
        add_token_embed=False,
        max_sequence_length=128,
    )

    if guidance_scale > 1.0:
        # duplicate batch for CFG
        latents = latents.repeat(2, 1, 1, 1)

    for t in timesteps:
        latent_in = scheduler.scale_model_input(latents, t)
        model_out = pipe.transformer(
            hidden_states=latent_in,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        )
        pred = model_out["sample"] if isinstance(model_out, dict) and "sample" in model_out else model_out
        if guidance_scale > 1.0:
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        latents = scheduler.step(pred, t, latents[:pred.shape[0]]).prev_sample
    if guidance_scale > 1.0:
        latents = latents[: shape[0]]
    return latents


def maybe_render(rec_npz_path: str, audio_path: str, out_dir: str, cfg: Dict):
    args_ns = type("Args", (), dict(
        render_video_fps=cfg["eval"].get("render_video_fps", 30),
        render_video_width=cfg["eval"].get("render_video_width", 1280),
        render_video_height=cfg["eval"].get("render_video_height", 720),
        render_concurrent_num=cfg["eval"].get("render_concurrent_num", 1),
        render_tmp_img_filetype=cfg["eval"].get("render_tmp_img_filetype", "png"),
        debug=False,
    ))
    ds_args, _ = parse_dataset_args(cfg["dataset"]["dataset_cfg"])
    model_folder = osp.join(ds_args.data_path_1, "smplx_models/")
    clip_path = other_tools_hf.render_one_sequence_no_gt(
        res_npz_path=rec_npz_path,
        output_dir=out_dir,
        audio_path=audio_path,
        model_folder=model_folder,
        args=args_ns,
    )
    print(f"Rendered: {clip_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--out_dir", default="./outputs/omniges_infer")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--audio_path", default=None)
    ap.add_argument("--t_out", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(cfg, device)
    bundle = cfg.get("omniflow_pretrained", "./checkpoints/omniflow_pretrained")
    pipe = OmniFlowPipeline.load_pretrained(bundle, device=device, weight_dtype=torch.bfloat16, load_ema=False)
    pipe.scheduler.config.num_inference_steps = args.steps

    b = 1
    cin = adapter.fuse_in_channels
    s = adapter.sample_size
    latents_2d = run_denoise(pipe, (b, cin, s, s), guidance_scale=args.guidance_scale)

    T_out = args.t_out or cfg["train"].get("sequence_length", 64)
    D_out = 333
    pose = adapter.decode(latents_2d, T_out=T_out, D_out=D_out)

    os.makedirs(args.out_dir, exist_ok=True)
    rec_npz = osp.join(args.out_dir, "gesture_rec.npz")
    if args.save_npz or args.render:
        import numpy as np
        betas = np.zeros((1, 300), dtype=np.float32)
        poses6d = pose[0].cpu().numpy()
        np.savez(rec_npz, betas=betas, poses=poses6d, expressions=np.zeros((T_out, 100), dtype=np.float32), trans=np.zeros((T_out, 3), dtype=np.float32))
        print(f"Saved: {rec_npz}")
    if args.render and args.audio_path:
        maybe_render(rec_npz, args.audio_path, args.out_dir, cfg)


if __name__ == "__main__":
    main()
