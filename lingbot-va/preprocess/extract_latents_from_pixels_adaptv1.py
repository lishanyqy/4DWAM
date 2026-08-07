#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read images from LeRobot-format video files
(videos/chunk-XXX/camera/episode_XXXXXX.mp4), extract latents with the
Wan2.2 VAE (diffusers AutoencoderKLWan) and UMT5 text encoder, and save them as
.pth files under latents/chunk-XXX/camera/episode_XXXXXX_start_end.pth.

Also supports automatically adding action_config to episodes.jsonl when it is missing.

Example input directory structure (--dataset-root):

your_dataset/
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000.mp4
│           └── ...
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── meta/
    └── episodes.jsonl

Output directory structure:

your_dataset/
├── latents/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_111.pth
│           └── ...
└── ...

Each .pth file contains a dictionary whose fields match the latent_bak format in the README:
    - latent, latent_num_frames, latent_height, latent_width
    - video_num_frames, video_height, video_width
    - text_emb, text
    - frame_ids, start_frame, end_frame
    - fps, ori_fps

Rules for automatically adding action_config when episodes.jsonl is missing it:
    - start_frame: 0
    - end_frame: episode length
    - action_text: tasks[0] (or the first element of the specified field)
    - skill: "" (empty string)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import io
import imageio

from torchvision.transforms.functional import resize
from diffusers import AutoencoderKLWan
from transformers import UMT5EncoderModel, T5TokenizerFast

# =================== Model path configuration ===================

# Output video size for each camera (H, W).
IMAGE_KEYS = {
    'observation.images.cam_high': (256, 320),
    'observation.images.cam_left_wrist': (128, 160),
    'observation.images.cam_right_wrist': (128, 160),
}


def build_wan2_2_components(models_root: Path, device: torch.device):
    """
    Load diffusers AutoencoderKLWan, transformers UMT5EncoderModel, and T5TokenizerFast.
    """
    vae_path = models_root / "vae"
    text_encoder_path = models_root / "text_encoder"
    tokenizer_path = models_root / "tokenizer"

    # VAE
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch.bfloat16,
    )
    vae = vae.to(device)
    vae.eval()

    # Text encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch.bfloat16,
    )
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    # Tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)

    return vae, text_encoder, tokenizer


@torch.no_grad()
def encode_video_with_vae(
    vae, video_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Encode video with diffusers AutoencoderKLWan.

    inputs:
        video_tensor: [T, 3, H, W], float32, [0,1]
    return:
        latents: [C, T_lat, H_lat, W_lat]
    """
    video_tensor = video_tensor.to(device=device, dtype=torch.bfloat16)
    # The diffusers VAE expects input with shape [B, C, T, H, W].
    video_batch = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]

    latent_dist = vae.encode(video_batch).latent_dist
    latents = latent_dist.sample()

    # Return [C, T_lat, H_lat, W_lat].
    return latents.squeeze(0)


@torch.no_grad()
def encode_text(
    text_encoder, tokenizer, text: str, device: torch.device, pad_length: int = 0
) -> torch.Tensor:
    """
    Encode action_text with UMT5EncoderModel.

    Returns:
        text_emb: [L, D] (bfloat16), with padding removed.
        If pad_length > 0, pad to the fixed length [pad_length, D].
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    output = text_encoder(**tokens)
    last_hidden = output.last_hidden_state  # [1, L, D]

    # Remove padding and keep only real tokens.
    attention_mask = tokens["attention_mask"][0]
    seq_len = attention_mask.sum().item()
    text_emb = last_hidden[0, :seq_len]  # [L, D]

    text_emb = text_emb.to(torch.bfloat16)

    # Optional: pad to a fixed length.
    if pad_length > 0 and text_emb.shape[0] < pad_length:
        pad = torch.zeros(pad_length - text_emb.shape[0], text_emb.shape[1],
                          dtype=text_emb.dtype, device=text_emb.device)
        text_emb = torch.cat([text_emb, pad], dim=0)

    return text_emb


# ============================ Utility functions ============================


def load_episodes_meta(meta_path: Path) -> Dict[int, Dict[str, Any]]:
    """Read meta/episodes.jsonl -> {episode_index: meta_dict}."""
    mapping: Dict[int, Dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["episode_index"])
            mapping[idx] = obj
    return mapping


def get_chunk_name(episode_index: int, episodes_per_chunk: int) -> str:
    """Infer the chunk name from episode_index."""
    chunk_id = episode_index // episodes_per_chunk
    return f"chunk-{chunk_id:03d}"


def sample_frames(
    total_len: int, start_frame: int, end_frame: int, ori_fps: int, target_fps: int, max_frames: int = 0
) -> List[int]:
    """
    Sample frame indices in [start_frame, end_frame) at the target fps.
    Use uniform interval sampling with step = ori_fps // target_fps.
    If max_frames > 0, sample at most max_frames frames.
    """
    start_frame = max(0, start_frame)
    end_frame = min(total_len, end_frame)
    if end_frame <= start_frame:
        return []

    if target_fps <= 0 or ori_fps <= 0:
        return list(range(start_frame, end_frame))

    step = max(1, ori_fps // target_fps)
    ids = list(range(start_frame, end_frame, step))

    if max_frames > 0 and len(ids) > max_frames:
        ids = ids[:max_frames]

    return ids


def read_episode_video_frames(
    video_path: Path,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Read all frames from an mp4 video.

    Returns:
        video: [T, 3, H, W], float32, [0,1]
        T, H, W
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        # frame: numpy [H, W, 3], uint8
        frames.append(frame)
    reader.close()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # Stack and convert to tensor [T, H, W, 3]
    video_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    video = torch.from_numpy(video_np)  # [T, H, W, 3]
    # Permute to [T, 3, H, W]
    video = video.permute(0, 3, 1, 2).contiguous()

    T, _, H, W = video.shape
    return video, T, H, W


def add_action_config_to_episodes(
    dataset_root: Path,
    action_text_source: str = "tasks",
    action_text_index: int = 0,
    skill: str = "",
    dry_run: bool = False,
) -> int:
    """
    Add action_config to episodes in episodes.jsonl that are missing the field.

    Rules:
    - Skip episodes that already have action_config.
    - Otherwise add action_config = [{
        "start_frame": 0,
        "end_frame": length,
        "action_text": tasks[action_text_index] (or the field specified by action_text_source),
        "skill": skill,
    }]

    Args:
        dataset_root: dataset root directory containing meta/episodes.jsonl
        action_text_source: field to read action_text from; defaults to "tasks"
        action_text_index: index in the tasks list to read; defaults to 0
        skill: value for the skill field; defaults to an empty string
        dry_run: if True, only print the planned changes without writing

    Returns:
        Number of modified episodes.
    """
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        print(f"[WARN] episodes.jsonl not found at {meta_path}, skip add_action_config.")
        return 0

    lines_out = []
    modified_count = 0

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                lines_out.append(line)
                continue
            obj = json.loads(line)

            if "action_config" in obj:
                # Keep existing action_config unchanged.
                lines_out.append(json.dumps(obj, ensure_ascii=False))
                continue

            length = int(obj.get("length", 0))
            tasks = obj.get("tasks", [])

            # Get action_text.
            if action_text_source == "tasks" and tasks:
                action_text = tasks[action_text_index] if action_text_index < len(tasks) else tasks[0]
            else:
                action_text = obj.get(action_text_source, "")

            action_config = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": action_text,
                "skill": skill,
            }]
            obj["action_config"] = action_config
            lines_out.append(json.dumps(obj, ensure_ascii=False))
            modified_count += 1

    if modified_count > 0 and not dry_run:
        # Back up the original file.
        backup_path = meta_path.with_suffix(".jsonl.bak")
        if not backup_path.exists():
            meta_path.rename(backup_path)
            print(f"[INFO] backed up original to {backup_path}")

        with meta_path.open("w", encoding="utf-8") as f:
            for line in lines_out:
                f.write(line + "\n")

    if dry_run:
        print(f"[DRY-RUN] Would modify {modified_count} episodes in {meta_path}")
    else:
        print(f"[INFO] modified {modified_count} episodes in {meta_path}")

    return modified_count


def process_dataset(
    dataset_root: Path,
    models_root: Path,
    image_column: str = "observation.images.cam_high",
    episodes_per_chunk: int = 500,
    ori_fps: int = 50,
    target_fps: int = 12,
    max_frames: int = 0,
    text_length: int = 128,
    device_str: str = "cuda:0",
    auto_add_action_config: bool = True,
):
    device = torch.device(device_str)

    # Automatically add action_config if it is missing.
    if auto_add_action_config:
        add_action_config_to_episodes(dataset_root)

    # Load models.
    vae, text_encoder, tokenizer = build_wan2_2_components(models_root, device)

    # Read episode metadata.
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {meta_path}")
    episodes_meta = load_episodes_meta(meta_path)

    # Video root and output directory.
    videos_root = dataset_root / "videos"
    latents_root = dataset_root / "latents"
    latents_root.mkdir(parents=True, exist_ok=True)

    # Get the target size for the current camera.
    target_size = IMAGE_KEYS.get(image_column)
    if target_size is None:
        raise KeyError(f"Unknown image_column: {image_column}")
    target_h, target_w = target_size

    for episode_index, meta in episodes_meta.items():
        length = int(meta["length"])
        tasks = meta.get("tasks", [])
        action_configs = meta.get("action_config", [])

        if not action_configs:
            # If action_config is missing, default to the whole episode.
            action_configs = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": tasks[0] if tasks else "",
            }]

        default_text = tasks[0] if tasks else ""
        chunk_name = get_chunk_name(episode_index, episodes_per_chunk)

        # Video input path.
        video_path = (
            videos_root / chunk_name / image_column / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            print(f"[WARN] missing episode video: {video_path}, skip.")
            continue

        try:
            video, T, H, W = read_episode_video_frames(video_path)
        except Exception as e:
            print(f"[WARN] failed to read {video_path}: {e}")
            continue

        # Resize to the target size.
        video_resized = resize(video, (target_h, target_w))  # [T, 3, H, W]

        # Output directory.
        out_dir = latents_root / chunk_name / image_column
        out_dir.mkdir(parents=True, exist_ok=True)

        for seg_id, segment in enumerate(action_configs):
            start_frame = int(segment.get("start_frame", 0))
            end_frame = int(segment.get("end_frame", length))
            action_text = segment.get("action_text", default_text)

            frame_ids = sample_frames(
                total_len=length,
                start_frame=start_frame,
                end_frame=end_frame,
                ori_fps=ori_fps,
                target_fps=target_fps,
                max_frames=max_frames,
            )
            if not frame_ids:
                print(
                    f"[WARN] episode {episode_index:06d} seg#{seg_id} "
                    f"[{start_frame},{end_frame}) -> empty frame_ids, skip."
                )
                continue

            # Sample frames.
            frames = video_resized[frame_ids]  # [N, 3, H, W]
            N, _, H_res, W_res = frames.shape

            # VAE encoding.
            latents = encode_video_with_vae(vae, frames, device=device)
            # latents: [C, T_lat, H_lat, W_lat]

            C_lat, T_lat, H_lat, W_lat = latents.shape

            # Convert to bfloat16 and flatten into [T_lat * H_lat * W_lat, C_lat].
            latents_bf16 = latents.to(dtype=torch.bfloat16)
            latent_flat = latents_bf16.permute(1, 2, 3, 0).reshape(-1, C_lat)

            # Text encoding.
            text_emb = encode_text(
                text_encoder, tokenizer, action_text, device=device, pad_length=text_length
            )
            text_emb_bf16 = text_emb.to(dtype=torch.bfloat16)
            text_emb_n, text_emb_d = text_emb_bf16.shape

            if text_emb_d != 4096:
                print(f"[WARN] text_emb_d is {text_emb_d}, expected 4096")

            # Build the output dictionary.
            row = {
                "latent": latent_flat.detach().cpu(),
                "latent_num_frames": int(T_lat),
                "latent_height": int(H_lat),
                "latent_width": int(W_lat),

                "video_num_frames": int(N),
                "video_height": int(H_res),
                "video_width": int(W_res),

                "text_emb": text_emb_bf16,
                "text": action_text,

                "frame_ids": [int(x) for x in frame_ids],
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "fps": int(target_fps),
                "ori_fps": int(ori_fps),
            }

            out_path = out_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            torch.save(row, out_path)
            print(f"[OK] saved latents: {out_path} (frames={N}, latent={latent_flat.shape})")

    print("All done.")


def main():
    '''
    Example:
    
    python preprocess/extract_latents_from_pixels0710.py \
        --dataset-root dataset_root \
        --models-root encoder_model_path \
        --max-frames 30 \
        --cameras view1,view2,view3
    '''
    # python preprocess/extract_latents_from_pixels0710.py --dataset-root os.environ['HOME']/dataspace/robotwin_samples/lift_pot-demo_clean_collect_200-50
    # --models-root os.environ['HOME']/modelspace/lingbot-va-base
    # --max-frames 30
    home_path = '~/lishan'
    parser = argparse.ArgumentParser(
        description="Extract Wan2.2 latents from LeRobot video episodes into .pth latent files."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(home_path, 'dataspace/robotwin_samples/lift_pot-demo_clean_collect_200-50'),
        help="Dataset directory (contains videos/, data/, meta/episodes.jsonl)",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=os.path.join(home_path, 'modelspace/lingbot-va-base'),
        help="Model directory (contains vae/, text_encoder/, tokenizer/)",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=500,
        help="Number of episodes per chunk",
    )
    parser.add_argument(
        "--ori-fps",
        type=int,
        default=50,
        help="Original video fps",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Target sampling fps for latent extraction",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to sample per segment (0=unlimited)",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=128,
        help="Text max length (tokenizer param, mostly unused with T5TokenizerFast)",
    )
    parser.add_argument(
        "--auto-add-action-config",
        action="store_true",
        default=True,
        help="Automatically add action_config to episodes.jsonl if missing (default: True)",
    )
    parser.add_argument(
        "--no-auto-add-action-config",
        action="store_false",
        dest="auto_add_action_config",
        help="Disable automatic addition of action_config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device: cuda:0 / cuda:1 / cpu",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated camera names to process (default: all in IMAGE_KEYS)",
    )

    args = parser.parse_args()

    cameras = args.cameras.split(",") if args.cameras else list(IMAGE_KEYS.keys())

    for key in cameras:
        if key not in IMAGE_KEYS:
            print(f"[WARN] Unknown camera '{key}', skipping.")
            continue
        print(f"\n=== Processing camera: {key} ===")
        process_dataset(
            dataset_root=Path(args.dataset_root),
            models_root=Path(args.models_root),
            image_column=key,
            episodes_per_chunk=args.episodes_per_chunk,
            ori_fps=args.ori_fps,
            target_fps=args.fps,
            max_frames=args.max_frames,
            text_length=args.text_length,
            device_str=args.device,
            auto_add_action_config=args.auto_add_action_config,
        )


if __name__ == "__main__":
    main()
